import { isMoE, storedTokens, weightSupport, type Estimate, type EstimateInput } from "./calc";
import type { GpuSpec, PerformanceSettings, WeightFormat } from "./config-schema";

/**
 * First-order decoding throughput estimate:
 * step time = max(bytes read / bandwidth, FLOPs / compute throughput) + collective latency.
 * Bytes read include accessed weights (active experts for MoE) and the batch's KV cache.
 */

export interface DecodePoint {
  batch: number;
  stepMs: number;
  tokPerSecPerSession: number;
  tokPerSecTotal: number;
  bound: "memory" | "compute";
}

export interface PerfEstimate {
  /** One active session. */
  single: DecodePoint;
  /** All available sessions active. */
  full: DecodePoint | null;
  /** Full-context prefill for one request, in seconds. */
  prefillSeconds: number;
  computeLabel: string;
  /** Operating point for an arbitrary number of active sessions. */
  at: (batch: number) => DecodePoint;
}

/** Tensor-core throughput for the weight format (emulated FP4 uses BF16). */
export function computeTflops(gpu: GpuSpec, wf: WeightFormat, emulated: boolean): { tflops: number; label: string } {
  const t = gpu.tflops;
  if (!emulated && wf.compute === "fp4" && t.fp4) return { tflops: t.fp4, label: "FP4" };
  if (!emulated && (wf.compute === "fp4" || wf.compute === "fp8") && t.fp8) return { tflops: t.fp8, label: "FP8" };
  return { tflops: t.bf16, label: gpu.id === "v100" ? "FP16" : "BF16" };
}

/** Decoding FLOPs per token: 2 x active parameters plus attention over the retained context. */
function decodeFlopsPerToken(input: EstimateInput): number {
  const { model: m, contextTokens: ctx, engine } = input;
  const active = (m.activeParamsB ?? m.paramsB) * 1e9;
  let attn = 0;
  for (const g of m.layers) {
    if (g.kind === "full" || g.kind === "sliding") {
      attn += 2 * g.count * m.attentionHeads * (g.headDim + (g.headDimV ?? g.headDim)) * storedTokens(g, ctx, engine);
    } else if (g.kind === "mla") {
      // Absorbed MLA: scores and outputs in latent space.
      attn += 2 * g.count * m.attentionHeads * (2 * g.kvLoraRank + g.ropeHeadDim) * ctx;
    } else if (g.kind === "compressed") {
      // Shared K = V entries: compressed entries plus the uncompressed window.
      attn += 4 * g.count * m.attentionHeads * g.headDim * storedTokens(g, ctx, engine);
    } else {
      attn += 4 * g.count * g.ssmElements;
    }
  }
  return 2 * active + attn;
}

/** Prefill FLOPs for a request of ctx tokens (causal attention). */
function prefillFlops(input: EstimateInput): number {
  const { model: m, contextTokens: ctx, engine } = input;
  const active = (m.activeParamsB ?? m.paramsB) * 1e9;
  let attn = 0;
  for (const g of m.layers) {
    if (g.kind === "full" || g.kind === "sliding") {
      const kept = storedTokens(g, ctx, engine);
      const pairs = kept < ctx ? ctx * kept : (ctx * ctx) / 2;
      attn += 2 * g.count * m.attentionHeads * (g.headDim + (g.headDimV ?? g.headDim)) * pairs;
    } else if (g.kind === "mla") {
      // Unabsorbed prefill: K (nope + rope) and V heads, approximately 256 + rope.
      attn += 2 * g.count * m.attentionHeads * (256 + g.ropeHeadDim) * ((ctx * ctx) / 2);
    } else if (g.kind === "compressed") {
      attn += 4 * g.count * m.attentionHeads * g.headDim * ((ctx * storedTokens(g, ctx, engine)) / 2);
    } else {
      attn += 4 * g.count * g.ssmElements * ctx;
    }
  }
  return 2 * active * ctx + attn;
}

function collectiveLatencyUs(gpu: GpuSpec, groupSize: number, perf: PerformanceSettings): number {
  if (groupSize <= 1) return 0;
  if (gpu.interconnect.nvlink && groupSize <= gpu.nvlinkDomain) return perf.collectiveLatencyUs.nvlink;
  if (groupSize <= gpu.maxGpusPerNode) return gpu.interconnect.nvlink ? perf.collectiveLatencyUs.nvlink : perf.collectiveLatencyUs.pcie;
  return perf.collectiveLatencyUs.network;
}

export function estimatePerf(input: EstimateInput, est: Estimate, perf: PerformanceSettings): PerfEstimate {
  const { model: m, gpu, engine, gpuCount: n, weightFormat, mode } = input;
  const sup = weightSupport(gpu, engine, weightFormat.id);
  const { tflops, label } = computeTflops(gpu, input.memoryWeightFormat ?? weightFormat, sup.level === "limited");
  const flopsPerSec = tflops * 1e12 * perf.computeEfficiency;
  const bytesPerSec = gpu.bandwidthTBs * 1e12 * perf.memoryEfficiency;
  const W = est.weights.totalBytes;
  const moe = isMoE(m);
  const a = moe ? (m.activeParamsB as number) / m.paramsB : 1;
  /** Weights read for a batch of b sequences: accessed experts increase with batch size. */
  const weightsRead = (bytes: number, b: number) => (moe ? bytes * Math.min(1, Math.max(a, 1 - Math.pow(1 - a, b))) : bytes);
  const flopsTok = decodeFlopsPerToken(input);
  const layers = m.layers.reduce((s, g) => s + g.count, 0);

  const point = (batch: number): DecodePoint => {
    let stepS: number;
    let memS: number;
    let compS: number;
    if (mode === "dp-ep") {
      const perGpuSeqs = Math.ceil(batch / n);
      const shared = est.weights.nonExpertBytes;
      memS = (shared + weightsRead(W - shared, batch) / n + perGpuSeqs * est.kv.perRequestPerGpuBytes) / bytesPerSec;
      compS = (batch * flopsTok) / n / flopsPerSec;
      stepS = Math.max(memS, compS) + (2 * layers * collectiveLatencyUs(gpu, n, perf)) / 1e6;
    } else {
      const { tp, pp } = est;
      const micro = Math.ceil(batch / pp);
      memS = ((weightsRead(W, micro) / W) * est.weights.perGpuBytes + micro * est.kv.perRequestPerGpuBytes) / bytesPerSec;
      compS = (micro * flopsTok) / n / flopsPerSec;
      const comm = (2 * (layers / pp) * collectiveLatencyUs(gpu, tp, perf) + (pp > 1 ? collectiveLatencyUs(gpu, n, perf) : 0)) / 1e6;
      // Each decoding step traverses pp stages; each stage handles pp microbatches.
      stepS = pp * (Math.max(memS, compS) + comm);
    }
    return {
      batch,
      stepMs: stepS * 1000,
      tokPerSecPerSession: 1 / stepS,
      tokPerSecTotal: batch / stepS,
      bound: memS >= compS ? "memory" : "compute",
    };
  };

  const prefillSeconds = prefillFlops(input) / (n * flopsPerSec);
  return {
    single: point(1),
    full: est.sessions >= 1 ? point(est.sessions) : null,
    prefillSeconds,
    computeLabel: label,
    at: point,
  };
}
