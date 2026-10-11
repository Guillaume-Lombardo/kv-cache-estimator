import { isMoE, storedTokens, weightSupport, type Estimate, type EstimateInput } from "./calc";
import type { GpuSpec, PerformanceSettings, WeightFormat } from "./config-schema";

/**
 * Estimation de débit en décodage, au premier ordre :
 * temps d'une étape = max(octets lus / bande passante, FLOPs / débit de calcul) + latence des collectives.
 * Les octets lus sont les poids touchés (experts activés pour un MoE) et le KV des requêtes du batch.
 */

export interface DecodePoint {
  batch: number;
  stepMs: number;
  tokPerSecPerSession: number;
  tokPerSecTotal: number;
  bound: "mémoire" | "calcul";
}

export interface PerfEstimate {
  /** Une seule session active. */
  single: DecodePoint;
  /** Toutes les sessions possibles actives. */
  full: DecodePoint | null;
  /** Prefill d'un contexte complet pour une requête seule, en secondes. */
  prefillSeconds: number;
  computeLabel: string;
  /** Point de fonctionnement pour un nombre quelconque de sessions actives. */
  at: (batch: number) => DecodePoint;
}

/** Débit tensor core utilisé selon le format des poids (FP4 émulé → BF16). */
export function computeTflops(gpu: GpuSpec, wf: WeightFormat, emulated: boolean): { tflops: number; label: string } {
  const t = gpu.tflops;
  if (!emulated && wf.compute === "fp4" && t.fp4) return { tflops: t.fp4, label: "FP4" };
  if (!emulated && (wf.compute === "fp4" || wf.compute === "fp8") && t.fp8) return { tflops: t.fp8, label: "FP8" };
  return { tflops: t.bf16, label: gpu.id === "v100" ? "FP16" : "BF16" };
}

/** FLOPs de décodage par token : 2 × paramètres actifs + attention sur le contexte retenu. */
function decodeFlopsPerToken(input: EstimateInput): number {
  const { model: m, contextTokens: ctx, engine } = input;
  const active = (m.activeParamsB ?? m.paramsB) * 1e9;
  let attn = 0;
  for (const g of m.layers) {
    if (g.kind === "full" || g.kind === "sliding") {
      attn += 2 * g.count * m.attentionHeads * (g.headDim + (g.headDimV ?? g.headDim)) * storedTokens(g, ctx, engine);
    } else if (g.kind === "mla") {
      // MLA absorbé : scores et sorties dans l'espace latent.
      attn += 2 * g.count * m.attentionHeads * (2 * g.kvLoraRank + g.ropeHeadDim) * ctx;
    } else {
      attn += 4 * g.count * g.ssmElements;
    }
  }
  return 2 * active + attn;
}

/** FLOPs du prefill d'une requête de ctx tokens (attention causale). */
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
      // Prefill non absorbé : têtes K (nope + rope) et V, ~256 + rope.
      attn += 2 * g.count * m.attentionHeads * (256 + g.ropeHeadDim) * ((ctx * ctx) / 2);
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
  /** Poids lus pour un batch de b séquences : les experts touchés croissent avec le batch. */
  const weightsRead = (bytes: number, b: number) => (moe ? bytes * Math.min(1, Math.max(a, 1 - Math.pow(1 - a, b))) : bytes);
  const flopsTok = decodeFlopsPerToken(input);
  const layers = m.layers.reduce((s, g) => s + g.count, 0);

  const point = (batch: number): DecodePoint => {
    let stepS: number;
    let memS: number;
    let compS: number;
    if (mode === "dp-ep") {
      const perGpuSeqs = Math.ceil(batch / n);
      const attn = est.weights.attentionBytes;
      memS = (attn + weightsRead(W - attn, batch) / n + perGpuSeqs * est.kv.perRequestPerGpuBytes) / bytesPerSec;
      compS = (batch * flopsTok) / n / flopsPerSec;
      stepS = Math.max(memS, compS) + (2 * layers * collectiveLatencyUs(gpu, n, perf)) / 1e6;
    } else {
      const { tp, pp } = est;
      const micro = Math.ceil(batch / pp);
      memS = (weightsRead(W, micro) / n + micro * est.kv.perRequestPerGpuBytes) / bytesPerSec;
      compS = (micro * flopsTok) / n / flopsPerSec;
      const comm = (2 * (layers / pp) * collectiveLatencyUs(gpu, tp, perf) + (pp > 1 ? collectiveLatencyUs(gpu, n, perf) : 0)) / 1e6;
      // Chaque étape de décodage traverse les pp étages ; chaque étage traite pp micro-batchs.
      stepS = pp * (Math.max(memS, compS) + comm);
    }
    return {
      batch,
      stepMs: stepS * 1000,
      tokPerSecPerSession: 1 / stepS,
      tokPerSecTotal: batch / stepS,
      bound: memS >= compS ? "mémoire" : "calcul",
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
