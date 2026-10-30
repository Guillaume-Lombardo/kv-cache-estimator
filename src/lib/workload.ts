import { collectIssues, estimate, runtimeDefaults, type EstimateInput, type ParallelMode } from "./calc";
import type { AppConfig, GpuSpec } from "./config-schema";
import { tr } from "./i18n";
import { estimatePerf, prefillFlops, type PerfEstimate } from "./perf";
import { formatsFor, gpuCountOptions, instanceInputs, type SizingBase } from "./planning";
import type { Workload } from "./scenario";

/**
 * Workload analysis: steady-state load of one serving instance (the selected GPUs) for an average request
 * of `inputTokens` + `outputTokens` arriving at `requestsPerSec`, against TTFT and ITL targets.
 *
 * First-order model, no queueing theory:
 * - KV memory: each resident request holds on average input + output / 2 tokens (paged KV);
 *   concurrency is capped by that capacity and by the engine's maxNumSeqs.
 * - Prefills share the instance's compute with decoding (chunked prefill): they use a fraction
 *   u = rate × prefill FLOPs / instance compute, which slows compute-bound steps by 1 / (1 - u).
 * - TTFT = single-request prefill time / (1 - u); ITL = decoding step time at the resulting concurrency.
 * - Concurrency follows Little's law: rate × (TTFT + output tokens × ITL), solved by fixed point.
 * Replicas: identical instances behind a router (llm-d) share the rate evenly.
 */

export interface InstanceLoad {
  requestsPerSec: number;
  /** Requests resident at once (prefilling or decoding). */
  concurrency: number;
  /** Concurrency allowed by KV memory and maxNumSeqs at the average request length. */
  capacity: number;
  /** Fraction of the instance's compute taken by prefills. */
  prefillUtil: number;
  ttftSeconds: number;
  itlMs: number;
  outputTokPerSec: number;
  /** Limits exceeded at this rate. */
  over: { memory: boolean; prefill: boolean; ttft: boolean; itl: boolean };
  ok: boolean;
}

export interface InstanceModel {
  input: EstimateInput;
  /** Average resident tokens per request. */
  avgTokens: number;
  capacity: number;
  perf: PerfEstimate;
  prefillFlopsPerRequest: number;
  /** Prefill time of one prompt on an idle instance, in seconds. */
  singlePrefillSeconds: number;
  /** KV of one prompt across the instance (with replication), in bytes: transferred in P/D disaggregation. */
  promptKvBytes: number;
  /** Prompts whose KV fits at once (prefill pool). */
  promptCapacity: number;
}

export type Infeasible = "invalid-config" | "request-exceeds-context" | "no-memory" | "ttft-unreachable" | "itl-unreachable";

export function instanceModel(cfg: AppConfig, input: EstimateInput, w: Workload): InstanceModel {
  const n = input.gpuCount;
  const avgTokens = Math.max(1, Math.round(w.inputTokens + w.outputTokens / 2));
  const avgInput = { ...input, contextTokens: avgTokens };
  const est = estimate(avgInput);
  const perf = estimatePerf(avgInput, est, cfg.performance);
  const schedulers = input.mode === "dp-ep" ? n : 1;
  const capacity = Math.min(est.sessions, runtimeDefaults(input.engine, input.gpu, input.memoryGiBOverride).maxNumSeqs * schedulers);
  const promptInput = { ...input, contextTokens: w.inputTokens };
  const promptEst = estimate(promptInput);
  const prefillFlopsPerRequest = prefillFlops(input, w.inputTokens);
  const prefillGpus = input.mode === "dp-ep" ? n : est.tp;
  const singlePrefillSeconds = prefillFlopsPerRequest / ((perf.instanceFlopsPerSec / n) * prefillGpus);
  return {
    input,
    avgTokens,
    capacity,
    perf,
    prefillFlopsPerRequest,
    singlePrefillSeconds,
    promptKvBytes: promptEst.kv.perRequestClusterBytes,
    promptCapacity: promptEst.sessions,
  };
}

/** Steady state of one instance at `rps`. withPrefill = false: decode-only instance of a P/D deployment. */
export function instanceLoad(m: InstanceModel, w: Workload, rps: number, withPrefill = true): InstanceLoad {
  const u = withPrefill ? (rps * m.prefillFlopsPerRequest) / m.perf.instanceFlopsPerSec : 0;
  const base = { requestsPerSec: rps, capacity: m.capacity, prefillUtil: u, outputTokPerSec: rps * w.outputTokens };
  if (u >= 1) {
    return { ...base, concurrency: Infinity, ttftSeconds: Infinity, itlMs: Infinity, over: { memory: true, prefill: true, ttft: true, itl: true }, ok: false };
  }
  const ttftSeconds = withPrefill ? m.singlePrefillSeconds / (1 - u) : 0;
  const stepS = (c: number) => m.perf.at(Math.max(1, Math.ceil(c)), u).stepMs / 1000;
  // Iterates rise monotonically to the smallest fixed point. Stop once they pass the capacity; a load that
  // has not settled after many steps sits at the edge of stability and is rejected.
  let c = rps * (ttftSeconds + w.outputTokens * stepS(1));
  let converged = false;
  for (let i = 0; i < 2000 && c <= m.capacity; i++) {
    const next = rps * (ttftSeconds + w.outputTokens * stepS(c));
    converged = Math.abs(next - c) <= 1e-3 * Math.max(1, c);
    c = next;
    if (converged) break;
  }
  const itlMs = stepS(c) * 1000;
  const over = { memory: !converged || c > m.capacity, prefill: false, ttft: ttftSeconds > w.ttftSeconds, itl: itlMs > w.itlMs };
  return { ...base, concurrency: c, ttftSeconds, itlMs, over, ok: !over.memory && !over.ttft && !over.itl };
}

/** Smallest replica count for which `ok(replicas)` holds (monotone), or null beyond `max`. */
function minReplicas(ok: (r: number) => boolean, max = 1 << 16): number | null {
  if (ok(1)) return 1;
  let lo = 1;
  let hi = 2;
  while (!ok(hi)) {
    lo = hi;
    hi *= 2;
    if (hi > max) return null;
  }
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (ok(mid)) hi = mid;
    else lo = mid;
  }
  return hi;
}

export interface AggregatedPlan {
  replicas: number;
  gpus: number;
  load: InstanceLoad;
}

export interface DisaggregatedPlan {
  prefillReplicas: number;
  decodeReplicas: number;
  gpus: number;
  /** Prefill pool: TTFT including the KV transfer. */
  ttftSeconds: number;
  prefillUtil: number;
  kvTransferSeconds: number;
  /** KV traffic from the prefill pool to the decode pool, in bytes per second. */
  kvTrafficBytesPerSec: number;
  decode: InstanceLoad;
}

export interface Cost {
  usdPerHour: number;
  usdPerMillionOutput: number;
  usdPerMillionTotal: number;
}

export interface WorkloadAnalysis {
  model: InstanceModel;
  /** All of the traffic on one instance. */
  single: InstanceLoad;
  aggregated: AggregatedPlan | null;
  disaggregated: DisaggregatedPlan | null;
  /** Why no replica count meets the targets. */
  infeasible: Infeasible | null;
  priceUsdPerHour: number | undefined;
  cost: { aggregated: Cost | null; disaggregated: Cost | null };
}

export function costOf(gpus: number, price: number | undefined, w: Workload): Cost | null {
  if (!price) return null;
  const usdPerHour = gpus * price;
  const perHourTokens = (tokens: number) => w.requestsPerSec * tokens * 3600;
  return {
    usdPerHour,
    usdPerMillionOutput: (usdPerHour / perHourTokens(w.outputTokens)) * 1e6,
    usdPerMillionTotal: (usdPerHour / perHourTokens(w.inputTokens + w.outputTokens)) * 1e6,
  };
}

/** Why the workload cannot be served with this instance whatever the replica count, if so. */
function infeasibility(cfg: AppConfig, m: InstanceModel, w: Workload): Infeasible | null {
  const input = m.input;
  if (collectIssues(input, estimate(input), cfg.weightFormats).some((i) => i.severity === "error")) return "invalid-config";
  if (w.inputTokens + w.outputTokens > m.input.contextTokens) return "request-exceeds-context";
  if (m.capacity < 1) return "no-memory";
  if (m.singlePrefillSeconds > w.ttftSeconds) return "ttft-unreachable";
  if (m.perf.at(1).stepMs > w.itlMs) return "itl-unreachable";
  return null;
}

function disaggregate(cfg: AppConfig, m: InstanceModel, w: Workload): DisaggregatedPlan | null {
  const n = m.input.gpuCount;
  const kvTransferSeconds = m.promptKvBytes / (n * cfg.performance.networkGBsPerGpu * 1e9);
  const prefillOk = (rps: number) => {
    const u = (rps * m.prefillFlopsPerRequest) / m.perf.instanceFlopsPerSec;
    if (u >= 1) return null;
    const ttft = m.singlePrefillSeconds / (1 - u) + kvTransferSeconds;
    // Prompts in flight hold their KV until it reaches the decode pool.
    if (rps * ttft > m.promptCapacity || ttft > w.ttftSeconds) return null;
    return { u, ttft };
  };
  const rps = w.requestsPerSec;
  const prefillReplicas = minReplicas((r) => prefillOk(rps / r) !== null);
  const decodeReplicas = minReplicas((r) => instanceLoad(m, w, rps / r, false).ok);
  if (prefillReplicas === null || decodeReplicas === null) return null;
  const p = prefillOk(rps / prefillReplicas)!;
  return {
    prefillReplicas,
    decodeReplicas,
    gpus: (prefillReplicas + decodeReplicas) * n,
    ttftSeconds: p.ttft,
    prefillUtil: p.u,
    kvTransferSeconds,
    kvTrafficBytesPerSec: rps * m.promptKvBytes,
    decode: instanceLoad(m, w, rps / decodeReplicas, false),
  };
}

export function analyzeWorkload(cfg: AppConfig, input: EstimateInput, w: Workload, priceUsdPerHour = input.gpu.priceUsdPerHour): WorkloadAnalysis {
  const m = instanceModel(cfg, input, w);
  const single = instanceLoad(m, w, w.requestsPerSec);
  const infeasible = infeasibility(cfg, m, w);
  let aggregated: AggregatedPlan | null = null;
  let disaggregated: DisaggregatedPlan | null = null;
  if (!infeasible) {
    const replicas = minReplicas((r) => instanceLoad(m, w, w.requestsPerSec / r).ok);
    if (replicas !== null) aggregated = { replicas, gpus: replicas * input.gpuCount, load: instanceLoad(m, w, w.requestsPerSec / replicas) };
    disaggregated = disaggregate(cfg, m, w);
  }
  return {
    model: m,
    single,
    aggregated,
    disaggregated,
    infeasible,
    priceUsdPerHour,
    cost: { aggregated: aggregated && costOf(aggregated.gpus, priceUsdPerHour, w), disaggregated: disaggregated && costOf(disaggregated.gpus, priceUsdPerHour, w) },
  };
}

export interface WorkloadSizingRow {
  gpu: GpuSpec;
  /** GPUs per instance (null: no configuration meets the targets). */
  gpuCount: number | null;
  mode: ParallelMode;
  tp: number;
  weightFormat: string;
  kvFormat: string;
  replicas: number;
  /** Total GPUs, aggregated serving. */
  gpus: number | null;
  /** Total GPUs with prefill/decode disaggregation on the same instance shape. */
  disaggregatedGpus: number | null;
  itlMs: number | null;
  ttftSeconds: number | null;
  cost: Cost | null;
  note?: string;
}

/**
 * For each GPU type: the instance shape (GPU count, TP/PP or DP-attention) that serves the workload within
 * the targets with the fewest GPUs in total (replicas included); ties go to the lowest ITL.
 * Only configurations without blocking errors at the selected context are considered.
 */
export function sizeForWorkload(cfg: AppConfig, base: SizingBase, w: Workload, baseGpuId?: string, priceOverride?: number): WorkloadSizingRow[] {
  const t = tr(base.lang ?? "fr");
  const rows: WorkloadSizingRow[] = cfg.gpus.map((gpu) => {
    const { wfId, kfId, notes } = formatsFor(cfg, base, gpu);
    const price = gpu.id === baseGpuId && priceOverride ? priceOverride : gpu.priceUsdPerHour;
    let best: { input: EstimateInput; plan: WorkloadAnalysis; gpus: number; unverified: boolean } | null = null;
    for (const n of gpuCountOptions(cfg, gpu)) {
      // A larger instance cannot use fewer GPUs than the best total found so far.
      if (best && n > best.gpus) break;
      for (const input of instanceInputs(cfg, base, gpu, n, baseGpuId)) {
        const est = estimate(input);
        if (est.sessions < 1) continue;
        const issues = collectIssues(input, est, cfg.weightFormats);
        if (issues.some((i) => i.severity === "error")) continue;
        const plan = analyzeWorkload(cfg, input, w, price);
        if (!plan.aggregated) continue;
        const unverified = issues.some((i) => i.unverifiedParallelism);
        const gpus = plan.aggregated.gpus;
        // Unverified parallelism ranks after verified configurations.
        const better =
          !best ||
          (best.unverified && !unverified) ||
          (best.unverified === unverified && (gpus < best.gpus || (gpus === best.gpus && plan.aggregated.load.itlMs < best.plan.aggregated!.load.itlMs)));
        if (better) best = { input, plan, gpus, unverified };
      }
    }
    if (!best)
      return {
        gpu,
        gpuCount: null,
        mode: "tp-pp",
        tp: 1,
        weightFormat: wfId,
        kvFormat: kfId,
        replicas: 0,
        gpus: null,
        disaggregatedGpus: null,
        itlMs: null,
        ttftSeconds: null,
        cost: null,
        note: [...notes, t("objectifs inatteignables", "targets unreachable")].join(", "),
      };
    const agg = best.plan.aggregated!;
    return {
      gpu,
      gpuCount: best.input.gpuCount,
      mode: best.input.mode,
      tp: best.input.tp,
      weightFormat: wfId,
      kvFormat: kfId,
      replicas: agg.replicas,
      gpus: agg.gpus,
      disaggregatedGpus: best.plan.disaggregated?.gpus ?? null,
      itlMs: agg.load.itlMs,
      ttftSeconds: agg.load.ttftSeconds,
      cost: best.plan.cost.aggregated,
      note: [...notes, ...(best.unverified ? [t("parallélisme non vérifié", "unverified parallelism")] : [])].join(", ") || undefined,
    };
  });
  // Cheapest first when every reachable row has a price, otherwise fewest GPUs; unreachable rows last.
  const allPriced = rows.every((r) => r.gpus === null || r.cost);
  const key = (r: WorkloadSizingRow) => (r.gpus === null ? Infinity : allPriced ? r.cost!.usdPerHour : r.gpus);
  return rows.sort((a, b) => key(a) - key(b));
}
