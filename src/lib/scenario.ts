import { MAX_CONTEXT, MIN_CONTEXT, isMoE, memoryFormatFor, type EstimateInput, type ParallelMode } from "./calc";
import type { AppConfig, ModelSpec } from "./config-schema";
import { tr, type Lang } from "./i18n";
import { defaultTp, divisors, gpuCountOptions, pickKvFormat, pickWeightFormat } from "./planning";

/** Complete scenario state: URL-serializable and pinnable for comparison. */
export interface Scenario {
  modelId: string;
  ctx: number;
  gpuId: string;
  engineId: string;
  gpuCount: number;
  mode: ParallelMode;
  tp: number;
  wf: string;
  kf: string;
  reservePct: number;
  /** Visible memory per GPU (GiB), if different from the catalogue. */
  memGiB?: number;
  /** CPU offload memory per GPU (GiB). */
  offloadGiB: number;
  /** Tokens per scheduling step, if different from the engine default. */
  batchedTokens?: number;
  /** Runtime overhead per GPU (GiB) entered instead of the estimate. */
  overheadGiB?: number;
  /** Workload for the load and SLO analysis. */
  workload: Workload;
  /** Price per GPU-hour (USD) for the selected card, if different from the catalogue. */
  priceUsd?: number;
}

export interface Workload {
  /** Average prompt length, in tokens. */
  inputTokens: number;
  /** Average generated length, in tokens. */
  outputTokens: number;
  requestsPerSec: number;
  /** Time-to-first-token target, in seconds. */
  ttftSeconds: number;
  /** Inter-token latency target, in milliseconds. */
  itlMs: number;
}

export const DEFAULT_WORKLOAD: Workload = { inputTokens: 2000, outputTokens: 500, requestsPerSec: 2, ttftSeconds: 2, itlMs: 50 };

export type QueryParams = Record<string, string | undefined>;

/** Context length within the accepted range, rounded to a whole token; the fallback when missing or invalid. */
export function clampContext(tokens: number | undefined, fallback = 32768): number {
  const t = tokens && tokens > 0 ? Math.round(tokens) : fallback;
  return Math.min(MAX_CONTEXT, Math.max(MIN_CONTEXT, t));
}

const numParam = (v: string | undefined) => (v !== undefined && v !== "" && Number.isFinite(Number(v)) ? Number(v) : undefined);
const positive = (v: string | undefined) => {
  const n = numParam(v);
  return n !== undefined && n > 0 ? n : undefined;
};

/** Build a valid scenario from potentially incomplete or outdated URL parameters. */
export function scenarioFromQuery(cfg: AppConfig, models: ModelSpec[], q: QueryParams): Scenario {
  const d = cfg.defaults;
  const model = models.find((m) => m.id === q.model) ?? models.find((m) => m.id === d.model) ?? models[0];
  const gpu = cfg.gpus.find((g) => g.id === q.gpu) ?? cfg.gpus.find((g) => g.id === d.gpu) ?? cfg.gpus[0];
  const engine = cfg.engines.find((e) => e.id === q.eng) ?? cfg.engines.find((e) => e.id === d.engine) ?? cfg.engines[0];
  const counts = gpuCountOptions(cfg, gpu);
  const nq = numParam(q.n);
  const gpuCount = nq !== undefined && counts.includes(nq) ? nq : counts.includes(d.gpuCount ?? 1) ? (d.gpuCount ?? 1) : 1;
  // Compatibility with legacy links: ?p=tp|pp|dp-attention
  let mode: ParallelMode = q.mode === "dp-ep" || q.p === "dp-attention" ? "dp-ep" : "tp-pp";
  if (mode === "dp-ep" && (!isMoE(model) || engine.features.dpAttention === false)) mode = "tp-pp";
  const tpq = numParam(q.tp) ?? (q.p === "pp" ? 1 : q.p === "tp" ? gpuCount : undefined);
  const tp = tpq !== undefined && divisors(gpuCount).includes(tpq) ? tpq : defaultTp(model, gpu, gpuCount);
  const r = numParam(q.r);
  const mem = numParam(q.mem);
  const off = numParam(q.off);
  const ovh = numParam(q.ovh);
  const wd = { ...DEFAULT_WORKLOAD, ...d.workload };
  const int = (v: string | undefined) => {
    const n = positive(v);
    return n === undefined ? undefined : Math.max(1, Math.round(n));
  };
  return {
    modelId: model.id,
    ctx: clampContext(numParam(q.ctx), d.contextTokens),
    gpuId: gpu.id,
    engineId: engine.id,
    gpuCount,
    mode,
    tp,
    wf: pickWeightFormat(cfg, model, gpu, engine, q.w),
    kf: pickKvFormat(cfg, gpu, engine, q.kv ?? d.kvCacheFormat),
    reservePct: r !== undefined && r >= 0 && r <= 50 ? r : (d.reservePct ?? 10),
    memGiB: mem !== undefined && mem > 0 ? mem : undefined,
    offloadGiB: off !== undefined && off >= 0 ? off : 0,
    batchedTokens: int(q.mbt),
    overheadGiB: ovh !== undefined && ovh >= 0 ? ovh : undefined,
    workload: {
      inputTokens: int(q.in) ?? wd.inputTokens,
      outputTokens: int(q.out) ?? wd.outputTokens,
      requestsPerSec: positive(q.rps) ?? wd.requestsPerSec,
      ttftSeconds: positive(q.ttft) ?? wd.ttftSeconds,
      itlMs: positive(q.itl) ?? wd.itlMs,
    },
    priceUsd: positive(q.usd),
  };
}

export function scenarioToQuery(s: Scenario): string {
  const q = new URLSearchParams({
    model: s.modelId,
    ctx: String(s.ctx),
    gpu: s.gpuId,
    eng: s.engineId,
    n: String(s.gpuCount),
    mode: s.mode,
    tp: String(s.tp),
    w: s.wf,
    kv: s.kf,
    r: String(s.reservePct),
  });
  if (s.memGiB) q.set("mem", String(s.memGiB));
  if (s.offloadGiB) q.set("off", String(s.offloadGiB));
  if (s.batchedTokens) q.set("mbt", String(s.batchedTokens));
  if (s.overheadGiB !== undefined) q.set("ovh", String(s.overheadGiB));
  const w = s.workload;
  q.set("in", String(w.inputTokens));
  q.set("out", String(w.outputTokens));
  q.set("rps", String(w.requestsPerSec));
  q.set("ttft", String(w.ttftSeconds));
  q.set("itl", String(w.itlMs));
  if (s.priceUsd) q.set("usd", String(s.priceUsd));
  return q.toString();
}

/** Resolve a scenario to calculator input (null if an identifier no longer exists). */
/** Scenario to calculation input; lang sets the language of generated labels and messages. */
export function resolveScenario(cfg: AppConfig, models: ModelSpec[], s: Scenario, lang?: Lang): EstimateInput | null {
  const model = models.find((m) => m.id === s.modelId);
  const gpu = cfg.gpus.find((g) => g.id === s.gpuId);
  const engine = cfg.engines.find((e) => e.id === s.engineId);
  const weightFormat = cfg.weightFormats.find((f) => f.id === s.wf);
  const kvFormat = cfg.kvCacheFormats.find((f) => f.id === s.kf);
  if (!model || !gpu || !engine || !weightFormat || !kvFormat) return null;
  return {
    model,
    gpu,
    engine,
    gpuCount: s.gpuCount,
    mode: s.mode,
    tp: s.tp,
    contextTokens: s.ctx,
    weightFormat,
    memoryWeightFormat: memoryFormatFor(cfg.weightFormats, gpu, engine, weightFormat),
    kvFormat,
    reservePct: s.reservePct,
    memoryGiBOverride: s.memGiB,
    offloadGiBPerGpu: s.offloadGiB,
    maxBatchedTokens: s.batchedTokens,
    overheadGiBOverride: s.overheadGiB,
    lang,
  };
}

/**
 * Indicative launch command for engines that declare their CLI options.
 * The catalogue repository covers only the native format. For another format, the checkpoint
 * must be supplied; the command states this rather than claiming to reproduce the scenario.
 */
export function launchCommand(input: EstimateInput, tp: number, pp: number): { text: string; matchesScenario: boolean } | null {
  const { engine, model, gpuCount, mode, kvFormat, reservePct, contextTokens, weightFormat } = input;
  const t = tr(input.lang ?? "fr");
  const f = engine.flags;
  if (!f.tensorParallel) return null;
  const repo = model.checkpoints[weightFormat.id] ?? (weightFormat.id === model.nativeWeightFormat ? model.source : undefined);
  const native = !!repo;
  const checkpoint = repo ?? t(`<checkpoint-${weightFormat.id}-de-${model.id}>`, `<checkpoint-${weightFormat.id}-of-${model.id}>`);
  // Profiles without `launch` describe vLLM-style options (catalogues written before it existed).
  const legacy = !f.launch;
  const parts = [`${f.launch ?? "vllm serve"} ${checkpoint}`];
  if (mode === "dp-ep" && f.dpAttention) parts.push(f.dpAttention.replace(/\bN\b/g, String(gpuCount)));
  else {
    if (tp > 1) parts.push(`${f.tensorParallel} ${tp}`);
    if (pp > 1 && f.pipelineParallel) parts.push(`${f.pipelineParallel} ${pp}`);
  }
  if (f.kvCacheDtype) parts.push(`${f.kvCacheDtype} ${f.kvCacheDtypeValues[kvFormat.id] ?? (kvFormat.id === "fp8" ? "fp8" : kvFormat.id === "nvfp4" ? "nvfp4" : "auto")}`);
  if (f.memoryUtilization) parts.push(`${f.memoryUtilization} ${(1 - reservePct / 100).toFixed(2)}`);
  const maxModelLen = f.maxModelLen ?? (legacy ? "--max-model-len" : undefined);
  if (maxModelLen) parts.push(`${maxModelLen} ${contextTokens}`);
  if (input.maxBatchedTokens && f.maxNumBatchedTokens) parts.push(`${f.maxNumBatchedTokens} ${input.maxBatchedTokens}`);
  const ssmFlag = f.mambaSsmCacheDtype ?? (legacy ? "--mamba-ssm-cache-dtype" : undefined);
  if (model.ssmStateBytes === 4 && ssmFlag) parts.push(`${ssmFlag} float32`);
  const header = native
    ? ""
    : t(
        `# Le catalogue n'associe aucun checkpoint ${weightFormat.label} à ${model.name} (dépôt de référence : ${model.source ?? "?"}, format ${model.nativeWeightFormat.toUpperCase()}).\n# Remplacez la ligne suivante par un checkpoint ${weightFormat.label} existant et supporté.\n`,
        `# The catalogue maps no ${weightFormat.label} checkpoint to ${model.name} (reference repository: ${model.source ?? "?"}, ${model.nativeWeightFormat.toUpperCase()} format).\n# Replace the next line with an existing, supported ${weightFormat.label} checkpoint.\n`,
      );
  return { text: header + parts.join(" \\\n  "), matchesScenario: native };
}
