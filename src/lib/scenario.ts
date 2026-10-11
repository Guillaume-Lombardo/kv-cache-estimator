import { CONTEXT_STEPS, isMoE, memoryFormatFor, type EstimateInput, type ParallelMode } from "./calc";
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
}

export type QueryParams = Record<string, string | undefined>;

export function nearestContext(tokens: number | undefined, fallback = 32768): number {
  const t = tokens && tokens > 0 ? tokens : fallback;
  return CONTEXT_STEPS.reduce((best, s) => (Math.abs(Math.log2(s / t)) < Math.abs(Math.log2(best / t)) ? s : best), CONTEXT_STEPS[0]);
}

const numParam = (v: string | undefined) => (v !== undefined && v !== "" && Number.isFinite(Number(v)) ? Number(v) : undefined);

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
  return {
    modelId: model.id,
    ctx: nearestContext(numParam(q.ctx), d.contextTokens),
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
  const parts = [`vllm serve ${checkpoint}`];
  if (mode === "dp-ep" && f.dpAttention) parts.push(f.dpAttention.replace("N", String(gpuCount)));
  else {
    if (tp > 1) parts.push(`${f.tensorParallel} ${tp}`);
    if (pp > 1 && f.pipelineParallel) parts.push(`${f.pipelineParallel} ${pp}`);
  }
  if (f.kvCacheDtype) parts.push(`${f.kvCacheDtype} ${kvFormat.id === "fp8" ? "fp8" : kvFormat.id === "nvfp4" ? "nvfp4" : "auto"}`);
  if (f.memoryUtilization) parts.push(`${f.memoryUtilization} ${(1 - reservePct / 100).toFixed(2)}`);
  parts.push(`--max-model-len ${contextTokens}`);
  if (model.ssmStateBytes === 4) parts.push("--mamba-ssm-cache-dtype float32");
  const header = native
    ? ""
    : t(
        `# Le catalogue n'associe aucun checkpoint ${weightFormat.label} à ${model.name} (dépôt de référence : ${model.source ?? "?"}, format ${model.nativeWeightFormat.toUpperCase()}).\n# Remplacez la ligne suivante par un checkpoint ${weightFormat.label} existant et supporté.\n`,
        `# The catalogue maps no ${weightFormat.label} checkpoint to ${model.name} (reference repository: ${model.source ?? "?"}, ${model.nativeWeightFormat.toUpperCase()} format).\n# Replace the next line with an existing, supported ${weightFormat.label} checkpoint.\n`,
      );
  return { text: header + parts.join(" \\\n  "), matchesScenario: native };
}
