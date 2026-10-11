import { CONTEXT_STEPS, isMoE, memoryFormatFor, type EstimateInput, type ParallelMode } from "./calc";
import type { AppConfig, ModelSpec } from "./config-schema";
import { defaultTp, divisors, gpuCountOptions, pickKvFormat, pickWeightFormat } from "./planning";

/** État complet d'un scénario : sérialisable dans l'URL, épinglable pour comparaison. */
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
  /** Mémoire visible par GPU (Gio) si différente du catalogue. */
  memGiB?: number;
  /** Mémoire CPU d'offload par GPU (Gio). */
  offloadGiB: number;
}

export type QueryParams = Record<string, string | undefined>;

export function nearestContext(tokens: number | undefined, fallback = 32768): number {
  const t = tokens && tokens > 0 ? tokens : fallback;
  return CONTEXT_STEPS.reduce((best, s) => (Math.abs(Math.log2(s / t)) < Math.abs(Math.log2(best / t)) ? s : best), CONTEXT_STEPS[0]);
}

const numParam = (v: string | undefined) => (v !== undefined && v !== "" && Number.isFinite(Number(v)) ? Number(v) : undefined);

/** Construit un scénario valide à partir de paramètres d'URL (incomplets ou périmés). */
export function scenarioFromQuery(cfg: AppConfig, models: ModelSpec[], q: QueryParams): Scenario {
  const d = cfg.defaults;
  const model = models.find((m) => m.id === q.model) ?? models.find((m) => m.id === d.model) ?? models[0];
  const gpu = cfg.gpus.find((g) => g.id === q.gpu) ?? cfg.gpus.find((g) => g.id === d.gpu) ?? cfg.gpus[0];
  const engine = cfg.engines.find((e) => e.id === q.eng) ?? cfg.engines.find((e) => e.id === d.engine) ?? cfg.engines[0];
  const counts = gpuCountOptions(cfg, gpu);
  const nq = numParam(q.n);
  const gpuCount = nq !== undefined && counts.includes(nq) ? nq : counts.includes(d.gpuCount ?? 1) ? (d.gpuCount ?? 1) : 1;
  // Compatibilité : anciens liens ?p=tp|pp|dp-attention
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

/** Résout un scénario en entrée de calcul (null si un identifiant n'existe plus). */
export function resolveScenario(cfg: AppConfig, models: ModelSpec[], s: Scenario): EstimateInput | null {
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
  };
}

/** Commande de lancement indicative pour les moteurs qui déclarent leurs options. */
export function launchCommand(input: EstimateInput, tp: number, pp: number): string | null {
  const { engine, model, gpuCount, mode, kvFormat, reservePct, contextTokens } = input;
  const f = engine.flags;
  if (!f.tensorParallel) return null;
  const parts = [`vllm serve ${model.source ?? model.id}`];
  if (mode === "dp-ep" && f.dpAttention) parts.push(f.dpAttention.replace("N", String(gpuCount)));
  else {
    if (tp > 1) parts.push(`${f.tensorParallel} ${tp}`);
    if (pp > 1 && f.pipelineParallel) parts.push(`${f.pipelineParallel} ${pp}`);
  }
  if (f.kvCacheDtype) parts.push(`${f.kvCacheDtype} ${kvFormat.id === "fp8" ? "fp8" : kvFormat.id === "nvfp4" ? "nvfp4" : "auto"}`);
  if (f.memoryUtilization) parts.push(`${f.memoryUtilization} ${(1 - reservePct / 100).toFixed(2)}`);
  parts.push(`--max-model-len ${contextTokens}`);
  if (model.ssmStateBytes === 4) parts.push("--mamba-ssm-cache-dtype float32");
  return parts.join(" \\\n  ");
}
