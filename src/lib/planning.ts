import {
  contextCandidates,
  collectIssues,
  estimate,
  isMoE,
  isValidTp,
  kvSupport,
  memoryFormatFor,
  weightSupport,
  type EstimateInput,
  type ParallelMode,
} from "./calc";
import type { AppConfig, EngineSpec, GpuSpec, ModelSpec } from "./config-schema";
import { tr } from "./i18n";
import { estimatePerf } from "./perf";

const usable = (l: string) => l !== "unsupported";

/** Weight format: current selection when possible, then the model's native format, then the closest available format. */
export function pickWeightFormat(cfg: AppConfig, m: ModelSpec, g: GpuSpec, e: EngineSpec, current?: string): string {
  if (current && cfg.weightFormats.some((f) => f.id === current) && usable(weightSupport(g, e, current).level)) return current;
  const native = m.nativeWeightFormat;
  if (usable(weightSupport(g, e, native).level)) return native;
  if (native === "bf16" && usable(weightSupport(g, e, "fp16").level)) return "fp16";
  const official = m.officialWeightFormats.find((f) => weightSupport(g, e, f).level === "native");
  if (official) return official;
  return cfg.weightFormats.find((f) => weightSupport(g, e, f.id).level === "native")?.id ?? cfg.weightFormats[0].id;
}

export function pickKvFormat(cfg: AppConfig, g: GpuSpec, e: EngineSpec, current?: string): string {
  if (current && cfg.kvCacheFormats.some((f) => f.id === current) && usable(kvSupport(g, e, current).level)) return current;
  for (const id of ["bf16", "fp16"]) if (kvSupport(g, e, id).level === "native") return id;
  return cfg.kvCacheFormats.find((f) => usable(kvSupport(g, e, f.id).level))?.id ?? cfg.kvCacheFormats[0].id;
}

/** Available GPU counts: 1-8, then larger sizes (72 is reserved for NVL72 domains). */
export function gpuCountOptions(cfg: AppConfig, g: GpuSpec): number[] {
  const large = cfg.largeGpuCounts.filter((c) => c % 72 !== 0 || g.nvlinkDomain >= 72);
  return [1, 2, 3, 4, 5, 6, 7, 8, ...large];
}

export function divisors(n: number): number[] {
  const out: number[] = [];
  for (let d = 1; d <= n; d++) if (n % d === 0) out.push(d);
  return out;
}

/** Practical TP limit: the scale-up fabric domain (NVLink, Infinity Fabric), or a PCIe server otherwise. */
export function maxTp(g: GpuSpec): number {
  return g.interconnect.nvlink ? Math.max(g.nvlinkDomain, 1) : g.maxGpusPerNode;
}

/** Default TP capped at 8 (common vLLM practice); use PP or DP beyond that. */
const DEFAULT_TP_CAP = 8;

/** Largest valid TP degree that divides n and stays within the interconnect domain. */
export function defaultTp(m: ModelSpec, g: GpuSpec, n: number): number {
  const candidates = divisors(n).filter((d) => d <= Math.min(maxTp(g), DEFAULT_TP_CAP) && isValidTp(m, d));
  return candidates.length ? candidates[candidates.length - 1] : 1;
}

export interface SizingRow {
  gpu: GpuSpec;
  gpuCount: number | null;
  mode: ParallelMode;
  tp: number;
  sessions: number;
  weightFormat: string;
  kvFormat: string;
  tokPerSecTotal: number | null;
  note?: string;
}

/**
 * Inverse sizing: for each GPU type, find the smallest count reaching `targetSessions`
 * without blocking errors, testing TP x PP (valid TP degrees) and DP-attention.
 */
/** Recommendable configuration: no blocking errors (formats, parallelism, context, capacity). */
export function isValidConfig(cfg: AppConfig, input: EstimateInput): boolean {
  const est = estimate(input);
  return !collectIssues(input, est, cfg.weightFormats).some((i) => i.severity === "error");
}

export type SizingBase = Omit<EstimateInput, "gpu" | "gpuCount" | "tp" | "mode" | "weightFormat" | "kvFormat"> & { weightFormatId: string; kvFormatId: string };

/** Formats used on one GPU type: the base selection when supported, with a note for each fallback. */
export function formatsFor(cfg: AppConfig, base: SizingBase, gpu: GpuSpec) {
  const t = tr(base.lang ?? "fr");
  const wfId = pickWeightFormat(cfg, base.model, gpu, base.engine, base.weightFormatId);
  const kfId = pickKvFormat(cfg, gpu, base.engine, base.kvFormatId);
  const weightFormat = cfg.weightFormats.find((f) => f.id === wfId)!;
  const kvFormat = cfg.kvCacheFormats.find((f) => f.id === kfId)!;
  const notes: string[] = [];
  if (wfId !== base.weightFormatId) notes.push(t(`poids ${weightFormat.label}`, `${weightFormat.label} weights`));
  if (kfId !== base.kvFormatId) notes.push(`KV ${kvFormat.label}`);
  return { wfId, kfId, weightFormat, kvFormat, notes };
}

/** Configurations to evaluate for n GPUs of one type: every valid TP degree (rest in PP), plus DP-attention for MoE. */
export function instanceInputs(cfg: AppConfig, base: SizingBase, gpu: GpuSpec, n: number, baseGpuId?: string): EstimateInput[] {
  const { weightFormat, kvFormat } = formatsFor(cfg, base, gpu);
  const configs: { mode: ParallelMode; tp: number }[] = divisors(n)
    .filter((d) => d <= maxTp(gpu) && isValidTp(base.model, d))
    .map((tp) => ({ mode: "tp-pp" as const, tp }));
  if (n > 1 && isMoE(base.model) && base.engine.features.dpAttention !== false) configs.push({ mode: "dp-ep", tp: 1 });
  const memoryWeightFormat = memoryFormatFor(cfg.weightFormats, gpu, base.engine, weightFormat);
  // The visible-memory override applies only to the GPU type for which it was entered.
  const memoryGiBOverride = gpu.id === baseGpuId ? base.memoryGiBOverride : undefined;
  const { weightFormatId: _w, kvFormatId: _k, ...rest } = base;
  return configs.map((c) => ({ ...rest, gpu, gpuCount: n, mode: c.mode, tp: c.tp, weightFormat, memoryWeightFormat, kvFormat, memoryGiBOverride }));
}

export function sizeForTarget(
  cfg: AppConfig,
  base: SizingBase,
  targetSessions: number,
  baseGpuId?: string,
): SizingRow[] {
  const rows: SizingRow[] = [];
  const maxCtx = base.model.maxContextExtended ?? base.model.maxContext;
  const t = tr(base.lang ?? "fr");
  for (const gpu of cfg.gpus) {
    const { wfId, kfId, notes } = formatsFor(cfg, base, gpu);

    let found: SizingRow | null = null;
    for (const n of gpuCountOptions(cfg, gpu)) {
      // Among configurations reaching the target, keep the highest aggregate throughput
      // (measured at the target concurrency, not at full capacity).
      let best: { input: EstimateInput; sessions: number; rate: number; score: number; unverified: boolean } | null = null;
      for (const input of instanceInputs(cfg, base, gpu, n, baseGpuId)) {
        const est = estimate(input);
        if (est.sessions < targetSessions) continue;
        const issues = collectIssues(input, est, cfg.weightFormats);
        if (issues.some((i) => i.severity === "error")) continue;
        // Rank configurations using unverified capabilities below verified configurations.
        const unverified = issues.some((i) => i.unverifiedParallelism);
        const rate = estimatePerf(input, est, cfg.performance).at(targetSessions).tokPerSecTotal;
        const score = unverified ? rate / 1e6 : rate;
        if (!best || score > best.score) best = { input, sessions: est.sessions, rate, score, unverified };
      }
      if (best) {
        found = {
          gpu,
          gpuCount: n,
          mode: best.input.mode,
          tp: best.input.tp,
          sessions: best.sessions,
          weightFormat: wfId,
          kvFormat: kfId,
          tokPerSecTotal: best.rate,
          note: [...notes, ...(best.unverified ? [t("parallélisme non vérifié", "unverified parallelism")] : [])].join(", ") || undefined,
        };
        break;
      }
    }
    if (!found && base.contextTokens > maxCtx) notes.push(t("contexte hors limite du modèle", "context beyond the model limit"));
    rows.push(
      found ?? {
        gpu,
        gpuCount: null,
        mode: "tp-pp",
        tp: 1,
        sessions: 0,
        weightFormat: wfId,
        kvFormat: kfId,
        tokPerSecTotal: null,
        note: [...notes, t("objectif inatteignable", "target unreachable")].join(", "),
      },
    );
  }
  return rows.sort((a, b) => (a.gpuCount ?? 1e9) - (b.gpuCount ?? 1e9) || b.sessions - a.sessions);
}

/** Largest preset context (or model limit) that supports `targetSessions` without errors. */
export function maxContextFor(cfg: AppConfig, input: EstimateInput, targetSessions: number): number | null {
  let best: number | null = null;
  for (const t of contextCandidates(input.model)) {
    const i = { ...input, contextTokens: t };
    if (estimate(i).sessions >= targetSessions && isValidConfig(cfg, i)) best = t;
  }
  return best;
}
