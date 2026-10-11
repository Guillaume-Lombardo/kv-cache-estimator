import {
  CONTEXT_STEPS,
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
import { estimatePerf } from "./perf";

const usable = (l: string) => l !== "unsupported";

/** Format de poids : courant si possible, sinon natif du modèle, sinon le plus proche disponible. */
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

/** Nombres de GPUs proposés : 1-8, puis les grandes tailles (72 réservé aux domaines NVL72). */
export function gpuCountOptions(cfg: AppConfig, g: GpuSpec): number[] {
  const large = cfg.largeGpuCounts.filter((c) => c % 72 !== 0 || g.nvlinkDomain >= 72);
  return [1, 2, 3, 4, 5, 6, 7, 8, ...large];
}

export function divisors(n: number): number[] {
  const out: number[] = [];
  for (let d = 1; d <= n; d++) if (n % d === 0) out.push(d);
  return out;
}

/** TP maximal raisonnable : le domaine NVLink, sinon un serveur PCIe. */
export function maxTp(g: GpuSpec): number {
  return g.interconnect.nvlink ? Math.max(g.nvlinkDomain, 1) : g.maxGpusPerNode;
}

/** TP par défaut plafonné à 8 (pratique courante avec vLLM) ; au-delà, PP ou DP. */
const DEFAULT_TP_CAP = 8;

/** Plus grand degré de TP valide qui divise n et reste dans le domaine d'interconnexion. */
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
 * Dimensionnement inverse : pour chaque GPU, plus petit nombre de cartes qui atteint `targetSessions`
 * sans erreur bloquante, en testant TP×PP (TP valides) et DP-attention.
 */
/** Configuration recommandable : aucune erreur bloquante (formats, parallélisme, contexte, capacité). */
export function isValidConfig(cfg: AppConfig, input: EstimateInput): boolean {
  const est = estimate(input);
  return !collectIssues(input, est, cfg.weightFormats).some((i) => i.severity === "error");
}

export function sizeForTarget(
  cfg: AppConfig,
  base: Omit<EstimateInput, "gpu" | "gpuCount" | "tp" | "mode" | "weightFormat" | "kvFormat"> & { weightFormatId: string; kvFormatId: string },
  targetSessions: number,
  baseGpuId?: string,
): SizingRow[] {
  const rows: SizingRow[] = [];
  const maxCtx = base.model.maxContextExtended ?? base.model.maxContext;
  for (const gpu of cfg.gpus) {
    const wfId = pickWeightFormat(cfg, base.model, gpu, base.engine, base.weightFormatId);
    const kfId = pickKvFormat(cfg, gpu, base.engine, base.kvFormatId);
    const weightFormat = cfg.weightFormats.find((f) => f.id === wfId)!;
    const kvFormat = cfg.kvCacheFormats.find((f) => f.id === kfId)!;
    const notes: string[] = [];
    if (wfId !== base.weightFormatId) notes.push(`poids ${weightFormat.label}`);
    if (kfId !== base.kvFormatId) notes.push(`KV ${kvFormat.label}`);

    let found: SizingRow | null = null;
    for (const n of gpuCountOptions(cfg, gpu)) {
      const configs: { mode: ParallelMode; tp: number }[] = divisors(n)
        .filter((d) => d <= maxTp(gpu) && isValidTp(base.model, d))
        .map((tp) => ({ mode: "tp-pp" as const, tp }));
      if (n > 1 && isMoE(base.model) && base.engine.features.dpAttention !== false) configs.push({ mode: "dp-ep", tp: 1 });
      // Parmi les configurations qui atteignent l'objectif, on garde celle au meilleur débit agrégé
      // (mesuré à l'objectif, pas à pleine charge).
      let best: { input: EstimateInput; sessions: number; rate: number; score: number; unverified: boolean } | null = null;
      for (const c of configs) {
        const memoryWeightFormat = memoryFormatFor(cfg.weightFormats, gpu, base.engine, weightFormat);
        // La mémoire visible personnalisée ne vaut que pour la carte sur laquelle elle a été saisie.
        const memoryGiBOverride = gpu.id === baseGpuId ? base.memoryGiBOverride : undefined;
        const input: EstimateInput = { ...base, gpu, gpuCount: n, mode: c.mode, tp: c.tp, weightFormat, memoryWeightFormat, kvFormat, memoryGiBOverride };
        const est = estimate(input);
        if (est.sessions < targetSessions) continue;
        const issues = collectIssues(input, est, cfg.weightFormats);
        if (issues.some((i) => i.severity === "error")) continue;
        // Une configuration qui repose sur une capacité non vérifiée passe après les configurations vérifiées.
        const unverified = issues.some((i) => /(non vérifié|non vérifiée)/.test(i.title) && /(parallel|DP-attention)/i.test(i.title));
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
          note: [...notes, ...(best.unverified ? ["parallélisme non vérifié"] : [])].join(", ") || undefined,
        };
        break;
      }
    }
    if (!found && base.contextTokens > maxCtx) notes.push("contexte hors limite du modèle");
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
        note: [...notes, "objectif inatteignable"].join(", "),
      },
    );
  }
  return rows.sort((a, b) => (a.gpuCount ?? 1e9) - (b.gpuCount ?? 1e9) || b.sessions - a.sessions);
}

/** Plus grand contexte (parmi les crans, dans la limite du modèle) qui permet `targetSessions` sessions sans erreur. */
export function maxContextFor(cfg: AppConfig, input: EstimateInput, targetSessions: number): number | null {
  let best: number | null = null;
  const limit = input.model.maxContextExtended ?? input.model.maxContext;
  for (const t of CONTEXT_STEPS) {
    if (t > limit) break;
    const i = { ...input, contextTokens: t };
    if (estimate(i).sessions >= targetSessions && isValidConfig(cfg, i)) best = t;
  }
  return best;
}
