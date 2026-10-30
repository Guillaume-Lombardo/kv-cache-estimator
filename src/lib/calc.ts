import type {
  EngineSpec,
  GpuSpec,
  KvFormat,
  LayerGroup,
  ModelSpec,
  Support,
  SupportLevel,
  WeightFormat,
} from "./config-schema";
import { tr, type Lang } from "./i18n";

export const GIB = 1024 ** 3;

/** 8k to 1M in doubling steps (powers of 2). */
export const CONTEXT_STEPS = [8, 16, 32, 64, 128, 256, 512, 1024].map((k) => k * 1024);

export function contextLabel(tokens: number): string {
  if (tokens >= 1024 * 1024) return `${+(tokens / (1024 * 1024)).toFixed(1)}M`;
  return `${Math.round(tokens / 1024)}k`;
}

/** tp-pp: TP within a stage, PP across stages (tp = n: pure TP; tp = 1: pure PP). dp-ep: DP-attention + expert parallelism. */
export type ParallelMode = "tp-pp" | "dp-ep";

export interface EstimateInput {
  model: ModelSpec;
  gpu: GpuSpec;
  engine: EngineSpec;
  gpuCount: number;
  mode: ParallelMode;
  /** TP degree (tp-pp mode); PP = gpuCount / tp. */
  tp: number;
  contextTokens: number;
  /** Weight file format. */
  weightFormat: WeightFormat;
  /** Effective GPU memory format when the engine converts the file format. */
  memoryWeightFormat?: WeightFormat;
  kvFormat: KvFormat;
  /** Fraction of each GPU's memory reserved for activations, CUDA graphs and fragmentation. */
  reservePct: number;
  /** Visible memory per GPU when different from the catalogue (nvidia-smi). */
  memoryGiBOverride?: number;
  /** CPU memory dedicated to KV offload, per GPU. */
  offloadGiBPerGpu?: number;
  /** Language of the generated labels and messages (French by default). */
  lang?: Lang;
}

export type Severity = "error" | "warning" | "info";
export interface Issue {
  severity: Severity;
  title: string;
  detail?: string;
  /** Tensor/pipeline parallel or DP-attention that the engine profile does not confirm. */
  unverifiedParallelism?: boolean;
}

export interface GroupBreakdown {
  kind: LayerGroup["kind"];
  label: string;
  count: number;
  detail: string;
  /** Bytes for one request at the selected context (logical view, without replication). */
  bytes: number;
  /** Bytes stored on one GPU for one request, accounting for parallelism. */
  perGpuBytes: number;
  /** Cache replication factor across GPUs (1 means no duplication). */
  replication: number;
}

export interface Estimate {
  tp: number;
  pp: number;
  weights: {
    bitsPerParam: number;
    /** Model weights (one copy). */
    totalBytes: number;
    /** Weights on the most heavily loaded GPU. */
    perGpuBytes: number;
    quantizedParams: number;
    unquantizedParams: number;
    /** DP-attention: non-expert tensors stored on every GPU. */
    nonExpertBytes: number;
    /** Additional weight copies (DP-attention, tied embeddings in PP). */
    replicatedBytes: number;
    /** DP-attention without an expert parameter count: capacity is not validated. */
    dpUnvalidated: boolean;
  };
  /** Layers per PP stage on the most heavily loaded rank. */
  layersPerStage: number;
  kv: {
    /** KV cache for one request, logical view. */
    perRequestBytes: number;
    /** Physical memory consumed by one request across all GPUs. */
    perRequestClusterBytes: number;
    /** Memory consumed by one request on its hosting GPU. */
    perRequestPerGpuBytes: number;
    perTokenBytes: number;
    constantBytes: number;
    groups: GroupBreakdown[];
  };
  memory: {
    gpuBytes: number;
    reservePerGpuBytes: number;
    usablePerGpuBytes: number;
    freeForKvPerGpuBytes: number;
    /** Total physical free memory: N x usable memory minus physical weights, including replicas. */
    freeForKvTotalBytes: number;
    /** Conservative budget: free memory on the most heavily loaded GPU x N (used for session capacity). */
    kvBudgetTotalBytes: number;
  };
  offload: {
    totalBytes: number;
    /** Additional sessions that can be parked in CPU memory (preempted or queued). */
    parkedSessions: number;
    /** Time to reload a full session onto the GPU, in ms. */
    reloadMs: number;
  };
  /** Logical sum of weights and one request, without replication. */
  modelPlusOneKvBytes: number;
  /** Physical memory usage with one request, including replicas. */
  physicalOneRequestBytes: number;
  fitsOneRequest: boolean;
  fitsWeights: boolean;
  sessions: number;
  sessionsPerGpu: number;
}

const LEVEL_ORDER: Record<SupportLevel, number> = { native: 0, limited: 1, unverified: 2, unsupported: 3 };

export function supportOf(map: Record<string, Support>, id: string, missing: SupportLevel = "unsupported"): Support {
  return map[id] ?? { level: missing };
}

/** Combined GPU/engine support: use the most restrictive level and combine notes. */
export function combinedSupport(
  gpuMap: Record<string, Support>,
  engineMap: Record<string, Support>,
  id: string,
  engineName: string,
  lang: Lang = "fr",
): Support {
  const t = tr(lang);
  const g = supportOf(gpuMap, id);
  const e = supportOf(engineMap, id);
  const level = LEVEL_ORDER[g.level] >= LEVEL_ORDER[e.level] ? g.level : e.level;
  const engineNote =
    e.note ??
    (e.level === "unsupported"
      ? t(`Non supporté par ${engineName}.`, `Not supported by ${engineName}.`)
      : e.level === "unverified"
        ? t(`Support non vérifié dans ${engineName}.`, `Support not verified in ${engineName}.`)
        : undefined);
  const notes = [g.note, engineNote].filter(Boolean);
  return { level, note: notes.length ? notes.join(" ") : undefined, memoryFormat: e.memoryFormat ?? g.memoryFormat };
}

export const weightSupport = (gpu: GpuSpec, engine: EngineSpec, id: string, lang?: Lang) =>
  combinedSupport(gpu.weightFormats, engine.weightFormats, id, engine.name, lang);
export const kvSupport = (gpu: GpuSpec, engine: EngineSpec, id: string, lang?: Lang) =>
  combinedSupport(gpu.kvCacheFormats, engine.kvCacheFormats, id, engine.name, lang);

/** Actual GPU memory format for a given weight file. */
export function memoryFormatFor(formats: WeightFormat[], gpu: GpuSpec, engine: EngineSpec, file: WeightFormat): WeightFormat {
  const id = weightSupport(gpu, engine, file.id).memoryFormat;
  return (id && formats.find((f) => f.id === id)) || file;
}

/** Apply sliding-window VRAM savings only if the engine frees blocks outside the window. */
export const slidingWindowSaves = (e: EngineSpec) => e.features.slidingWindowAttention === true && e.features.slidingWindowFreesBlocks === true;

export function embeddingParams(m: ModelSpec): number {
  return m.vocabSize * m.hiddenSize * (m.tiedEmbeddings ? 1 : 2);
}

export function effectiveBits(m: ModelSpec, f: WeightFormat): number {
  return f.id === m.nativeWeightFormat && m.nativeBitsPerParam ? m.nativeBitsPerParam : f.bitsPerParam;
}

/** Embeddings / lm_head (and the vision encoder) stay at 16 bits when quantizing. */
export function weightBytes(m: ModelSpec, f: WeightFormat) {
  const bits = effectiveBits(m, f);
  const total = m.paramsB * 1e9;
  if (bits >= 16) return { bytes: (total * bits) / 8, quantized: total, unquantized: 0, bits };
  const keep = Math.min(total, embeddingParams(m) + m.extraUnquantizedParamsB * 1e9);
  return { bytes: ((total - keep) * bits) / 8 + keep * 2, quantized: total - keep, unquantized: keep, bits };
}

export function isMoE(m: ModelSpec): boolean {
  return m.activeParamsB !== undefined && m.activeParamsB < m.paramsB * 0.8;
}

/** Cache-bearing layers. */
export function totalLayers(m: ModelSpec): number {
  return m.layers.reduce((a, g) => a + g.count, 0);
}

/** Decoder layers (used to distribute weights in PP). */
export function decoderLayers(m: ModelSpec): number {
  return Math.max(m.numLayers ?? 0, totalLayers(m));
}

function groupLabel(g: LayerGroup, t: ReturnType<typeof tr>): string {
  switch (g.kind) {
    case "full":
      return t("Attention globale", "Global attention");
    case "sliding":
      return t("Fenêtre glissante", "Sliding window");
    case "mla":
      return g.kvShared ? t("MLA (latent partagé)", "MLA (shared latent)") : t("MLA (latent compressé)", "MLA (compressed latent)");
    case "linear":
      return g.label ?? t("Linéaire / SSM (état constant)", "Linear / SSM (constant state)");
    case "compressed":
      return g.compressRatio
        ? t(`Attention compressée (1/${g.compressRatio})`, `Compressed attention (1/${g.compressRatio})`)
        : t("Fenêtre glissante seule", "Sliding window only");
  }
}

export function windowLabel(window: number): string {
  return window < 1024 ? `${window} tok.` : contextLabel(window);
}

/** Tokens retained by a sliding-window layer, depending on the engine. */
export function storedTokens(g: LayerGroup, ctx: number, engine: EngineSpec): number {
  if (g.kind === "sliding" && slidingWindowSaves(engine)) return Math.min(ctx, g.window);
  if (g.kind === "compressed") return compressedEntries(g, ctx) + Math.min(ctx, g.window);
  return ctx;
}

/** Compressed entries stored by a compressed-attention layer for ctx tokens. */
export function compressedEntries(g: Extract<LayerGroup, { kind: "compressed" }>, ctx: number): number {
  return g.compressRatio ? Math.ceil(ctx / g.compressRatio) : 0;
}

/** Shard a layer group's cache across GPUs in TP. */
function tpSharding(g: LayerGroup, tp: number): { perGpu: number; replication: number } {
  if (tp === 1) return { perGpu: 1, replication: 1 };
  switch (g.kind) {
    case "full":
    case "sliding": {
      const headsPerGpu = Math.ceil(g.kvHeads / tp);
      return { perGpu: headsPerGpu / g.kvHeads, replication: (headsPerGpu * tp) / g.kvHeads };
    }
    case "mla":
    case "compressed":
      // The MLA latent (or single shared KV head) has no head dimension: each TP rank keeps the whole cache.
      return { perGpu: 1, replication: tp };
    case "linear":
      return { perGpu: 1 / tp, replication: 1 };
  }
}

/** TP degree compatible with the model's heads (vLLM constraints). */
export function isValidTp(m: ModelSpec, tp: number): boolean {
  if (tp === 1) return true;
  if (m.attentionHeads % tp !== 0) return false;
  return m.layers.every((g) => (g.kind !== "full" && g.kind !== "sliding") || g.kvHeads % tp === 0 || tp % g.kvHeads === 0);
}

export function estimate(input: EstimateInput): Estimate {
  const { model: m, gpu, engine, gpuCount: n, contextTokens: ctx, weightFormat, kvFormat, mode } = input;
  const tp = mode === "dp-ep" ? 1 : input.tp;
  const pp = mode === "dp-ep" ? 1 : Math.max(1, Math.round(n / tp));
  const kvB = kvFormat.bytesPerElement;
  const t = tr(input.lang ?? "fr");
  const B = t("o", "B");

  // --- Weights (effective memory format)
  const w = weightBytes(m, input.memoryWeightFormat ?? weightFormat);
  /** Bytes for params parameters: selected precision when quantized, otherwise 16 bits. */
  const bytesOf = (params: number, quantized: boolean) => (params * (quantized || w.bits >= 16 ? w.bits : 16)) / 8;
  const embBytes = bytesOf(embeddingParams(m), false);
  const L = decoderLayers(m);
  const layersPerStage = Math.ceil(L / pp);
  let weightsPerGpu = w.bytes / n;
  let nonExpertBytes = w.bytes;
  let replicatedBytes = 0;
  let dpUnvalidated = false;
  if (mode === "dp-ep" && n > 1) {
    if (m.expertParamsB !== undefined) {
      // Shard routed experts; everything else (attention, embeddings, dense layers,
      // shared experts, router) is stored on each rank at its actual precision.
      const expertBytes = bytesOf(Math.min(m.expertParamsB * 1e9, w.quantized), true);
      nonExpertBytes = w.bytes - expertBytes;
      weightsPerGpu = nonExpertBytes + expertBytes / n;
    } else {
      dpUnvalidated = true;
      nonExpertBytes = w.bytes;
      weightsPerGpu = w.bytes;
    }
    replicatedBytes = nonExpertBytes * (n - 1);
  } else if (pp > 1) {
    // Whole layers per stage: size for the most heavily loaded rank (ceil(L / pp) layers),
    // plus its input or output embeddings (copied at both ends when tied).
    const embSide = m.tiedEmbeddings ? embBytes : embBytes / 2;
    const body = w.bytes - embBytes;
    weightsPerGpu = ((body * layersPerStage) / L + embSide) / tp;
    if (m.tiedEmbeddings) replicatedBytes = embBytes;
  }

  // --- KV cache per request
  const groups: GroupBreakdown[] = [];
  const perLayer: { cost: number; group: number }[] = [];
  let perTokenBytes = 0;
  let constantBytes = 0;
  m.layers.forEach((g, gi) => {
    let bytes: number;
    let detail: string;
    switch (g.kind) {
      case "full":
      case "sliding": {
        const dv = g.headDimV ?? g.headDim;
        const tokens = storedTokens(g, ctx, engine);
        const perTok = g.kvHeads * (g.headDim + dv) * kvB + (g.extraBytesPerToken ?? 0);
        bytes = g.count * perTok * tokens;
        if (tokens === ctx) perTokenBytes += g.count * perTok;
        detail = t(`${g.kvHeads} têtes KV × (${g.headDim}+${dv})`, `${g.kvHeads} KV heads × (${g.headDim}+${dv})`);
        if (g.kind === "sliding")
          detail += slidingWindowSaves(engine)
            ? t(` · fenêtre ${windowLabel(g.window)}`, ` · ${windowLabel(g.window)} window`)
            : t(` · fenêtre ${windowLabel(g.window)}, blocs non libérés`, ` · ${windowLabel(g.window)} window, blocks not freed`);
        break;
      }
      case "mla": {
        // A layer sharing an earlier layer's latent stores only its own indexer keys.
        const perTok = (g.kvShared ? 0 : (g.kvLoraRank + g.ropeHeadDim) * kvB) + (g.extraBytesPerToken ?? 0);
        bytes = g.count * perTok * ctx;
        perTokenBytes += g.count * perTok;
        detail = g.kvShared
          ? t("latent d'une couche précédente, non stocké", "earlier layer's latent, not stored")
          : `latent ${g.kvLoraRank}${g.ropeHeadDim ? ` + RoPE ${g.ropeHeadDim}` : " (NoPE)"}`;
        if (g.extraBytesPerToken) detail += t(` + ${g.extraBytesPerToken} o/tok (indexeur)`, ` + ${g.extraBytesPerToken} B/tok (indexer)`);
        break;
      }
      case "linear": {
        bytes = g.count * (g.ssmElements * m.ssmStateBytes + g.convElements * m.convStateBytes);
        constantBytes += bytes;
        detail = `${t("état", "state")} ${(g.ssmElements / 1e6).toFixed(2)} M × ${m.ssmStateBytes} ${B}${g.convElements ? ` + conv ${(g.convElements / 1e3).toFixed(0)} k × ${m.convStateBytes} ${B}` : ""}`;
        break;
      }
      case "compressed": {
        const perEntry = g.headDim * kvB + (g.extraBytesPerEntry ?? 0);
        bytes = g.count * (compressedEntries(g, ctx) * perEntry + Math.min(ctx, g.window) * g.headDim * kvB);
        if (g.compressRatio) perTokenBytes += (g.count * perEntry) / g.compressRatio;
        detail = g.compressRatio
          ? g.compressRatio > 1
            ? t(`KV ${g.headDim} (K = V) pour ${g.compressRatio} tokens`, `KV ${g.headDim} (K = V) per ${g.compressRatio} tokens`)
            : t(`KV ${g.headDim} (K = V) par token`, `KV ${g.headDim} (K = V) per token`)
          : t(`KV ${g.headDim} (K = V)`, `KV ${g.headDim} (K = V)`);
        if (g.extraBytesPerEntry) detail += t(` + ${g.extraBytesPerEntry} o/entrée (indexeur)`, ` + ${g.extraBytesPerEntry} B/entry (indexer)`);
        detail += t(` · fenêtre ${windowLabel(g.window)}`, ` · ${windowLabel(g.window)} window`);
        break;
      }
    }
    const s = mode === "dp-ep" || n === 1 ? { perGpu: 1, replication: 1 } : tpSharding(g, tp);
    for (let k = 0; k < g.count; k++) perLayer.push({ cost: (bytes / g.count) * s.perGpu, group: gi });
    groups.push({ kind: g.kind, label: groupLabel(g, t), count: g.count, detail, bytes, perGpuBytes: 0, replication: s.replication });
  });
  // Most heavily loaded rank: at most ceil(L / pp) layers, choosing the most expensive (conservative bound,
  // since the exact layer order is unknown).
  perLayer.sort((a, b) => b.cost - a.cost);
  for (const l of perLayer.slice(0, Math.min(layersPerStage, perLayer.length))) groups[l.group].perGpuBytes += l.cost;
  const perRequestBytes = groups.reduce((a, g) => a + g.bytes, 0);
  const perRequestPerGpuBytes = groups.reduce((a, g) => a + g.perGpuBytes, 0);
  const perRequestClusterBytes = mode === "dp-ep" || n === 1 ? perRequestBytes : groups.reduce((a, g) => a + g.bytes * g.replication, 0);

  // --- Memory
  const gpuBytes = (input.memoryGiBOverride ?? gpu.memoryGiB) * GIB;
  const reservePerGpuBytes = gpuBytes * (input.reservePct / 100);
  const usablePerGpuBytes = gpuBytes - reservePerGpuBytes;
  const freeForKvPerGpuBytes = Math.max(0, usablePerGpuBytes - weightsPerGpu);
  const fitsWeights = weightsPerGpu <= usablePerGpuBytes;

  const sessionsPerGpu = perRequestPerGpuBytes > 0 ? Math.floor(freeForKvPerGpuBytes / perRequestPerGpuBytes) : 0;
  const sessions = mode === "dp-ep" ? sessionsPerGpu * n : sessionsPerGpu;

  // --- CPU offload
  const offloadTotal = engine.features.kvOffload !== false ? (input.offloadGiBPerGpu ?? 0) * GIB * n : 0;
  const parkedSessions = perRequestClusterBytes > 0 ? Math.floor(offloadTotal / perRequestClusterBytes) : 0;
  const reloadMs = (perRequestPerGpuBytes / (gpu.hostLinkGBs * 1e9)) * 1000;

  return {
    tp,
    pp,
    layersPerStage,
    weights: {
      bitsPerParam: w.bits,
      totalBytes: w.bytes,
      perGpuBytes: weightsPerGpu,
      quantizedParams: w.quantized,
      unquantizedParams: w.unquantized,
      nonExpertBytes,
      replicatedBytes,
      dpUnvalidated,
    },
    kv: { perRequestBytes, perRequestClusterBytes, perRequestPerGpuBytes, perTokenBytes, constantBytes, groups },
    memory: {
      gpuBytes,
      reservePerGpuBytes,
      usablePerGpuBytes,
      freeForKvPerGpuBytes,
      freeForKvTotalBytes: Math.max(0, usablePerGpuBytes * n - (w.bytes + replicatedBytes)),
      kvBudgetTotalBytes: freeForKvPerGpuBytes * n,
    },
    offload: { totalBytes: offloadTotal, parkedSessions, reloadMs },
    modelPlusOneKvBytes: w.bytes + perRequestBytes,
    physicalOneRequestBytes: w.bytes + replicatedBytes + perRequestClusterBytes,
    fitsWeights,
    fitsOneRequest: fitsWeights && sessions >= 1,
    sessions,
    sessionsPerGpu,
  };
}

const LEVEL_LABELS: Record<Lang, Record<SupportLevel, string>> = {
  fr: { native: "natif", limited: "limité", unverified: "non vérifié", unsupported: "non supporté" },
  en: { native: "native", limited: "limited", unverified: "not verified", unsupported: "not supported" },
};
export const supportLevelLabel = (l: SupportLevel, lang: Lang = "fr") => LEVEL_LABELS[lang][l];

/** List limitations and warnings for a configuration, in input.lang. */
export function collectIssues(input: EstimateInput, est: Estimate, weightFormats: WeightFormat[]): Issue[] {
  const { model: m, gpu, engine, gpuCount: n, contextTokens: ctx, weightFormat: wf, kvFormat: kf, mode } = input;
  const lang = input.lang ?? "fr";
  const t = tr(lang);
  const LEVEL = LEVEL_LABELS[lang];
  const gibText = (b: number) =>
    b >= GIB ? `${(b / GIB).toFixed(1).replace(".", t(",", "."))} ${t("Gio", "GiB")}` : `${Math.round(b / 1024 ** 2)} ${t("Mio", "MiB")}`;
  const { tp, pp } = est;
  const out: Issue[] = [];

  // --- Capacity
  if (!est.fitsWeights) {
    out.push({
      severity: "error",
      title: t("Les poids ne tiennent pas en mémoire", "The weights do not fit in memory"),
      detail: t(
        `${gibText(est.weights.perGpuBytes)} de poids par GPU pour ${gibText(est.memory.usablePerGpuBytes)} utilisables. Ajoutez des GPUs, quantifiez davantage ou réduisez la réserve.`,
        `${gibText(est.weights.perGpuBytes)} of weights per GPU for ${gibText(est.memory.usablePerGpuBytes)} usable. Add GPUs, quantize further or reduce the reserve.`,
      ),
    });
  } else if (est.sessions < 1) {
    out.push({
      severity: "error",
      title: t("Pas assez de mémoire pour une seule requête à ce contexte", "Not enough memory for a single request at this context"),
      detail: t(
        `Il reste ${gibText(est.memory.freeForKvPerGpuBytes)} par GPU, et une requête en demande ${gibText(est.kv.perRequestPerGpuBytes)} par GPU.`,
        `${gibText(est.memory.freeForKvPerGpuBytes)} per GPU remain, and one request needs ${gibText(est.kv.perRequestPerGpuBytes)} per GPU.`,
      ),
    });
  }

  // --- Engine
  const eg = engine.gpus[gpu.id];
  if (eg && eg.level !== "native")
    out.push({
      severity: eg.level === "unsupported" ? "error" : "warning",
      title: t(`${engine.name} sur ${gpu.name} : ${LEVEL[eg.level]}`, `${engine.name} on ${gpu.name}: ${LEVEL[eg.level]}`),
      detail: eg.note,
    });
  if (engine.modelFamilies && !engine.modelFamilies.includes(m.family))
    out.push({
      severity: "warning",
      title: t(`Famille ${m.family} non annoncée par ${engine.name}`, `${m.family} family not advertised by ${engine.name}`),
      detail: t(
        `Familles annoncées : ${engine.modelFamilies.join(", ")}. Le modèle peut fonctionner, mais rien ne le garantit.`,
        `Advertised families: ${engine.modelFamilies.join(", ")}. The model may work, but nothing guarantees it.`,
      ),
    });
  if (engine.confidence !== "high")
    out.push({ severity: "warning", title: t(`Profil ${engine.name} partiellement vérifié`, `${engine.name} profile partially verified`), detail: engine.notes[0] });
  const fe = engine.features;
  if (fe.kvPaging === false)
    out.push({
      severity: "info",
      title: t(`KV non paginé dans ${engine.name}`, `KV not paged in ${engine.name}`),
      detail: t(
        "Chaque requête réserve son contexte maximal : c'est déjà l'hypothèse du calcul (pire cas).",
        "Each request reserves its maximum context: the calculation already assumes this (worst case).",
      ),
    });
  else if (fe.kvPaging === "unverified")
    out.push({
      severity: "info",
      title: t(`Pagination du KV non vérifiée dans ${engine.name}`, `KV paging not verified in ${engine.name}`),
      detail: t("Sans effet sur ce calcul, fait au contexte maximal pour chaque session.", "No effect on this calculation, which uses the maximum context for every session."),
    });
  if (fe.kvPoolPreallocated === "unverified")
    out.push({
      severity: "info",
      title: t(`Préallocation du pool KV non vérifiée dans ${engine.name}`, `KV pool preallocation not verified in ${engine.name}`),
      detail: t(
        "Si le pool est dimensionné autrement que par la réserve choisie, la capacité réelle diffère.",
        "If the pool is sized other than by the selected reserve, the actual capacity differs.",
      ),
    });

  // --- Formats vs GPU and engine
  const ws = weightSupport(gpu, engine, wf.id, lang);
  if (ws.level !== "native")
    out.push({ severity: ws.level === "unsupported" ? "error" : "warning", title: t(`Poids ${wf.label} : ${LEVEL[ws.level]}`, `${wf.label} weights: ${LEVEL[ws.level]}`), detail: ws.note });
  else if (ws.note) out.push({ severity: "info", title: t(`Poids ${wf.label}`, `${wf.label} weights`), detail: ws.note });
  const memFmt = input.memoryWeightFormat;
  if (memFmt && memFmt.id !== wf.id)
    out.push({
      severity: "info",
      title: t(`Fichier ${wf.label} chargé en ${memFmt.label} en mémoire`, `${wf.label} file loaded as ${memFmt.label} in memory`),
      detail: t(
        `${engine.name} déquantifie les poids au chargement : ${gibText(est.weights.totalBytes)} en VRAM au lieu de ${gibText(weightBytes(m, wf).bytes)}.`,
        `${engine.name} dequantizes the weights at load time: ${gibText(est.weights.totalBytes)} in VRAM instead of ${gibText(weightBytes(m, wf).bytes)}.`,
      ),
    });
  else if (ws.level === "unverified" && wf.bitsPerParam < 16) {
    const bf16 = gibText(weightBytes(m, { ...wf, id: "bf16", bitsPerParam: 16 }).bytes);
    out.push({
      severity: "info",
      title: t(`Format en mémoire non vérifié pour ${wf.label}`, `In-memory format not verified for ${wf.label}`),
      detail: t(
        `Calcul fait en supposant que les poids restent en ${wf.label}. S'ils sont déquantifiés en BF16 au chargement : ${bf16}.`,
        `Calculated assuming the weights stay in ${wf.label}. If they are dequantized to BF16 at load time: ${bf16}.`,
      ),
    });
  }

  const ks = kvSupport(gpu, engine, kf.id, lang);
  if (ks.level !== "native")
    out.push({ severity: ks.level === "unsupported" ? "error" : "warning", title: t(`KV cache ${kf.label} : ${LEVEL[ks.level]}`, `${kf.label} KV cache: ${LEVEL[ks.level]}`), detail: ks.note });
  else if (ks.note) out.push({ severity: "info", title: t(`KV cache ${kf.label}`, `${kf.label} KV cache`), detail: ks.note });

  // --- Formats vs model
  const native = m.nativeWeightFormat;
  const wNote = m.weightFormatNotes[wf.id];
  if (wNote) out.push({ severity: wNote.level === "unsupported" ? "error" : wNote.level, title: t(`${m.name} en ${wf.label}`, `${m.name} in ${wf.label}`), detail: wNote.note });
  const nativeFmt = weightFormats.find((f) => f.id === native);
  const nativeFmtBits = nativeFmt ? effectiveBits(m, nativeFmt) : undefined;
  // Same width (e.g. BF16 to FP16): conversion only, without quantization.
  if (wf.id !== native && nativeFmtBits !== wf.bitsPerParam) {
    if (nativeFmtBits !== undefined && wf.bitsPerParam > nativeFmtBits)
      out.push({
        severity: "info",
        title: t(`Poids publiés nativement en ${nativeFmt?.label ?? native}`, `Weights natively published in ${nativeFmt?.label ?? native}`),
        detail: t(`Charger en ${wf.label} augmente la mémoire sans gain de qualité.`, `Loading in ${wf.label} increases memory without any quality gain.`),
      });
    else if (!m.officialWeightFormats.includes(wf.id))
      out.push({
        severity: "info",
        title: t(`Pas de checkpoint officiel en ${wf.label}`, `No official ${wf.label} checkpoint`),
        detail: t(
          "Quantification communautaire ou maison : vérifiez la disponibilité et l'impact qualité.",
          "Community or in-house quantization: check availability and quality impact.",
        ),
      });
  }
  const kNote = m.kvFormatNotes[kf.id];
  if (kNote) out.push({ severity: kNote.level === "unsupported" ? "error" : kNote.level, title: t(`KV ${kf.label} avec ${m.name}`, `${kf.label} KV with ${m.name}`), detail: kNote.note });

  // --- Context
  if (ctx > m.maxContext) {
    if (m.maxContextExtended && ctx <= m.maxContextExtended)
      out.push({
        severity: "warning",
        title: t(`Contexte au-delà du natif (${contextLabel(m.maxContext)})`, `Context beyond native (${contextLabel(m.maxContext)})`),
        detail: t(
          `Possible jusqu'à ${contextLabel(m.maxContextExtended)} avec extension RoPE (YaRN). La qualité peut baisser sur les textes courts.`,
          `Possible up to ${contextLabel(m.maxContextExtended)} with RoPE extension (YaRN). Quality may drop on short texts.`,
        ),
      });
    else
      out.push({
        severity: "error",
        title: t(
          `Contexte supérieur au maximum du modèle (${contextLabel(m.maxContextExtended ?? m.maxContext)})`,
          `Context above the model maximum (${contextLabel(m.maxContextExtended ?? m.maxContext)})`,
        ),
        detail: t("Le calcul reste indicatif : le modèle n'est pas prévu pour cette longueur.", "The result is only indicative: the model is not designed for this length."),
      });
  }
  const sliding = m.layers.filter((g): g is Extract<LayerGroup, { kind: "sliding" }> => g.kind === "sliding");
  if (sliding.length) {
    const cnt = sliding.reduce((a, g) => a + g.count, 0);
    const win = Math.max(...sliding.map((g) => g.window));
    const f = engine.features;
    if (f.slidingWindowAttention === false)
      out.push({
        severity: "error",
        title: t(`Attention glissante non supportée par ${engine.name}`, `Sliding-window attention not supported by ${engine.name}`),
        detail: t(
          `${cnt} couches de ${m.name} utilisent une fenêtre de ${windowLabel(win)} : sans support, les résultats du modèle sont faux.`,
          `${cnt} layers of ${m.name} use a ${windowLabel(win)} window: without support, the model outputs are wrong.`,
        ),
      });
    else if (slidingWindowSaves(engine))
      out.push({
        severity: "info",
        title: t(`${cnt} couches à fenêtre glissante (${windowLabel(win)})`, `${cnt} sliding-window layers (${windowLabel(win)})`),
        detail: t(
          `${engine.name} libère les blocs KV sortis de la fenêtre : KV plafonné à la fenêtre.`,
          `${engine.name} frees KV blocks that leave the window: KV capped at the window.`,
        ),
      });
    else
      out.push({
        severity: "warning",
        title: t(`${cnt} couches à fenêtre glissante comptées au contexte complet`, `${cnt} sliding-window layers counted at full context`),
        detail: [
          f.slidingWindowAttention === "unverified"
            ? t(`Support de l'attention glissante non vérifié dans ${engine.name}.`, `Sliding-window attention support not verified in ${engine.name}.`)
            : t(`Attention glissante supportée par ${engine.name}.`, `Sliding-window attention supported by ${engine.name}.`),
          f.slidingWindowFreesBlocks === false
            ? t("Les blocs hors fenêtre ne sont pas libérés : aucune économie de VRAM.", "Blocks outside the window are not freed: no VRAM savings.")
            : t(
                "Libération des blocs hors fenêtre non vérifiée : aucune économie de VRAM appliquée.",
                "Freeing of blocks outside the window not verified: no VRAM savings applied.",
              ),
          engine.featureNotes.slidingWindowFreesBlocks ?? engine.featureNotes.slidingWindowAttention,
        ]
          .filter(Boolean)
          .join(" "),
      });
  }
  if (m.layers.some((g) => g.kind === "linear"))
    out.push({
      severity: "info",
      title: t("Couches linéaires / SSM : état récurrent constant", "Linear / SSM layers: constant recurrent state"),
      detail: t(
        `${gibText(est.kv.constantBytes)} par requête quel que soit le contexte (état en ${m.ssmStateBytes * 8} bits), indépendamment du format KV choisi.`,
        `${gibText(est.kv.constantBytes)} per request whatever the context (${m.ssmStateBytes * 8}-bit state), regardless of the selected KV format.`,
      ),
    });
  if (m.layers.some((g) => g.kind === "mla" && (g.extraBytesPerToken ?? 0) > 0))
    out.push({
      severity: "info",
      title: t("Index d'attention parcimonieuse (DSA) inclus", "Sparse attention index (DSA) included"),
      detail: t("Clés de l'indexeur stockées en FP8 + échelle, en plus du latent.", "Indexer keys stored in FP8 + scale, in addition to the latent."),
    });
  if (m.layers.some((g) => g.kind === "full" && (g.extraBytesPerToken ?? 0) > 0))
    out.push({
      severity: "info",
      title: t("Index d'attention parcimonieuse inclus", "Sparse attention index included"),
      detail: t(
        "Clés de l'indexeur comptées à taille fixe par token, en plus du KV et quel que soit le format KV choisi.",
        "Indexer keys counted at a fixed size per token, in addition to the KV and regardless of the selected KV format.",
      ),
    });
  const compressed = m.layers.filter((g): g is Extract<LayerGroup, { kind: "compressed" }> => g.kind === "compressed");
  if (compressed.length)
    out.push({
      severity: "info",
      title: t("Attention compressée : une entrée KV pour plusieurs tokens", "Compressed attention: one KV entry for several tokens"),
      detail: t(
        `Chaque couche garde ses ${windowLabel(Math.max(...compressed.map((g) => g.window)))} récents non compressés ; les entrées compressées et l'indexeur suivent le contexte. L'état FP32 des compresseurs (quelques Mo par requête) n'est pas compté.`,
        `Each layer keeps its most recent ${windowLabel(Math.max(...compressed.map((g) => g.window)))} uncompressed; compressed entries and the indexer grow with the context. The compressors' FP32 state (a few MB per request) is not counted.`,
      ),
    });

  // --- Interconnect
  const linkName = (k: number) =>
    gpu.interconnect.nvlink && k <= gpu.nvlinkDomain ? "NVLink" : k <= gpu.maxGpusPerNode ? gpu.interconnect.label : t("le réseau inter-nœuds", "the inter-node network");
  if (n > 1) {
    if (mode === "tp-pp" && tp > 1 && !gpu.interconnect.nvlink)
      out.push({
        severity: "warning",
        title: t(`TP=${tp} sans NVLink sur ${gpu.name}`, `TP=${tp} without NVLink on ${gpu.name}`),
        detail: t(
          `Le TP fait deux all-reduce par couche via ${gpu.interconnect.label} : débit et latence nettement dégradés. Préférez PP (TP=1), ou plusieurs instances indépendantes.`,
          `TP performs two all-reduces per layer over ${gpu.interconnect.label}: throughput and latency are much worse. Prefer PP (TP=1), or several independent instances.`,
        ),
      });
    if (mode === "tp-pp" && tp > 1 && gpu.interconnect.nvlink && tp > gpu.nvlinkDomain)
      out.push({
        severity: "warning",
        title: t(`TP=${tp} dépasse le domaine NVLink (${gpu.nvlinkDomain})`, `TP=${tp} exceeds the NVLink domain (${gpu.nvlinkDomain})`),
        detail: t(
          "Les all-reduce passent par le réseau. Gardez le TP dans le domaine NVLink et utilisez PP au-delà.",
          "All-reduces go over the network. Keep TP within the NVLink domain and use PP beyond it.",
        ),
      });
    if (mode === "dp-ep" && n > Math.max(gpu.nvlinkDomain, 1) && n > gpu.maxGpusPerNode)
      out.push({
        severity: "info",
        title: t("Expert parallel inter-nœuds", "Inter-node expert parallel"),
        detail: t(
          "Les all-to-all MoE traversent le réseau : prévoyez un réseau RDMA rapide (InfiniBand / RoCE).",
          "MoE all-to-alls cross the network: plan for a fast RDMA network (InfiniBand / RoCE).",
        ),
      });
    if (n > gpu.maxGpusPerNode)
      out.push({
        severity: n > gpu.nvlinkDomain ? "warning" : "info",
        title: t(
          `${n} GPUs : ${Math.ceil(n / gpu.maxGpusPerNode)} nœuds de ${gpu.maxGpusPerNode} GPUs`,
          `${n} GPUs: ${Math.ceil(n / gpu.maxGpusPerNode)} nodes of ${gpu.maxGpusPerNode} GPUs`,
        ),
        detail:
          n <= gpu.nvlinkDomain
            ? t(`Tous dans le même domaine NVLink (${gpu.nvlinkDomain} GPUs).`, `All in the same NVLink domain (${gpu.nvlinkDomain} GPUs).`)
            : t(`Liaison entre nœuds par ${linkName(n)}.`, `Nodes linked by ${linkName(n)}.`),
      });
  }

  // --- Parallelism
  if (n > 1 && mode === "tp-pp") {
    const cap = (k: "tensorParallel" | "pipelineParallel", used: boolean, label: string) => {
      const v = engine.features[k];
      if (!used || v === true) return;
      out.push({
        severity: v === false ? "error" : "warning",
        title: t(`${label} ${v === false ? "non supporté" : "non vérifié"} dans ${engine.name}`, `${label} ${v === false ? "not supported" : "not verified"} in ${engine.name}`),
        detail: engine.featureNotes[k],
        unverifiedParallelism: v === "unverified",
      });
    };
    cap("tensorParallel", tp > 1, "Tensor parallel");
    cap("pipelineParallel", pp > 1, "Pipeline parallel");
    if (tp > 1 && m.attentionHeads % tp !== 0)
      out.push({
        severity: "error",
        title: t(
          `TP=${tp} impossible : ${m.attentionHeads} têtes d'attention non divisibles par ${tp}`,
          `TP=${tp} impossible: ${m.attentionHeads} attention heads not divisible by ${tp}`,
        ),
        detail: t(`Choisissez un degré de TP diviseur de ${m.attentionHeads} et complétez avec du PP.`, `Choose a TP degree that divides ${m.attentionHeads} and complete with PP.`),
      });
    for (const g of m.layers) {
      if (g.kind !== "full" && g.kind !== "sliding") continue;
      if (tp > 1 && g.kvHeads % tp !== 0 && tp % g.kvHeads !== 0)
        out.push({
          severity: "error",
          title: t(`TP=${tp} incompatible avec ${g.kvHeads} têtes KV`, `TP=${tp} incompatible with ${g.kvHeads} KV heads`),
          detail: t("Le nombre de têtes KV doit être multiple ou diviseur du degré de TP.", "The number of KV heads must be a multiple or a divisor of the TP degree."),
        });
      else if (tp > g.kvHeads) {
        out.push({
          severity: "warning",
          title: t(`KV répliqué ×${tp / g.kvHeads} (${g.kvHeads} têtes KV pour TP=${tp})`, `KV replicated ×${tp / g.kvHeads} (${g.kvHeads} KV heads for TP=${tp})`),
          detail: t(
            "Chaque tête KV est dupliquée sur plusieurs GPUs. Réduisez le TP et ajoutez du PP, ou passez en DP-attention.",
            "Each KV head is duplicated on several GPUs. Reduce TP and add PP, or switch to DP-attention.",
          ),
        });
        break;
      }
    }
    if (tp > 1 && m.layers.some((g) => g.kind === "mla")) {
      const flag = engine.flags.dpAttention ? ` (${engine.flags.dpAttention})` : "";
      out.push({
        severity: "warning",
        title: t(`Cache MLA répliqué sur les ${tp} GPUs du TP`, `MLA cache replicated on the ${tp} TP GPUs`),
        detail: t(
          `Chaque rang TP stocke le latent complet. En DP-attention${flag}, chaque GPU garde ses propres requêtes : capacité ≈ ×${tp}.`,
          `Each TP rank stores the full latent. With DP-attention${flag}, each GPU keeps its own requests: capacity ≈ ×${tp}.`,
        ),
      });
    }
    if (pp > 1) {
      const L = decoderLayers(m);
      if (pp > L)
        out.push({
          severity: "error",
          title: t(`PP=${pp} supérieur au nombre de couches (${L})`, `PP=${pp} greater than the number of layers (${L})`),
          detail: t("Chaque étage doit porter au moins une couche entière.", "Each stage must hold at least one whole layer."),
        });
      else if (L % pp !== 0)
        out.push({
          severity: "info",
          title: t(`PP=${pp} : jusqu'à ${est.layersPerStage} couches sur l'étage le plus chargé`, `PP=${pp}: up to ${est.layersPerStage} layers on the most loaded stage`),
          detail: t(
            `${L} couches non divisibles par ${pp} : poids et KV calculés pour le rang le plus chargé (couches entières, les plus coûteuses en KV).`,
            `${L} layers not divisible by ${pp}: weights and KV calculated for the most loaded rank (whole layers, the most KV-expensive ones).`,
          ),
        });
      if (m.tiedEmbeddings)
        out.push({
          severity: "info",
          title: t("Embeddings liés en PP", "Tied embeddings with PP"),
          detail: t(
            `Copie sur le premier et le dernier étage (+${gibText(est.weights.replicatedBytes)}).`,
            `Copied on the first and last stages (+${gibText(est.weights.replicatedBytes)}).`,
          ),
        });
      if (n % tp !== 0)
        out.push({
          severity: "error",
          title: t(`${n} GPUs non divisibles par TP=${tp}`, `${n} GPUs not divisible by TP=${tp}`),
          detail: t("TP × PP doit être égal au nombre de GPUs.", "TP × PP must equal the number of GPUs."),
        });
    }
  }
  if (mode === "dp-ep") {
    if (engine.features.dpAttention === false)
      out.push({ severity: "error", title: t(`DP-attention indisponible dans ${engine.name}`, `DP-attention unavailable in ${engine.name}`), detail: t("Choisissez TP/PP.", "Choose TP/PP.") });
    else if (engine.features.dpAttention === "unverified")
      out.push({
        severity: "warning",
        title: t(`DP-attention non vérifiée dans ${engine.name}`, `DP-attention not verified in ${engine.name}`),
        detail: engine.featureNotes.dpAttention ?? t("Aucune source ne confirme ce mode : résultat théorique.", "No source confirms this mode: theoretical result."),
        unverifiedParallelism: true,
      });
    if (!isMoE(m))
      out.push({
        severity: "warning",
        title: t("DP-attention conçu pour les MoE", "DP-attention is designed for MoE models"),
        detail: t("Sur un modèle dense, cela revient à N répliques complètes.", "On a dense model, this amounts to N full replicas."),
      });
    else if (n > 1 && est.weights.dpUnvalidated)
      out.push({
        severity: "error",
        title: t(`DP-attention : paramètres des experts inconnus pour ${m.name}`, `DP-attention: expert parameters unknown for ${m.name}`),
        detail: t(
          "Impossible de séparer experts répartis et tensors répliqués : capacité non validée (calcul avec le modèle complet par GPU). Renseignez expertParamsB dans le catalogue.",
          "Sharded experts cannot be separated from replicated tensors: capacity not validated (calculated with the full model per GPU). Set expertParamsB in the catalogue.",
        ),
      });
    else if (n > 1)
      out.push({
        severity: "info",
        title: "DP-attention + expert parallel",
        detail: t(
          `Seuls les experts routés sont répartis. Attention, embeddings, couches denses et experts partagés (${gibText(est.weights.nonExpertBytes)}, à leur précision réelle) sont présents sur chaque GPU : +${gibText(est.weights.replicatedBytes)} au total. Chaque requête vit sur un seul GPU.`,
          `Only routed experts are sharded. Attention, embeddings, dense layers and shared experts (${gibText(est.weights.nonExpertBytes)}, at their actual precision) are present on every GPU: +${gibText(est.weights.replicatedBytes)} in total. Each request lives on a single GPU.`,
        ),
      });
  }

  // --- Offload
  if ((input.offloadGiBPerGpu ?? 0) > 0) {
    if (engine.features.kvOffload === false)
      out.push({
        severity: "warning",
        title: t(`Offload KV non disponible dans ${engine.name}`, `KV offload unavailable in ${engine.name}`),
        detail: t("La mémoire CPU saisie n'est pas comptée.", "The CPU memory entered is not counted."),
      });
    else {
      if (engine.features.kvOffload === "unverified")
        out.push({
          severity: "warning",
          title: t(`Offload KV non vérifié dans ${engine.name}`, `KV offload not verified in ${engine.name}`),
          detail: engine.featureNotes.kvOffload ?? t("Sessions en réserve théoriques.", "Parked sessions are theoretical."),
        });
      const reload = `${Math.round(est.offload.reloadMs)} ms`;
      out.push({
        severity: "info",
        title: t(`Offload CPU : ${est.offload.parkedSessions} sessions en réserve`, `CPU offload: ${est.offload.parkedSessions} parked sessions`),
        detail: t(
          `Le KV déchargé n'est pas lu pendant le décodage : il sert aux sessions en attente, préemptées ou à réutiliser (prefix cache). Rechargement d'une session ≈ ${reload} via ${gpu.hostLinkLabel}.`,
          `Offloaded KV is not read during decoding: it serves waiting, preempted or reusable sessions (prefix cache). Reloading one session ≈ ${reload} over ${gpu.hostLinkLabel}.`,
        ),
      });
    }
  }

  if (m.confidence !== "high")
    out.push({
      severity: m.confidence === "low" ? "warning" : "info",
      title: t(
        `Architecture de ${m.name} à vérifier (fiabilité ${m.confidence === "low" ? "faible" : "moyenne"})`,
        `${m.name} architecture to be checked (${m.confidence} confidence)`,
      ),
      detail: t(
        `Comparez avec le config.json${m.source ? ` de ${m.source}` : ""} (pnpm run import-model) et corrigez le fichier de configuration si besoin.`,
        `Compare with the config.json${m.source ? ` of ${m.source}` : ""} (pnpm run import-model) and fix the configuration file if needed.`,
      ),
    });

  const order: Record<Severity, number> = { error: 0, warning: 1, info: 2 };
  const seen = new Set<string>();
  return out
    .filter((i) => !seen.has(i.title) && seen.add(i.title))
    .sort((a, b) => order[a.severity] - order[b.severity]);
}
