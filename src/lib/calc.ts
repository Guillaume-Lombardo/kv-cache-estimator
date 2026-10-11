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

export const GIB = 1024 ** 3;

/** 8k → 1M par doublement (puissances de 2). */
export const CONTEXT_STEPS = [8, 16, 32, 64, 128, 256, 512, 1024].map((k) => k * 1024);

export function contextLabel(tokens: number): string {
  if (tokens >= 1024 * 1024) return `${+(tokens / (1024 * 1024)).toFixed(1)}M`;
  return `${Math.round(tokens / 1024)}k`;
}

/** tp-pp : TP dans un étage, PP entre étages (tp = n → TP pur, tp = 1 → PP pur). dp-ep : DP-attention + expert parallel. */
export type ParallelMode = "tp-pp" | "dp-ep";

export interface EstimateInput {
  model: ModelSpec;
  gpu: GpuSpec;
  engine: EngineSpec;
  gpuCount: number;
  mode: ParallelMode;
  /** Degré de TP (mode tp-pp) ; PP = gpuCount / tp. */
  tp: number;
  contextTokens: number;
  /** Format du fichier de poids. */
  weightFormat: WeightFormat;
  /** Format effectif en mémoire GPU, si le moteur ne conserve pas celui du fichier. */
  memoryWeightFormat?: WeightFormat;
  kvFormat: KvFormat;
  /** Part de la mémoire de chaque GPU laissée aux activations, graphes CUDA, fragmentation. */
  reservePct: number;
  /** Mémoire visible par GPU si différente du catalogue (nvidia-smi). */
  memoryGiBOverride?: number;
  /** Mémoire CPU dédiée au déchargement du KV, par GPU. */
  offloadGiBPerGpu?: number;
}

export type Severity = "error" | "warning" | "info";
export interface Issue {
  severity: Severity;
  title: string;
  detail?: string;
}

export interface GroupBreakdown {
  kind: LayerGroup["kind"];
  label: string;
  count: number;
  detail: string;
  /** Octets pour une requête au contexte demandé (vue logique, sans réplication). */
  bytes: number;
  /** Octets stockés sur un GPU pour une requête, selon le parallélisme. */
  perGpuBytes: number;
  /** Facteur de réplication du cache entre GPUs (1 = pas de duplication). */
  replication: number;
}

export interface Estimate {
  tp: number;
  pp: number;
  weights: {
    bitsPerParam: number;
    /** Poids du modèle (une copie). */
    totalBytes: number;
    /** Poids sur le GPU le plus chargé. */
    perGpuBytes: number;
    quantizedParams: number;
    unquantizedParams: number;
    /** DP-attention : tensors non experts présents sur chaque GPU. */
    nonExpertBytes: number;
    /** Copies supplémentaires de poids (DP-attention, embeddings liés en PP). */
    replicatedBytes: number;
    /** DP-attention sans nombre de paramètres des experts : capacité non validée. */
    dpUnvalidated: boolean;
  };
  /** Couches par étage PP sur le rang le plus chargé. */
  layersPerStage: number;
  kv: {
    /** KV d'une requête, vue logique. */
    perRequestBytes: number;
    /** Mémoire réellement consommée par une requête sur l'ensemble des GPUs. */
    perRequestClusterBytes: number;
    /** Mémoire consommée par une requête sur le GPU qui l'héberge. */
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
    /** Mémoire libre physique totale : N × mémoire utilisable − poids physiques (réplications comprises). */
    freeForKvTotalBytes: number;
    /** Budget prudent : mémoire libre du GPU le plus chargé × N (base du nombre de sessions). */
    kvBudgetTotalBytes: number;
  };
  offload: {
    totalBytes: number;
    /** Sessions supplémentaires pouvant être gardées en mémoire CPU (préemptées / en attente). */
    parkedSessions: number;
    /** Temps de rechargement d'une session complète vers le GPU, ms. */
    reloadMs: number;
  };
  /** Somme logique poids + une requête (sans réplication). */
  modelPlusOneKvBytes: number;
  /** Occupation physique réelle (réplications comprises) avec une requête. */
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

/** Support combiné GPU × moteur : le plus restrictif l'emporte, les notes sont cumulées. */
export function combinedSupport(gpuMap: Record<string, Support>, engineMap: Record<string, Support>, id: string, engineName: string): Support {
  const g = supportOf(gpuMap, id);
  const e = supportOf(engineMap, id);
  const level = LEVEL_ORDER[g.level] >= LEVEL_ORDER[e.level] ? g.level : e.level;
  const engineNote = e.note ?? (e.level === "unsupported" ? `Non supporté par ${engineName}.` : e.level === "unverified" ? `Support non vérifié dans ${engineName}.` : undefined);
  const notes = [g.note, engineNote].filter(Boolean);
  return { level, note: notes.length ? notes.join(" ") : undefined, memoryFormat: e.memoryFormat ?? g.memoryFormat };
}

export const weightSupport = (gpu: GpuSpec, engine: EngineSpec, id: string) =>
  combinedSupport(gpu.weightFormats, engine.weightFormats, id, engine.name);
export const kvSupport = (gpu: GpuSpec, engine: EngineSpec, id: string) =>
  combinedSupport(gpu.kvCacheFormats, engine.kvCacheFormats, id, engine.name);

/** Format réellement occupé en mémoire GPU pour un fichier de poids donné. */
export function memoryFormatFor(formats: WeightFormat[], gpu: GpuSpec, engine: EngineSpec, file: WeightFormat): WeightFormat {
  const id = weightSupport(gpu, engine, file.id).memoryFormat;
  return (id && formats.find((f) => f.id === id)) || file;
}

/** L'économie de VRAM des fenêtres glissantes n'est appliquée que si le moteur libère les blocs hors fenêtre. */
export const slidingWindowSaves = (e: EngineSpec) => e.features.slidingWindowAttention === true && e.features.slidingWindowFreesBlocks === true;

export function embeddingParams(m: ModelSpec): number {
  return m.vocabSize * m.hiddenSize * (m.tiedEmbeddings ? 1 : 2);
}

export function effectiveBits(m: ModelSpec, f: WeightFormat): number {
  return f.id === m.nativeWeightFormat && m.nativeBitsPerParam ? m.nativeBitsPerParam : f.bitsPerParam;
}

/** Les embeddings / lm_head (et l'encodeur vision) restent en 16 bits quand on quantifie. */
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

/** Couches portant un cache. */
export function totalLayers(m: ModelSpec): number {
  return m.layers.reduce((a, g) => a + g.count, 0);
}

/** Couches du décodeur (répartition des poids en PP). */
export function decoderLayers(m: ModelSpec): number {
  return Math.max(m.numLayers ?? 0, totalLayers(m));
}

function groupLabel(g: LayerGroup): string {
  switch (g.kind) {
    case "full":
      return "Attention globale";
    case "sliding":
      return "Fenêtre glissante";
    case "mla":
      return "MLA (latent compressé)";
    case "linear":
      return g.label ?? "Linéaire / SSM (état constant)";
  }
}

export function windowLabel(window: number): string {
  return window < 1024 ? `${window} tok.` : contextLabel(window);
}

/** Tokens retenus par une couche à fenêtre glissante, selon le moteur. */
export function storedTokens(g: LayerGroup, ctx: number, engine: EngineSpec): number {
  if (g.kind === "sliding" && slidingWindowSaves(engine)) return Math.min(ctx, g.window);
  return ctx;
}

/** Partage du cache d'un groupe de couches entre GPUs en TP. */
function tpSharding(g: LayerGroup, tp: number): { perGpu: number; replication: number } {
  if (tp === 1) return { perGpu: 1, replication: 1 };
  switch (g.kind) {
    case "full":
    case "sliding": {
      const headsPerGpu = Math.ceil(g.kvHeads / tp);
      return { perGpu: headsPerGpu / g.kvHeads, replication: (headsPerGpu * tp) / g.kvHeads };
    }
    case "mla":
      // Le latent MLA n'a pas de dimension "tête" : chaque rang TP garde tout le cache.
      return { perGpu: 1, replication: tp };
    case "linear":
      return { perGpu: 1 / tp, replication: 1 };
  }
}

/** Degré de TP compatible avec les têtes du modèle (contraintes vLLM). */
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

  // --- Poids (format effectif en mémoire)
  const w = weightBytes(m, input.memoryWeightFormat ?? weightFormat);
  /** Octets de params paramètres : au format choisi s'ils sont quantifiés, en 16 bits sinon. */
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
      // Les experts routés sont répartis ; tout le reste (attention, embeddings, couches denses,
      // experts partagés, routeur) est présent sur chaque rang, à sa précision réelle.
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
    // Couches entières par étage : on dimensionne le rang le plus chargé (ceil(L / pp) couches),
    // plus les embeddings d'entrée ou de sortie qu'il porte (copie sur les deux bords s'ils sont liés).
    const embSide = m.tiedEmbeddings ? embBytes : embBytes / 2;
    const body = w.bytes - embBytes;
    weightsPerGpu = ((body * layersPerStage) / L + embSide) / tp;
    if (m.tiedEmbeddings) replicatedBytes = embBytes;
  }

  // --- KV par requête
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
        detail = `${g.kvHeads} têtes KV × (${g.headDim}+${dv})`;
        if (g.kind === "sliding")
          detail += slidingWindowSaves(engine) ? ` · fenêtre ${windowLabel(g.window)}` : ` · fenêtre ${windowLabel(g.window)}, blocs non libérés`;
        break;
      }
      case "mla": {
        const perTok = (g.kvLoraRank + g.ropeHeadDim) * kvB + (g.extraBytesPerToken ?? 0);
        bytes = g.count * perTok * ctx;
        perTokenBytes += g.count * perTok;
        detail = `latent ${g.kvLoraRank}${g.ropeHeadDim ? ` + RoPE ${g.ropeHeadDim}` : " (NoPE)"}${g.extraBytesPerToken ? ` + ${g.extraBytesPerToken} o/tok (indexeur)` : ""}`;
        break;
      }
      case "linear": {
        bytes = g.count * (g.ssmElements * m.ssmStateBytes + g.convElements * m.convStateBytes);
        constantBytes += bytes;
        detail = `état ${(g.ssmElements / 1e6).toFixed(2)} M × ${m.ssmStateBytes} o${g.convElements ? ` + conv ${(g.convElements / 1e3).toFixed(0)} k × ${m.convStateBytes} o` : ""}`;
        break;
      }
    }
    const s = mode === "dp-ep" || n === 1 ? { perGpu: 1, replication: 1 } : tpSharding(g, tp);
    for (let k = 0; k < g.count; k++) perLayer.push({ cost: (bytes / g.count) * s.perGpu, group: gi });
    groups.push({ kind: g.kind, label: groupLabel(g), count: g.count, detail, bytes, perGpuBytes: 0, replication: s.replication });
  });
  // Rang le plus chargé : au plus ceil(L / pp) couches, en prenant les plus coûteuses (borne prudente,
  // l'ordre exact des couches n'étant pas connu).
  perLayer.sort((a, b) => b.cost - a.cost);
  for (const l of perLayer.slice(0, Math.min(layersPerStage, perLayer.length))) groups[l.group].perGpuBytes += l.cost;
  const perRequestBytes = groups.reduce((a, g) => a + g.bytes, 0);
  const perRequestPerGpuBytes = groups.reduce((a, g) => a + g.perGpuBytes, 0);
  const perRequestClusterBytes = mode === "dp-ep" || n === 1 ? perRequestBytes : groups.reduce((a, g) => a + g.bytes * g.replication, 0);

  // --- Mémoire
  const gpuBytes = (input.memoryGiBOverride ?? gpu.memoryGiB) * GIB;
  const reservePerGpuBytes = gpuBytes * (input.reservePct / 100);
  const usablePerGpuBytes = gpuBytes - reservePerGpuBytes;
  const freeForKvPerGpuBytes = Math.max(0, usablePerGpuBytes - weightsPerGpu);
  const fitsWeights = weightsPerGpu <= usablePerGpuBytes;

  const sessionsPerGpu = perRequestPerGpuBytes > 0 ? Math.floor(freeForKvPerGpuBytes / perRequestPerGpuBytes) : 0;
  const sessions = mode === "dp-ep" ? sessionsPerGpu * n : sessionsPerGpu;

  // --- Offload CPU
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

const LEVEL_FR: Record<SupportLevel, string> = {
  native: "natif",
  limited: "limité",
  unverified: "non vérifié",
  unsupported: "non supporté",
};
export const supportLevelLabel = (l: SupportLevel) => LEVEL_FR[l];

const gibText = (b: number) =>
  b >= GIB ? `${(b / GIB).toFixed(1).replace(".", ",")} Gio` : `${Math.round(b / 1024 ** 2)} Mio`;

/** Liste des limitations / avertissements pour une configuration. */
export function collectIssues(input: EstimateInput, est: Estimate, weightFormats: WeightFormat[]): Issue[] {
  const { model: m, gpu, engine, gpuCount: n, contextTokens: ctx, weightFormat: wf, kvFormat: kf, mode } = input;
  const { tp, pp } = est;
  const out: Issue[] = [];

  // --- Capacité
  if (!est.fitsWeights) {
    out.push({
      severity: "error",
      title: "Les poids ne tiennent pas en mémoire",
      detail: `${gibText(est.weights.perGpuBytes)} de poids par GPU pour ${gibText(est.memory.usablePerGpuBytes)} utilisables. Ajoutez des GPUs, quantifiez davantage ou réduisez la réserve.`,
    });
  } else if (est.sessions < 1) {
    out.push({
      severity: "error",
      title: "Pas assez de mémoire pour une seule requête à ce contexte",
      detail: `Il reste ${gibText(est.memory.freeForKvPerGpuBytes)} par GPU, et une requête en demande ${gibText(est.kv.perRequestPerGpuBytes)} par GPU.`,
    });
  }

  // --- Moteur
  const eg = engine.gpus[gpu.id];
  if (eg && eg.level !== "native")
    out.push({
      severity: eg.level === "unsupported" ? "error" : "warning",
      title: `${engine.name} sur ${gpu.name} : ${LEVEL_FR[eg.level]}`,
      detail: eg.note,
    });
  if (engine.modelFamilies && !engine.modelFamilies.includes(m.family))
    out.push({
      severity: "warning",
      title: `Famille ${m.family} non annoncée par ${engine.name}`,
      detail: `Familles annoncées : ${engine.modelFamilies.join(", ")}. Le modèle peut fonctionner, mais rien ne le garantit.`,
    });
  if (engine.confidence !== "high")
    out.push({ severity: "warning", title: `Profil ${engine.name} partiellement vérifié`, detail: engine.notes[0] });
  const fe = engine.features;
  if (fe.kvPaging === false)
    out.push({ severity: "info", title: `KV non paginé dans ${engine.name}`, detail: "Chaque requête réserve son contexte maximal : c'est déjà l'hypothèse du calcul (pire cas)." });
  else if (fe.kvPaging === "unverified")
    out.push({ severity: "info", title: `Pagination du KV non vérifiée dans ${engine.name}`, detail: "Sans effet sur ce calcul, fait au contexte maximal pour chaque session." });
  if (fe.kvPoolPreallocated === "unverified")
    out.push({ severity: "info", title: `Préallocation du pool KV non vérifiée dans ${engine.name}`, detail: "Si le pool est dimensionné autrement que par la réserve choisie, la capacité réelle diffère." });

  // --- Formats vs GPU et moteur
  const ws = weightSupport(gpu, engine, wf.id);
  if (ws.level !== "native")
    out.push({ severity: ws.level === "unsupported" ? "error" : "warning", title: `Poids ${wf.label} : ${LEVEL_FR[ws.level]}`, detail: ws.note });
  else if (ws.note) out.push({ severity: "info", title: `Poids ${wf.label}`, detail: ws.note });
  const memFmt = input.memoryWeightFormat;
  if (memFmt && memFmt.id !== wf.id)
    out.push({
      severity: "info",
      title: `Fichier ${wf.label} chargé en ${memFmt.label} en mémoire`,
      detail: `${engine.name} déquantifie les poids au chargement : ${gibText(est.weights.totalBytes)} en VRAM au lieu de ${gibText(weightBytes(m, wf).bytes)}.`,
    });
  else if (ws.level === "unverified" && wf.bitsPerParam < 16)
    out.push({
      severity: "info",
      title: `Format en mémoire non vérifié pour ${wf.label}`,
      detail: `Calcul fait en supposant que les poids restent en ${wf.label}. S'ils sont déquantifiés en BF16 au chargement : ${gibText(weightBytes(m, { ...wf, id: "bf16", bitsPerParam: 16 }).bytes)}.`,
    });

  const ks = kvSupport(gpu, engine, kf.id);
  if (ks.level !== "native")
    out.push({ severity: ks.level === "unsupported" ? "error" : "warning", title: `KV cache ${kf.label} : ${LEVEL_FR[ks.level]}`, detail: ks.note });
  else if (ks.note) out.push({ severity: "info", title: `KV cache ${kf.label}`, detail: ks.note });

  // --- Formats vs modèle
  const native = m.nativeWeightFormat;
  const wNote = m.weightFormatNotes[wf.id];
  if (wNote) out.push({ severity: wNote.level === "unsupported" ? "error" : wNote.level, title: `${m.name} en ${wf.label}`, detail: wNote.note });
  const nativeFmt = weightFormats.find((f) => f.id === native);
  const nativeFmtBits = nativeFmt ? effectiveBits(m, nativeFmt) : undefined;
  // Même largeur (ex. BF16 → FP16) : simple conversion, pas de quantification.
  if (wf.id !== native && nativeFmtBits !== wf.bitsPerParam) {
    if (nativeFmtBits !== undefined && wf.bitsPerParam > nativeFmtBits)
      out.push({
        severity: "info",
        title: `Poids publiés nativement en ${nativeFmt?.label ?? native}`,
        detail: `Charger en ${wf.label} augmente la mémoire sans gain de qualité.`,
      });
    else if (!m.officialWeightFormats.includes(wf.id))
      out.push({
        severity: "info",
        title: `Pas de checkpoint officiel en ${wf.label}`,
        detail: "Quantification communautaire ou maison : vérifiez la disponibilité et l'impact qualité.",
      });
  }
  const kNote = m.kvFormatNotes[kf.id];
  if (kNote) out.push({ severity: kNote.level === "unsupported" ? "error" : kNote.level, title: `KV ${kf.label} avec ${m.name}`, detail: kNote.note });

  // --- Contexte
  if (ctx > m.maxContext) {
    if (m.maxContextExtended && ctx <= m.maxContextExtended)
      out.push({
        severity: "warning",
        title: `Contexte au-delà du natif (${contextLabel(m.maxContext)})`,
        detail: `Possible jusqu'à ${contextLabel(m.maxContextExtended)} avec extension RoPE (YaRN). La qualité peut baisser sur les textes courts.`,
      });
    else
      out.push({
        severity: "error",
        title: `Contexte supérieur au maximum du modèle (${contextLabel(m.maxContextExtended ?? m.maxContext)})`,
        detail: "Le calcul reste indicatif : le modèle n'est pas prévu pour cette longueur.",
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
        title: `Attention glissante non supportée par ${engine.name}`,
        detail: `${cnt} couches de ${m.name} utilisent une fenêtre de ${windowLabel(win)} : sans support, les résultats du modèle sont faux.`,
      });
    else if (slidingWindowSaves(engine))
      out.push({ severity: "info", title: `${cnt} couches à fenêtre glissante (${windowLabel(win)})`, detail: `${engine.name} libère les blocs KV sortis de la fenêtre : KV plafonné à la fenêtre.` });
    else
      out.push({
        severity: "warning",
        title: `${cnt} couches à fenêtre glissante comptées au contexte complet`,
        detail: [
          f.slidingWindowAttention === "unverified" ? `Support de l'attention glissante non vérifié dans ${engine.name}.` : `Attention glissante supportée par ${engine.name}.`,
          f.slidingWindowFreesBlocks === false
            ? "Les blocs hors fenêtre ne sont pas libérés : aucune économie de VRAM."
            : "Libération des blocs hors fenêtre non vérifiée : aucune économie de VRAM appliquée.",
          engine.featureNotes.slidingWindowFreesBlocks ?? engine.featureNotes.slidingWindowAttention,
        ]
          .filter(Boolean)
          .join(" "),
      });
  }
  if (m.layers.some((g) => g.kind === "linear"))
    out.push({
      severity: "info",
      title: "Couches linéaires / SSM : état récurrent constant",
      detail: `${gibText(est.kv.constantBytes)} par requête quel que soit le contexte (état en ${m.ssmStateBytes * 8} bits), indépendamment du format KV choisi.`,
    });
  if (m.layers.some((g) => (g.kind === "mla" || g.kind === "full") && (g.extraBytesPerToken ?? 0) > 0))
    out.push({ severity: "info", title: "Index d'attention parcimonieuse (DSA) inclus", detail: "Clés de l'indexeur stockées en FP8 + échelle, en plus du latent." });

  // --- Interconnexion
  const linkName = (k: number) => (gpu.interconnect.nvlink && k <= gpu.nvlinkDomain ? "NVLink" : k <= gpu.maxGpusPerNode ? gpu.interconnect.label : "le réseau inter-nœuds");
  if (n > 1) {
    if (mode === "tp-pp" && tp > 1 && !gpu.interconnect.nvlink)
      out.push({
        severity: "warning",
        title: `TP=${tp} sans NVLink sur ${gpu.name}`,
        detail: `Le TP fait deux all-reduce par couche via ${gpu.interconnect.label} : débit et latence nettement dégradés. Préférez PP (TP=1), ou plusieurs instances indépendantes.`,
      });
    if (mode === "tp-pp" && tp > 1 && gpu.interconnect.nvlink && tp > gpu.nvlinkDomain)
      out.push({
        severity: "warning",
        title: `TP=${tp} dépasse le domaine NVLink (${gpu.nvlinkDomain})`,
        detail: "Les all-reduce passent par le réseau. Gardez le TP dans le domaine NVLink et utilisez PP au-delà.",
      });
    if (mode === "dp-ep" && n > Math.max(gpu.nvlinkDomain, 1) && n > gpu.maxGpusPerNode)
      out.push({ severity: "info", title: "Expert parallel inter-nœuds", detail: "Les all-to-all MoE traversent le réseau : prévoyez un réseau RDMA rapide (InfiniBand / RoCE)." });
    if (n > gpu.maxGpusPerNode)
      out.push({
        severity: n > gpu.nvlinkDomain ? "warning" : "info",
        title: `${n} GPUs : ${Math.ceil(n / gpu.maxGpusPerNode)} nœuds de ${gpu.maxGpusPerNode} GPUs`,
        detail: n <= gpu.nvlinkDomain ? `Tous dans le même domaine NVLink (${gpu.nvlinkDomain} GPUs).` : `Liaison entre nœuds par ${linkName(n)}.`,
      });
  }

  // --- Parallélisme
  if (n > 1 && mode === "tp-pp") {
    const cap = (k: "tensorParallel" | "pipelineParallel", used: boolean, label: string) => {
      const v = engine.features[k];
      if (!used || v === true) return;
      out.push({
        severity: v === false ? "error" : "warning",
        title: `${label} ${v === false ? "non supporté" : "non vérifié"} dans ${engine.name}`,
        detail: engine.featureNotes[k],
      });
    };
    cap("tensorParallel", tp > 1, "Tensor parallel");
    cap("pipelineParallel", pp > 1, "Pipeline parallel");
    if (tp > 1 && m.attentionHeads % tp !== 0)
      out.push({
        severity: "error",
        title: `TP=${tp} impossible : ${m.attentionHeads} têtes d'attention non divisibles par ${tp}`,
        detail: `Choisissez un degré de TP diviseur de ${m.attentionHeads} et complétez avec du PP.`,
      });
    for (const g of m.layers) {
      if (g.kind !== "full" && g.kind !== "sliding") continue;
      if (tp > 1 && g.kvHeads % tp !== 0 && tp % g.kvHeads !== 0)
        out.push({ severity: "error", title: `TP=${tp} incompatible avec ${g.kvHeads} têtes KV`, detail: "Le nombre de têtes KV doit être multiple ou diviseur du degré de TP." });
      else if (tp > g.kvHeads) {
        out.push({
          severity: "warning",
          title: `KV répliqué ×${tp / g.kvHeads} (${g.kvHeads} têtes KV pour TP=${tp})`,
          detail: "Chaque tête KV est dupliquée sur plusieurs GPUs. Réduisez le TP et ajoutez du PP, ou passez en DP-attention.",
        });
        break;
      }
    }
    if (tp > 1 && m.layers.some((g) => g.kind === "mla"))
      out.push({
        severity: "warning",
        title: `Cache MLA répliqué sur les ${tp} GPUs du TP`,
        detail: `Chaque rang TP stocke le latent complet. En DP-attention${engine.flags.dpAttention ? ` (${engine.flags.dpAttention})` : ""}, chaque GPU garde ses propres requêtes : capacité ≈ ×${tp}.`,
      });
    if (pp > 1) {
      const L = decoderLayers(m);
      if (pp > L)
        out.push({ severity: "error", title: `PP=${pp} supérieur au nombre de couches (${L})`, detail: "Chaque étage doit porter au moins une couche entière." });
      else if (L % pp !== 0)
        out.push({
          severity: "info",
          title: `PP=${pp} : jusqu'à ${est.layersPerStage} couches sur l'étage le plus chargé`,
          detail: `${L} couches non divisibles par ${pp} : poids et KV calculés pour le rang le plus chargé (couches entières, les plus coûteuses en KV).`,
        });
      if (m.tiedEmbeddings) out.push({ severity: "info", title: "Embeddings liés en PP", detail: `Copie sur le premier et le dernier étage (+${gibText(est.weights.replicatedBytes)}).` });
      if (n % tp !== 0) out.push({ severity: "error", title: `${n} GPUs non divisibles par TP=${tp}`, detail: "TP × PP doit être égal au nombre de GPUs." });
    }
  }
  if (mode === "dp-ep") {
    if (engine.features.dpAttention === false)
      out.push({ severity: "error", title: `DP-attention indisponible dans ${engine.name}`, detail: "Choisissez TP/PP." });
    else if (engine.features.dpAttention === "unverified")
      out.push({ severity: "warning", title: `DP-attention non vérifiée dans ${engine.name}`, detail: engine.featureNotes.dpAttention ?? "Aucune source ne confirme ce mode : résultat théorique." });
    if (!isMoE(m))
      out.push({ severity: "warning", title: "DP-attention conçu pour les MoE", detail: "Sur un modèle dense, cela revient à N répliques complètes." });
    else if (n > 1 && est.weights.dpUnvalidated)
      out.push({
        severity: "error",
        title: `DP-attention : paramètres des experts inconnus pour ${m.name}`,
        detail: "Impossible de séparer experts répartis et tensors répliqués : capacité non validée (calcul avec le modèle complet par GPU). Renseignez expertParamsB dans le catalogue.",
      });
    else if (n > 1)
      out.push({
        severity: "info",
        title: "DP-attention + expert parallel",
        detail: `Seuls les experts routés sont répartis. Attention, embeddings, couches denses et experts partagés (${gibText(est.weights.nonExpertBytes)}, à leur précision réelle) sont présents sur chaque GPU : +${gibText(est.weights.replicatedBytes)} au total. Chaque requête vit sur un seul GPU.`,
      });
  }

  // --- Offload
  if ((input.offloadGiBPerGpu ?? 0) > 0) {
    if (engine.features.kvOffload === false)
      out.push({ severity: "warning", title: `Offload KV non disponible dans ${engine.name}`, detail: "La mémoire CPU saisie n'est pas comptée." });
    else {
      if (engine.features.kvOffload === "unverified")
        out.push({ severity: "warning", title: `Offload KV non vérifié dans ${engine.name}`, detail: engine.featureNotes.kvOffload ?? "Sessions en réserve théoriques." });
      out.push({
        severity: "info",
        title: `Offload CPU : ${est.offload.parkedSessions} sessions en réserve`,
        detail: `Le KV déchargé n'est pas lu pendant le décodage : il sert aux sessions en attente, préemptées ou à réutiliser (prefix cache). Rechargement d'une session ≈ ${Math.round(est.offload.reloadMs)} ms via ${gpu.hostLinkLabel}.`,
      });
    }
  }

  if (m.confidence !== "high")
    out.push({
      severity: m.confidence === "low" ? "warning" : "info",
      title: `Architecture de ${m.name} à vérifier (fiabilité ${m.confidence === "low" ? "faible" : "moyenne"})`,
      detail: `Comparez avec le config.json${m.source ? ` de ${m.source}` : ""} (pnpm run import-model) et corrigez le fichier de configuration si besoin.`,
    });

  const order: Record<Severity, number> = { error: 0, warning: 1, info: 2 };
  const seen = new Set<string>();
  return out
    .filter((i) => !seen.has(i.title) && seen.add(i.title))
    .sort((a, b) => order[a.severity] - order[b.severity]);
}
