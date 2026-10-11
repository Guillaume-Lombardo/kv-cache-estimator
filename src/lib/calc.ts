import type {
  GpuSpec,
  KvFormat,
  LayerGroup,
  ModelSpec,
  Support,
  WeightFormat,
} from "./config-schema";

export const GIB = 1024 ** 3;

/** 8k → 1M par doublement (puissances de 2). */
export const CONTEXT_STEPS = [8, 16, 32, 64, 128, 256, 512, 1024].map((k) => k * 1024);

export function contextLabel(tokens: number): string {
  return tokens >= 1024 * 1024 ? `${tokens / (1024 * 1024)}M` : `${Math.round(tokens / 1024)}k`;
}

export type Parallelism = "tp" | "pp" | "dp-attention";

export const PARALLELISM_LABELS: Record<Parallelism, string> = {
  tp: "Tensor parallel (TP)",
  pp: "Pipeline parallel (PP)",
  "dp-attention": "DP-attention + EP",
};

export interface EstimateInput {
  model: ModelSpec;
  gpu: GpuSpec;
  gpuCount: number;
  contextTokens: number;
  weightFormat: WeightFormat;
  kvFormat: KvFormat;
  /** Part de la mémoire de chaque GPU laissée aux activations, graphes CUDA, fragmentation. */
  reservePct: number;
  parallelism: Parallelism;
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
  /** Tokens effectivement stockés par couche (fenêtre glissante, état constant). */
  storedTokens: number | null;
  /** Octets stockés sur un GPU pour une requête, selon le parallélisme. */
  perGpuBytes: number;
  /** Facteur de réplication du cache entre GPUs (1 = pas de duplication). */
  replication: number;
}

export interface Estimate {
  weights: {
    totalBytes: number;
    perGpuBytes: number;
    quantizedParams: number;
    unquantizedParams: number;
    replicatedAttentionBytes: number;
  };
  kv: {
    /** KV d'une requête, vue logique. */
    perRequestBytes: number;
    /** Mémoire réellement consommée par une requête sur l'ensemble des GPUs. */
    perRequestClusterBytes: number;
    /** Mémoire consommée par une requête sur le GPU le plus chargé. */
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
    freeForKvTotalBytes: number;
  };
  /** Poids + une requête (vue logique). */
  modelPlusOneKvBytes: number;
  fitsOneRequest: boolean;
  fitsWeights: boolean;
  sessions: number;
}

export function supportOf(map: Record<string, Support>, id: string): Support {
  return map[id] ?? { level: "unsupported" };
}

export function embeddingParams(m: ModelSpec): number {
  return m.vocabSize * m.hiddenSize * (m.tiedEmbeddings ? 1 : 2);
}

/** Les embeddings / lm_head (et l'encodeur vision) restent en 16 bits quand on quantifie. */
export function weightBytes(m: ModelSpec, f: WeightFormat) {
  const total = m.paramsB * 1e9;
  if (f.bitsPerParam >= 16) {
    return { bytes: (total * f.bitsPerParam) / 8, quantized: total, unquantized: 0 };
  }
  const keep = Math.min(total, embeddingParams(m) + m.extraUnquantizedParamsB * 1e9);
  return {
    bytes: ((total - keep) * f.bitsPerParam) / 8 + keep * 2,
    quantized: total - keep,
    unquantized: keep,
  };
}

/** Paramètres d'attention (Q, K, V, O) : valeur du fichier ou estimation à partir des couches GQA. */
export function attentionParams(m: ModelSpec): number {
  if (m.attentionParamsB !== undefined) return m.attentionParamsB * 1e9;
  let p = 0;
  for (const g of m.layers) {
    if (g.kind !== "full" && g.kind !== "sliding") continue;
    const dv = g.headDimV ?? g.headDim;
    const q = m.hiddenSize * m.attentionHeads * g.headDim;
    const o = m.attentionHeads * dv * m.hiddenSize;
    const kv = m.hiddenSize * g.kvHeads * (g.headDim + dv);
    p += g.count * (q + o + kv);
  }
  return p;
}

export function isMoE(m: ModelSpec): boolean {
  return m.activeParamsB !== undefined && m.activeParamsB < m.paramsB * 0.8;
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

function slidingLabel(window: number): string {
  return window < 1024 ? `${window} tok.` : contextLabel(window);
}

/** Partage du cache d'un groupe de couches entre GPUs. */
function sharding(g: LayerGroup, n: number, par: Parallelism): { perGpu: number; replication: number } {
  if (n === 1 || par === "dp-attention") return { perGpu: 1, replication: 1 };
  if (par === "pp") return { perGpu: 1 / n, replication: 1 };
  switch (g.kind) {
    case "full":
    case "sliding": {
      const headsPerGpu = Math.ceil(g.kvHeads / n);
      return { perGpu: headsPerGpu / g.kvHeads, replication: (headsPerGpu * n) / g.kvHeads };
    }
    case "mla":
      // Le latent MLA n'a pas de dimension "tête" : chaque rang TP garde tout le cache.
      return { perGpu: 1, replication: n };
    case "linear":
      return { perGpu: 1 / n, replication: 1 };
  }
}

export function estimate(input: EstimateInput): Estimate {
  const { model: m, gpu, gpuCount: n, contextTokens: ctx, weightFormat, kvFormat, parallelism } = input;
  const kvB = kvFormat.bytesPerElement;

  // --- Poids
  const w = weightBytes(m, weightFormat);
  let weightsPerGpu = w.bytes / n;
  let replicatedAttentionBytes = 0;
  if (parallelism === "dp-attention" && n > 1) {
    const attnBytes = Math.min(w.bytes, (attentionParams(m) * weightFormat.bitsPerParam) / 8);
    weightsPerGpu = attnBytes + (w.bytes - attnBytes) / n;
    replicatedAttentionBytes = attnBytes * (n - 1);
  }

  // --- KV par requête
  const groups: GroupBreakdown[] = [];
  let perTokenBytes = 0;
  let constantBytes = 0;
  for (const g of m.layers) {
    let bytes: number;
    let storedTokens: number | null;
    let detail: string;
    switch (g.kind) {
      case "full":
      case "sliding": {
        const dv = g.headDimV ?? g.headDim;
        const tokens = g.kind === "sliding" ? Math.min(ctx, g.window) : ctx;
        const perTok = g.kvHeads * (g.headDim + dv) * kvB + (g.extraBytesPerToken ?? 0);
        bytes = g.count * perTok * tokens;
        storedTokens = tokens;
        if (g.kind === "full") perTokenBytes += g.count * perTok;
        detail = `${g.kvHeads} têtes KV × (${g.headDim}+${dv})${g.kind === "sliding" ? ` · fenêtre ${slidingLabel(g.window)}` : ""}`;
        break;
      }
      case "mla": {
        const perTok = (g.kvLoraRank + g.ropeHeadDim) * kvB + (g.extraBytesPerToken ?? 0);
        bytes = g.count * perTok * ctx;
        storedTokens = ctx;
        perTokenBytes += g.count * perTok;
        detail = `latent ${g.kvLoraRank} + RoPE ${g.ropeHeadDim}${g.extraBytesPerToken ? ` + ${g.extraBytesPerToken} o/tok (indexeur)` : ""}`;
        break;
      }
      case "linear": {
        bytes = g.count * g.stateElements * m.stateBytesPerElement;
        storedTokens = null;
        constantBytes += bytes;
        detail = `${(g.stateElements / 1e6).toFixed(2)} M éléments/couche · ${m.stateBytesPerElement} o/élément`;
        break;
      }
    }
    const s = sharding(g, n, parallelism);
    groups.push({
      kind: g.kind,
      label: groupLabel(g),
      count: g.count,
      detail,
      bytes,
      storedTokens,
      perGpuBytes: bytes * s.perGpu,
      replication: s.replication,
    });
  }
  const perRequestBytes = groups.reduce((a, g) => a + g.bytes, 0);
  const perRequestPerGpuBytes = groups.reduce((a, g) => a + g.perGpuBytes, 0);
  const perRequestClusterBytes =
    parallelism === "dp-attention" || n === 1 ? perRequestBytes : perRequestPerGpuBytes * n;

  // --- Mémoire
  const gpuBytes = gpu.memoryGiB * GIB;
  const reservePerGpuBytes = gpuBytes * (input.reservePct / 100);
  const usablePerGpuBytes = gpuBytes - reservePerGpuBytes;
  const freeForKvPerGpuBytes = Math.max(0, usablePerGpuBytes - weightsPerGpu);
  const fitsWeights = weightsPerGpu <= usablePerGpuBytes;

  const sessionsPerGpu = perRequestPerGpuBytes > 0 ? Math.floor(freeForKvPerGpuBytes / perRequestPerGpuBytes) : 0;
  const sessions = parallelism === "dp-attention" ? sessionsPerGpu * n : sessionsPerGpu;

  return {
    weights: {
      totalBytes: w.bytes,
      perGpuBytes: weightsPerGpu,
      quantizedParams: w.quantized,
      unquantizedParams: w.unquantized,
      replicatedAttentionBytes,
    },
    kv: { perRequestBytes, perRequestClusterBytes, perRequestPerGpuBytes, perTokenBytes, constantBytes, groups },
    memory: {
      gpuBytes,
      reservePerGpuBytes,
      usablePerGpuBytes,
      freeForKvPerGpuBytes,
      freeForKvTotalBytes: freeForKvPerGpuBytes * n,
    },
    modelPlusOneKvBytes: w.bytes + perRequestBytes,
    fitsWeights,
    fitsOneRequest: fitsWeights && sessions >= 1,
    sessions,
  };
}

const LEVEL_FR: Record<Support["level"], string> = {
  native: "natif",
  limited: "limité",
  unsupported: "non supporté",
};
export const supportLevelLabel = (l: Support["level"]) => LEVEL_FR[l];

/** Liste des limitations / avertissements pour une configuration. */
export function collectIssues(input: EstimateInput, est: Estimate, weightFormats: WeightFormat[]): Issue[] {
  const { model: m, gpu, gpuCount: n, contextTokens: ctx, weightFormat: wf, kvFormat: kf, parallelism } = input;
  const out: Issue[] = [];
  const gib = (b: number) =>
    b >= GIB ? `${(b / GIB).toFixed(1).replace(".", ",")} Gio` : `${Math.round(b / 1024 ** 2)} Mio`;

  // --- Capacité
  if (!est.fitsWeights) {
    out.push({
      severity: "error",
      title: "Les poids ne tiennent pas en mémoire",
      detail: `${gib(est.weights.perGpuBytes)} de poids par GPU pour ${gib(est.memory.usablePerGpuBytes)} utilisables. Ajoutez des GPUs, quantifiez davantage ou réduisez la réserve.`,
    });
  } else if (est.sessions < 1) {
    out.push({
      severity: "error",
      title: "Pas assez de mémoire pour une seule requête à ce contexte",
      detail: `Il reste ${gib(est.memory.freeForKvPerGpuBytes)} par GPU, et une requête en demande ${gib(est.kv.perRequestPerGpuBytes)} par GPU.`,
    });
  }

  // --- Formats vs GPU
  const ws = supportOf(gpu.weightFormats, wf.id);
  if (ws.level !== "native")
    out.push({
      severity: ws.level === "unsupported" ? "error" : "warning",
      title: `Poids ${wf.label} : ${LEVEL_FR[ws.level]} sur ${gpu.name}`,
      detail: ws.note,
    });
  else if (ws.note) out.push({ severity: "info", title: `Poids ${wf.label} sur ${gpu.name}`, detail: ws.note });

  const ks = supportOf(gpu.kvCacheFormats, kf.id);
  if (ks.level !== "native")
    out.push({
      severity: ks.level === "unsupported" ? "error" : "warning",
      title: `KV cache ${kf.label} : ${LEVEL_FR[ks.level]} sur ${gpu.name}`,
      detail: ks.note,
    });
  else if (ks.note) out.push({ severity: "info", title: `KV cache ${kf.label} sur ${gpu.name}`, detail: ks.note });

  // --- Formats vs modèle
  const native = input.model.nativeWeightFormat;
  const wNote = m.weightFormatNotes[wf.id];
  if (wNote)
    out.push({
      severity: wNote.level === "unsupported" ? "error" : wNote.level,
      title: `${m.name} en ${wf.label}`,
      detail: wNote.note,
    });
  const nativeFmtBits = weightFormats.find((f) => f.id === native)?.bitsPerParam;
  // Même largeur (ex. BF16 → FP16) : simple conversion, pas de quantification.
  if (wf.id !== native && nativeFmtBits !== wf.bitsPerParam) {
    if (nativeFmtBits !== undefined && wf.bitsPerParam > nativeFmtBits)
      out.push({
        severity: "info",
        title: `Poids publiés nativement en ${native.toUpperCase()}`,
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
  if (kNote)
    out.push({
      severity: kNote.level === "unsupported" ? "error" : kNote.level,
      title: `KV ${kf.label} avec ${m.name}`,
      detail: kNote.note,
    });

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
  const sliding = m.layers.filter((g) => g.kind === "sliding");
  if (sliding.length) {
    const cnt = sliding.reduce((a, g) => a + g.count, 0);
    const win = Math.max(...sliding.map((g) => (g.kind === "sliding" ? g.window : 0)));
    out.push({
      severity: "info",
      title: `${cnt} couches à fenêtre glissante (${slidingLabel(win)})`,
      detail:
        "Leur KV est plafonné à la fenêtre. Le moteur doit le gérer (gestionnaire hybride de vLLM ≥ 0.10, SGLang, TensorRT-LLM) ; sinon il alloue le contexte complet sur toutes les couches.",
    });
  }
  if (m.layers.some((g) => g.kind === "linear"))
    out.push({
      severity: "info",
      title: "Couches linéaires / SSM : état récurrent constant",
      detail: `${gib(est.kv.constantBytes)} par requête quel que soit le contexte, stockés en ${m.stateBytesPerElement * 8} bits indépendamment du format KV choisi.`,
    });

  // --- Interconnexion
  if (n > 1) {
    if (!gpu.interconnect.nvlink)
      out.push({
        severity: "warning",
        title: `Pas de NVLink sur ${gpu.name}`,
        detail: `Communication inter-GPU via ${gpu.interconnect.label}. ${parallelism === "tp" ? "Le TP fait un all-reduce à chaque couche : débit et latence nettement dégradés. Préférez PP, ou plusieurs instances indépendantes." : "Le PP ne transfère que les activations entre étages : c'est le meilleur choix sans NVLink."}`,
      });
    else if (n > gpu.nvlinkDomain)
      out.push({
        severity: "warning",
        title: `${n} GPUs : au-delà du domaine NVLink (${gpu.nvlinkDomain})`,
        detail: "Une partie des échanges passe par le réseau (InfiniBand/Ethernet). Réservez le TP au domaine NVLink et utilisez PP au-delà.",
      });
    if (n > gpu.maxGpusPerNode)
      out.push({
        severity: "warning",
        title: `${n} GPUs : plus d'un nœud (${gpu.maxGpusPerNode} GPU/nœud)`,
        detail: "Déploiement multi-nœuds nécessaire.",
      });
  }

  // --- Parallélisme
  if (n > 1 && parallelism === "tp") {
    if (m.attentionHeads % n !== 0)
      out.push({
        severity: "error",
        title: `TP=${n} impossible : ${m.attentionHeads} têtes d'attention non divisibles par ${n}`,
        detail: `Choisissez un nombre de GPUs diviseur de ${m.attentionHeads}, ou passez en PP (ou TP×PP).`,
      });
    for (const g of m.layers) {
      if (g.kind !== "full" && g.kind !== "sliding") continue;
      if (g.kvHeads % n !== 0 && n % g.kvHeads !== 0)
        out.push({
          severity: "error",
          title: `TP=${n} incompatible avec ${g.kvHeads} têtes KV`,
          detail: "Le nombre de têtes KV doit être multiple ou diviseur du degré de TP.",
        });
      else if (n > g.kvHeads) {
        out.push({
          severity: "warning",
          title: `KV répliqué ×${n / g.kvHeads} (${g.kvHeads} têtes KV pour TP=${n})`,
          detail: "Chaque tête KV est dupliquée sur plusieurs GPUs : la capacité en sessions n'augmente plus avec le nombre de GPUs. Envisagez TP plus petit + PP, ou DP-attention.",
        });
        break;
      }
    }
    if (m.layers.some((g) => g.kind === "mla"))
      out.push({
        severity: "warning",
        title: `Cache MLA répliqué sur les ${n} GPUs en TP`,
        detail:
          "Chaque rang TP stocke le latent complet. En DP-attention (SGLang --enable-dp-attention, vLLM --data-parallel-size N --enable-expert-parallel), chaque GPU garde ses propres requêtes : capacité ≈ ×N.",
      });
  }
  if (n > 1 && parallelism === "pp") {
    const layers = m.layers.reduce((a, g) => a + g.count, 0);
    if (layers % n !== 0)
      out.push({
        severity: "info",
        title: `PP=${n} : ${layers} couches non divisibles par ${n}`,
        detail: "Répartition inégale : le GPU le plus chargé limite la capacité (calcul ici sur une moyenne).",
      });
    out.push({
      severity: "info",
      title: "Pipeline parallel",
      detail: "Pas de duplication du KV, mais la latence par token n'est pas réduite et il faut plusieurs requêtes en vol pour occuper tous les étages.",
    });
  }
  if (parallelism === "dp-attention") {
    if (!isMoE(m))
      out.push({
        severity: "warning",
        title: "DP-attention conçu pour les MoE",
        detail: "Sur un modèle dense, cela revient à N répliques complètes : le calcul suppose les experts partagés, ce qui n'a pas de sens ici.",
      });
    else if (n > 1)
      out.push({
        severity: "info",
        title: "DP-attention + expert parallel",
        detail: `Poids d'attention répliqués sur chaque GPU (+${gib(est.weights.replicatedAttentionBytes)} au total), experts répartis. Chaque requête vit sur un seul GPU.${m.attentionParamsB === undefined && m.layers.some((g) => g.kind === "mla" || g.kind === "linear") ? " Paramètres d'attention estimés partiellement (couches MLA/linéaires non comptées)." : ""}`,
      });
  }

  // --- Spécificités GPU
  if (n > 1 && gpu.id.startsWith("gh200"))
    out.push({
      severity: "info",
      title: "GH200 : 1 GPU par superchip",
      detail: "Le multi-GPU dépend de la plateforme (NVL2, NVL32, nœuds quad-GH200). Vérifiez la topologie NVLink.",
    });

  if (m.confidence !== "high")
    out.push({
      severity: m.confidence === "low" ? "warning" : "info",
      title: `Architecture de ${m.name} à vérifier (fiabilité ${m.confidence === "low" ? "faible" : "moyenne"})`,
      detail: `Comparez avec le config.json${m.source ? ` de ${m.source}` : ""} et corrigez le fichier de configuration si besoin.`,
    });

  const order: Record<Severity, number> = { error: 0, warning: 1, info: 2 };
  const seen = new Set<string>();
  return out
    .filter((i) => !seen.has(i.title) && seen.add(i.title))
    .sort((a, b) => order[a.severity] - order[b.severity]);
}
