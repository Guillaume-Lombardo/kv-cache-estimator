import { z } from "zod";

/** Niveau de support d'un format sur un GPU ou dans un moteur. */
export const SUPPORT_LEVELS = ["native", "limited", "unsupported"] as const;
export type SupportLevel = (typeof SUPPORT_LEVELS)[number];

const supportSchema = z
  .union([
    z.enum(SUPPORT_LEVELS),
    z.strictObject({ level: z.enum(SUPPORT_LEVELS), note: z.string().optional() }),
  ])
  .transform((v) => (typeof v === "string" ? { level: v } : v));
export type Support = z.output<typeof supportSchema>;

const weightFormatSchema = z.strictObject({
  id: z.string(),
  label: z.string(),
  /** Bits effectifs par paramètre, échelles de quantification incluses. */
  bitsPerParam: z.number().positive(),
  /** Débit de calcul utilisé pour l'estimation de performance. */
  compute: z.enum(["bf16", "fp8", "fp4"]),
  description: z.string().optional(),
});

const kvFormatSchema = z.strictObject({
  id: z.string(),
  label: z.string(),
  /** Octets effectifs par élément stocké, échelles incluses. */
  bytesPerElement: z.number().positive(),
  description: z.string().optional(),
});

const gpuSchema = z.strictObject({
  id: z.string(),
  name: z.string(),
  architecture: z.string(),
  computeCapability: z.string(),
  /** Mémoire visible par le driver, en Gio. */
  memoryGiB: z.number().positive(),
  memoryLabel: z.string(),
  /** Bande passante mémoire, To/s. */
  bandwidthTBs: z.number().positive(),
  /** Débit tensor core dense (sans sparsité), TFLOPS. */
  tflops: z.strictObject({
    bf16: z.number().positive(),
    fp8: z.number().positive().optional(),
    fp4: z.number().positive().optional(),
  }),
  interconnect: z.strictObject({
    label: z.string(),
    nvlink: z.boolean(),
    /** Bande passante GPU↔GPU par GPU, Go/s. */
    bandwidthGBs: z.number().positive(),
  }),
  /** GPUs dans un même serveur. */
  maxGpusPerNode: z.number().int().positive(),
  /** GPUs joignables en NVLink (1 = pas de NVLink). */
  nvlinkDomain: z.number().int().positive(),
  /** Mémoire CPU attachée (ex. LPDDR5X de Grace) par GPU, Gio. 0 = à saisir. */
  hostMemoryGiBPerGpu: z.number().nonnegative().default(0),
  /** Lien CPU↔GPU par GPU, Go/s (PCIe ou NVLink-C2C). */
  hostLinkGBs: z.number().positive(),
  hostLinkLabel: z.string(),
  weightFormats: z.record(z.string(), supportSchema),
  kvCacheFormats: z.record(z.string(), supportSchema),
  notes: z.array(z.string()).default([]),
});

const engineSchema = z.strictObject({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  confidence: z.enum(["high", "medium", "low"]).default("high"),
  features: z.strictObject({
    /** Les couches à fenêtre glissante n'allouent que la fenêtre. */
    slidingWindowOptimized: z.boolean(),
    /** DP-attention + expert parallel disponible. */
    dpAttention: z.boolean(),
    /** Déchargement du KV vers la mémoire CPU. */
    kvOffload: z.boolean(),
  }),
  /** Absent = non supporté par le moteur. */
  weightFormats: z.record(z.string(), supportSchema),
  kvCacheFormats: z.record(z.string(), supportSchema),
  /** Restrictions par GPU (absent = pas de restriction). */
  gpus: z.record(z.string(), supportSchema).default({}),
  notes: z.array(z.string()).default([]),
  /** Options de ligne de commande indicatives. */
  flags: z
    .strictObject({
      memoryUtilization: z.string().optional(),
      kvCacheDtype: z.string().optional(),
      tensorParallel: z.string().optional(),
      pipelineParallel: z.string().optional(),
      dpAttention: z.string().optional(),
      kvOffload: z.string().optional(),
    })
    .default({}),
});

const gqaFields = {
  count: z.number().int().positive(),
  kvHeads: z.number().int().positive(),
  headDim: z.number().int().positive(),
  /** Dimension des têtes V si différente de K. */
  headDimV: z.number().int().positive().optional(),
  /** Octets supplémentaires par token et par couche (indexeur, etc.), indépendants du format KV. */
  extraBytesPerToken: z.number().nonnegative().optional(),
};

const layerGroupSchema = z.discriminatedUnion("kind", [
  z.strictObject({ kind: z.literal("full"), ...gqaFields }),
  z.strictObject({
    kind: z.literal("sliding"),
    ...gqaFields,
    window: z.number().int().positive(),
  }),
  z.strictObject({
    kind: z.literal("mla"),
    count: z.number().int().positive(),
    kvLoraRank: z.number().int().positive(),
    ropeHeadDim: z.number().int().nonnegative(),
    extraBytesPerToken: z.number().nonnegative().optional(),
  }),
  z.strictObject({
    kind: z.literal("linear"),
    count: z.number().int().positive(),
    /** Éléments de l'état récurrent (SSM / DeltaNet / KDA) par couche et par requête. */
    ssmElements: z.number().int().positive(),
    /** Éléments de l'état de convolution courte par couche et par requête. */
    convElements: z.number().int().nonnegative().default(0),
    label: z.string().optional(),
  }),
]);
export type LayerGroup = z.output<typeof layerGroupSchema>;

const formatNoteSchema = z.strictObject({
  level: z.enum(["unsupported", "warning", "info"]),
  note: z.string(),
});

export const modelSchema = z.strictObject({
  id: z.string(),
  name: z.string(),
  family: z.string(),
  /** Dépôt Hugging Face de référence. */
  source: z.string().optional(),
  /** Révision du config.json utilisé. */
  revision: z.string().optional(),
  paramsB: z.number().positive(),
  activeParamsB: z.number().positive().optional(),
  hiddenSize: z.number().int().positive(),
  vocabSize: z.number().int().positive(),
  tiedEmbeddings: z.boolean().default(false),
  /** Paramètres conservés en 16 bits en plus des embeddings (encodeur vision, attention...). */
  extraUnquantizedParamsB: z.number().nonnegative().default(0),
  attentionHeads: z.number().int().positive(),
  /** Paramètres d'attention (répliqués en DP-attention). Calculé si absent. */
  attentionParamsB: z.number().nonnegative().optional(),
  layers: z.array(layerGroupSchema).min(1),
  maxContext: z.number().int().positive(),
  /** Contexte atteignable par extension RoPE (YaRN...). */
  maxContextExtended: z.number().int().positive().optional(),
  nativeWeightFormat: z.string(),
  /** Bits effectifs réels du checkpoint natif s'ils diffèrent du format générique (ex. INT4 g32 = 4,5). */
  nativeBitsPerParam: z.number().positive().optional(),
  officialWeightFormats: z.array(z.string()).default([]),
  /** Octets par élément de l'état récurrent (4 si mamba_ssm_cache_dtype = float32). */
  ssmStateBytes: z.number().positive().default(2),
  /** Octets par élément de l'état de convolution. */
  convStateBytes: z.number().positive().default(2),
  weightFormatNotes: z.record(z.string(), formatNoteSchema).default({}),
  kvFormatNotes: z.record(z.string(), formatNoteSchema).default({}),
  tags: z.array(z.string()).default([]),
  notes: z.array(z.string()).default([]),
  /** Fiabilité des paramètres d'architecture saisis. */
  confidence: z.enum(["high", "medium", "low"]).default("high"),
});

export const configSchema = z
  .strictObject({
    defaults: z
      .strictObject({
        model: z.string().optional(),
        gpu: z.string().optional(),
        engine: z.string().optional(),
        gpuCount: z.number().int().min(1).optional(),
        contextTokens: z.number().int().positive().optional(),
        kvCacheFormat: z.string().optional(),
        reservePct: z.number().min(0).max(50).optional(),
      })
      .default({}),
    performance: z
      .strictObject({
        /** Part de la bande passante mémoire réellement atteinte en décodage. */
        memoryEfficiency: z.number().min(0.05).max(1).default(0.7),
        /** Part du débit tensor core atteinte (MFU). */
        computeEfficiency: z.number().min(0.05).max(1).default(0.45),
        /** Latence d'un all-reduce / all-to-all de décodage, µs. */
        collectiveLatencyUs: z.strictObject({ nvlink: z.number().positive(), pcie: z.number().positive(), network: z.number().positive() }).default({ nvlink: 8, pcie: 30, network: 40 }),
      })
      .default({ memoryEfficiency: 0.7, computeEfficiency: 0.45, collectiveLatencyUs: { nvlink: 8, pcie: 30, network: 40 } }),
    /** Nombres de GPUs proposés au-delà de 8. */
    largeGpuCounts: z.array(z.number().int().min(9)).default([16, 32, 64, 72]),
    weightFormats: z.array(weightFormatSchema).min(1),
    kvCacheFormats: z.array(kvFormatSchema).min(1),
    engines: z.array(engineSchema).min(1),
    gpus: z.array(gpuSchema).min(1),
    models: z.array(modelSchema).min(1),
  })
  .superRefine((cfg, ctx) => {
    const wf = new Set(cfg.weightFormats.map((f) => f.id));
    const kf = new Set(cfg.kvCacheFormats.map((f) => f.id));
    const gpuIds = new Set(cfg.gpus.map((g) => g.id));
    const dup = (ids: string[], path: string) => {
      const seen = new Set<string>();
      for (const id of ids) {
        if (seen.has(id)) ctx.addIssue({ code: "custom", path: [path], message: `id dupliqué : ${id}` });
        seen.add(id);
      }
    };
    dup(cfg.gpus.map((g) => g.id), "gpus");
    dup(cfg.models.map((m) => m.id), "models");
    dup(cfg.engines.map((e) => e.id), "engines");
    dup(cfg.weightFormats.map((f) => f.id), "weightFormats");
    dup(cfg.kvCacheFormats.map((f) => f.id), "kvCacheFormats");

    const checkFormats = (map: Record<string, unknown>, known: Set<string>, path: (string | number)[], what: string) => {
      for (const k of Object.keys(map))
        if (!known.has(k)) ctx.addIssue({ code: "custom", path: [...path, k], message: `${what} inconnu : ${k}` });
    };
    cfg.gpus.forEach((g, i) => {
      checkFormats(g.weightFormats, wf, ["gpus", i, "weightFormats"], "format de poids");
      checkFormats(g.kvCacheFormats, kf, ["gpus", i, "kvCacheFormats"], "format KV");
    });
    cfg.engines.forEach((e, i) => {
      checkFormats(e.weightFormats, wf, ["engines", i, "weightFormats"], "format de poids");
      checkFormats(e.kvCacheFormats, kf, ["engines", i, "kvCacheFormats"], "format KV");
      checkFormats(e.gpus, gpuIds, ["engines", i, "gpus"], "GPU");
    });
    cfg.models.forEach((m, i) => {
      if (!wf.has(m.nativeWeightFormat))
        ctx.addIssue({ code: "custom", path: ["models", i, "nativeWeightFormat"], message: `format inconnu : ${m.nativeWeightFormat}` });
      for (const f of m.officialWeightFormats)
        if (!wf.has(f)) ctx.addIssue({ code: "custom", path: ["models", i, "officialWeightFormats"], message: `format inconnu : ${f}` });
    });
    const d = cfg.defaults;
    if (d.model && !cfg.models.some((m) => m.id === d.model))
      ctx.addIssue({ code: "custom", path: ["defaults", "model"], message: `modèle inconnu : ${d.model}` });
    if (d.gpu && !gpuIds.has(d.gpu)) ctx.addIssue({ code: "custom", path: ["defaults", "gpu"], message: `GPU inconnu : ${d.gpu}` });
    if (d.engine && !cfg.engines.some((e) => e.id === d.engine))
      ctx.addIssue({ code: "custom", path: ["defaults", "engine"], message: `moteur inconnu : ${d.engine}` });
  });

export type AppConfig = z.output<typeof configSchema>;
export type ModelSpec = AppConfig["models"][number];
export type ModelInput = z.input<typeof modelSchema>;
export type GpuSpec = AppConfig["gpus"][number];
export type EngineSpec = AppConfig["engines"][number];
export type WeightFormat = AppConfig["weightFormats"][number];
export type KvFormat = AppConfig["kvCacheFormats"][number];
export type PerformanceSettings = AppConfig["performance"];
