import { z } from "zod";

/** Niveau de support d'un format sur un GPU. */
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
  bandwidthTBs: z.number().positive().optional(),
  interconnect: z.strictObject({
    label: z.string(),
    nvlink: z.boolean(),
    bandwidthGBs: z.number().positive().optional(),
  }),
  /** GPUs dans un même serveur. */
  maxGpusPerNode: z.number().int().positive(),
  /** GPUs joignables en NVLink (1 = pas de NVLink). */
  nvlinkDomain: z.number().int().positive(),
  weightFormats: z.record(z.string(), supportSchema),
  kvCacheFormats: z.record(z.string(), supportSchema),
  notes: z.array(z.string()).default([]),
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
    /** Éléments d'état récurrent (SSM + conv) par couche et par requête. */
    stateElements: z.number().int().positive(),
    label: z.string().optional(),
  }),
]);
export type LayerGroup = z.output<typeof layerGroupSchema>;

const formatNoteSchema = z.strictObject({
  level: z.enum(["unsupported", "warning", "info"]),
  note: z.string(),
});

const modelSchema = z.strictObject({
  id: z.string(),
  name: z.string(),
  family: z.string(),
  /** Dépôt Hugging Face de référence, pour vérification. */
  source: z.string().optional(),
  paramsB: z.number().positive(),
  activeParamsB: z.number().positive().optional(),
  hiddenSize: z.number().int().positive(),
  vocabSize: z.number().int().positive(),
  tiedEmbeddings: z.boolean().default(false),
  /** Paramètres conservés en 16 bits en plus des embeddings (encodeur vision...). */
  extraUnquantizedParamsB: z.number().nonnegative().default(0),
  attentionHeads: z.number().int().positive(),
  /** Paramètres d'attention (répliqués en DP-attention). Calculé si absent. */
  attentionParamsB: z.number().nonnegative().optional(),
  layers: z.array(layerGroupSchema).min(1),
  maxContext: z.number().int().positive(),
  /** Contexte atteignable par extension RoPE (YaRN...). */
  maxContextExtended: z.number().int().positive().optional(),
  nativeWeightFormat: z.string(),
  officialWeightFormats: z.array(z.string()).default([]),
  /** Octets par élément des états récurrents (couches linéaires / Mamba). */
  stateBytesPerElement: z.number().positive().default(2),
  weightFormatNotes: z.record(z.string(), formatNoteSchema).default({}),
  kvFormatNotes: z.record(z.string(), formatNoteSchema).default({}),
  tags: z.array(z.string()).default([]),
  notes: z.array(z.string()).default([]),
  /** Fiabilité des paramètres d'architecture saisis. */
  confidence: z.enum(["high", "medium", "low"]).default("high"),
});

export const configSchema = z
  .object({
    defaults: z
      .object({
        model: z.string().optional(),
        gpu: z.string().optional(),
        gpuCount: z.number().int().min(1).max(8).optional(),
        contextTokens: z.number().int().positive().optional(),
        kvCacheFormat: z.string().optional(),
        reservePct: z.number().min(0).max(50).optional(),
      })
      .default({}),
    weightFormats: z.array(weightFormatSchema).min(1),
    kvCacheFormats: z.array(kvFormatSchema).min(1),
    gpus: z.array(gpuSchema).min(1),
    models: z.array(modelSchema).min(1),
  })
  .superRefine((cfg, ctx) => {
    const wf = new Set(cfg.weightFormats.map((f) => f.id));
    const kf = new Set(cfg.kvCacheFormats.map((f) => f.id));
    const dup = (ids: string[], path: string) => {
      const seen = new Set<string>();
      for (const id of ids) {
        if (seen.has(id)) ctx.addIssue({ code: "custom", path: [path], message: `id dupliqué : ${id}` });
        seen.add(id);
      }
    };
    dup(cfg.gpus.map((g) => g.id), "gpus");
    dup(cfg.models.map((m) => m.id), "models");
    dup([...wf], "weightFormats");
    dup([...kf], "kvCacheFormats");

    cfg.gpus.forEach((g, i) => {
      for (const k of Object.keys(g.weightFormats))
        if (!wf.has(k))
          ctx.addIssue({ code: "custom", path: ["gpus", i, "weightFormats", k], message: `format de poids inconnu : ${k}` });
      for (const k of Object.keys(g.kvCacheFormats))
        if (!kf.has(k))
          ctx.addIssue({ code: "custom", path: ["gpus", i, "kvCacheFormats", k], message: `format KV inconnu : ${k}` });
    });
    cfg.models.forEach((m, i) => {
      if (!wf.has(m.nativeWeightFormat))
        ctx.addIssue({ code: "custom", path: ["models", i, "nativeWeightFormat"], message: `format inconnu : ${m.nativeWeightFormat}` });
      for (const f of m.officialWeightFormats)
        if (!wf.has(f))
          ctx.addIssue({ code: "custom", path: ["models", i, "officialWeightFormats"], message: `format inconnu : ${f}` });
    });
    const d = cfg.defaults;
    if (d.model && !cfg.models.some((m) => m.id === d.model))
      ctx.addIssue({ code: "custom", path: ["defaults", "model"], message: `modèle inconnu : ${d.model}` });
    if (d.gpu && !cfg.gpus.some((g) => g.id === d.gpu))
      ctx.addIssue({ code: "custom", path: ["defaults", "gpu"], message: `GPU inconnu : ${d.gpu}` });
  });

export type AppConfig = z.output<typeof configSchema>;
export type ModelSpec = AppConfig["models"][number];
export type GpuSpec = AppConfig["gpus"][number];
export type WeightFormat = AppConfig["weightFormats"][number];
export type KvFormat = AppConfig["kvCacheFormats"][number];
