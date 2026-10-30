import { z } from "zod";
import type { Localized } from "./i18n";

/** Free text shown in the UI: one string for every language, or one per language. */
const text = z.union([z.string(), z.strictObject({ fr: z.string(), en: z.string() })]);

/**
 * Support level for a format on a GPU or in an inference engine.
 * unverified: no source confirms or rules out support (selectable, with a warning).
 */
export const SUPPORT_LEVELS = ["native", "limited", "unverified", "unsupported"] as const;
export type SupportLevel = (typeof SUPPORT_LEVELS)[number];

const supportSchema = z
  .union([
    z.enum(SUPPORT_LEVELS),
    z.strictObject({
      level: z.enum(SUPPORT_LEVELS),
      note: text.optional(),
      /** GPU memory format when the engine converts the file format (e.g. FP8 dequantized to BF16). */
      memoryFormat: z.string().optional(),
    }),
  ])
  .transform((v): { level: SupportLevel; note?: z.output<typeof text>; memoryFormat?: string } => (typeof v === "string" ? { level: v } : v));
export type Support = Localized<z.output<typeof supportSchema>>;

/** Engine capability: supported, unsupported, or unverified without a source. */
const capabilitySchema = z.union([z.boolean(), z.literal("unverified")]);
export type Capability = z.output<typeof capabilitySchema>;

const weightFormatSchema = z.strictObject({
  id: z.string(),
  label: z.string(),
  /** Effective bits per parameter, including quantization scales. */
  bitsPerParam: z.number().positive(),
  /** Compute throughput used for performance estimates. */
  compute: z.enum(["bf16", "fp8", "fp4"]),
  description: text.optional(),
});

const kvFormatSchema = z.strictObject({
  id: z.string(),
  label: z.string(),
  /** Effective bytes per stored element, including scales. */
  bytesPerElement: z.number().positive(),
  description: text.optional(),
});

const gpuSchema = z.strictObject({
  id: z.string(),
  name: text,
  architecture: z.string(),
  computeCapability: z.string(),
  /** Driver-visible memory, in GiB. */
  memoryGiB: z.number().positive(),
  memoryLabel: text,
  /** Memory bandwidth, in TB/s. */
  bandwidthTBs: z.number().positive(),
  /** Dense tensor-core throughput (without sparsity), in TFLOPS. */
  tflops: z.strictObject({
    bf16: z.number().positive(),
    fp8: z.number().positive().optional(),
    fp4: z.number().positive().optional(),
  }),
  interconnect: z.strictObject({
    label: text,
    nvlink: z.boolean(),
    /** GPU-to-GPU bandwidth per GPU, in GB/s. */
    bandwidthGBs: z.number().positive(),
  }),
  /** GPUs within one server. */
  maxGpusPerNode: z.number().int().positive(),
  /** GPUs reachable through NVLink (1 means no NVLink). */
  nvlinkDomain: z.number().int().positive(),
  /** Attached CPU memory (e.g. Grace LPDDR5X) per GPU, in GiB. 0 requires user input. */
  hostMemoryGiBPerGpu: z.number().nonnegative().default(0),
  /** CPU-to-GPU link per GPU, in GB/s (PCIe or NVLink-C2C). */
  hostLinkGBs: z.number().positive(),
  hostLinkLabel: text,
  weightFormats: z.record(z.string(), supportSchema),
  kvCacheFormats: z.record(z.string(), supportSchema),
  notes: z.array(text).default([]),
});

const engineSchema = z.strictObject({
  id: z.string(),
  name: z.string(),
  description: text.optional(),
  confidence: z.enum(["high", "medium", "low"]).default("high"),
  /** Version or revision described by this profile. */
  version: text.optional(),
  features: z.strictObject({
    /** Paged KV cache (PagedAttention): blocks allocated on demand. */
    kvPaging: capabilitySchema,
    /** KV pool preallocated on the GPU at startup (can coexist with paging). */
    kvPoolPreallocated: capabilitySchema,
    prefixCaching: capabilitySchema,
    continuousBatching: capabilitySchema,
    /** Attention kernels apply the sliding window for correct outputs. */
    slidingWindowAttention: capabilitySchema,
    /** KV blocks outside the window are freed, which is required to save VRAM. */
    slidingWindowFreesBlocks: capabilitySchema,
    tensorParallel: capabilitySchema,
    pipelineParallel: capabilitySchema,
    /** DP-attention + expert parallel. */
    dpAttention: capabilitySchema,
    /** KV cache offloading to CPU memory. */
    kvOffload: capabilitySchema,
  }),
  /** Details or sources for each capability. */
  featureNotes: z.record(z.string(), text).default({}),
  sources: z.array(z.strictObject({ label: text, url: z.string() })).default([]),
  /** Model families advertised by the engine (`family` values). Omitted means unrestricted. */
  modelFamilies: z.array(z.string()).optional(),
  /** Omitted means unsupported by the engine. */
  weightFormats: z.record(z.string(), supportSchema),
  kvCacheFormats: z.record(z.string(), supportSchema),
  /** Per-GPU restrictions (omitted means unrestricted). */
  gpus: z.record(z.string(), supportSchema).default({}),
  notes: z.array(text).default([]),
  /** Indicative command-line options. */
  flags: z
    .strictObject({
      memoryUtilization: z.string().optional(),
      kvCacheDtype: z.string().optional(),
      tensorParallel: z.string().optional(),
      pipelineParallel: z.string().optional(),
      dpAttention: z.string().optional(),
      kvOffload: text.optional(),
    })
    .default({}),
});

const gqaFields = {
  count: z.number().int().positive(),
  kvHeads: z.number().int().positive(),
  headDim: z.number().int().positive(),
  /** V head dimension when it differs from K. */
  headDimV: z.number().int().positive().optional(),
  /** Additional bytes per token per layer (indexer, etc.), independent of the KV format. */
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
    /** Cross-layer KV sharing: the layer reads an earlier layer's latent and stores only extraBytesPerToken. */
    kvShared: z.boolean().optional(),
  }),
  z.strictObject({
    kind: z.literal("linear"),
    count: z.number().int().positive(),
    /** Recurrent state elements (SSM / DeltaNet / KDA) per layer per request. */
    ssmElements: z.number().int().positive(),
    /** Short convolution state elements per layer per request. */
    convElements: z.number().int().nonnegative().default(0),
    label: text.optional(),
  }),
  z.strictObject({
    /** Compressed attention (DeepSeek V4 CSA/HCA): one shared K=V vector per stored entry. */
    kind: z.literal("compressed"),
    count: z.number().int().positive(),
    headDim: z.number().int().positive(),
    /** Context tokens pooled into one stored entry. Omitted: the layer keeps only its window (or reads another layer's entries). */
    compressRatio: z.number().int().positive().optional(),
    /** Most recent tokens kept uncompressed by every layer. */
    window: z.number().int().positive(),
    /** Additional bytes per compressed entry (sparse indexer keys), independent of the KV format. */
    extraBytesPerEntry: z.number().nonnegative().optional(),
  }),
]);
export type LayerGroup = Localized<z.output<typeof layerGroupSchema>>;

const formatNoteSchema = z.strictObject({
  level: z.enum(["unsupported", "warning", "info"]),
  note: text,
});

export const modelSchema = z.strictObject({
  id: z.string(),
  name: z.string(),
  family: z.string(),
  /** Reference Hugging Face repository. */
  source: z.string().optional(),
  /** Revision of the source config.json. */
  revision: z.string().optional(),
  paramsB: z.number().positive(),
  activeParamsB: z.number().positive().optional(),
  hiddenSize: z.number().int().positive(),
  vocabSize: z.number().int().positive(),
  tiedEmbeddings: z.boolean().default(false),
  /** Parameters kept at 16 bits in addition to embeddings (vision encoder, attention, etc.). */
  extraUnquantizedParamsB: z.number().nonnegative().default(0),
  attentionHeads: z.number().int().positive(),
  /** Total decoder layers (num_hidden_layers), if different from cache-bearing layers (Mamba/MLP, NAS). */
  numLayers: z.number().int().positive().optional(),
  /** Routed expert parameters (MoE), in billions: only these are sharded in DP-attention + EP. */
  expertParamsB: z.number().nonnegative().optional(),
  layers: z.array(layerGroupSchema).min(1),
  maxContext: z.number().int().positive(),
  /** Context reachable through RoPE extension (YaRN, etc.). */
  maxContextExtended: z.number().int().positive().optional(),
  nativeWeightFormat: z.string(),
  /** Effective bits in the native checkpoint if different from the generic format (e.g. INT4 g32 = 4.5). */
  nativeBitsPerParam: z.number().positive().optional(),
  officialWeightFormats: z.array(z.string()).default([]),
  /** Official checkpoint repository per format (native format uses source). */
  checkpoints: z.record(z.string(), z.string()).default({}),
  /** Bytes per recurrent state element (4 when mamba_ssm_cache_dtype = float32). */
  ssmStateBytes: z.number().positive().default(2),
  /** Bytes per convolution state element. */
  convStateBytes: z.number().positive().default(2),
  weightFormatNotes: z.record(z.string(), formatNoteSchema).default({}),
  kvFormatNotes: z.record(z.string(), formatNoteSchema).default({}),
  tags: z.array(z.string()).default([]),
  notes: z.array(text).default([]),
  /** Confidence in the supplied architecture parameters. */
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
        /** Fraction of memory bandwidth reached during decoding. */
        memoryEfficiency: z.number().min(0.05).max(1).default(0.7),
        /** Fraction of tensor-core throughput reached (MFU). */
        computeEfficiency: z.number().min(0.05).max(1).default(0.45),
        /** Latency of a decoding all-reduce / all-to-all, in microseconds. */
        collectiveLatencyUs: z.strictObject({ nvlink: z.number().positive(), pcie: z.number().positive(), network: z.number().positive() }).default({ nvlink: 8, pcie: 30, network: 40 }),
      })
      .default({ memoryEfficiency: 0.7, computeEfficiency: 0.45, collectiveLatencyUs: { nvlink: 8, pcie: 30, network: 40 } }),
    /** GPU counts offered above 8. */
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
      for (const [k, v] of Object.entries(e.weightFormats))
        if (v.memoryFormat && !wf.has(v.memoryFormat))
          ctx.addIssue({ code: "custom", path: ["engines", i, "weightFormats", k, "memoryFormat"], message: `format inconnu : ${v.memoryFormat}` });
      for (const k of Object.keys(e.featureNotes))
        if (!(k in e.features)) ctx.addIssue({ code: "custom", path: ["engines", i, "featureNotes", k], message: `capacité inconnue : ${k}` });
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

/** Configuration as written in the file, with per-language text. */
export type ConfigSource = z.output<typeof configSchema>;
export type ModelSource = ConfigSource["models"][number];
/** Configuration with every text resolved to one language (see localize in i18n.ts). */
export type AppConfig = Localized<ConfigSource>;
export type ModelSpec = AppConfig["models"][number];
export type ModelInput = z.input<typeof modelSchema>;
export type GpuSpec = AppConfig["gpus"][number];
export type EngineSpec = AppConfig["engines"][number];
export type WeightFormat = AppConfig["weightFormats"][number];
export type KvFormat = AppConfig["kvCacheFormats"][number];
export type PerformanceSettings = AppConfig["performance"];
