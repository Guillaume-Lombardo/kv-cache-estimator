/**
 * Convert a Hugging Face config.json (or Mistral params.json) into a YAML `models:` entry.
 * Used by `pnpm run import-model` and the UI import panel.
 */
import { Document, isMap, isSeq } from "yaml";
import type { ModelInput } from "./config-schema";
import { tr, type Lang } from "./i18n";

type LayerGroup = ModelInput["layers"][number];

type Json = Record<string, unknown>;

export interface ImportOptions {
  id?: string;
  name?: string;
  family?: string;
  /** Hugging Face repository (org/name). */
  source?: string;
  revision?: string;
  /** Total parameters, in billions (overrides metadata and estimates). */
  paramsB?: number;
  activeParamsB?: number;
  /** Language of warnings and errors (French by default). Model text is written in both languages. */
  lang?: Lang;
}

export interface ImportResult {
  model: ModelInput;
  warnings: string[];
  /** Source of the parameter count. */
  paramsSource: "option" | "safetensors" | "repo-name" | "estimate";
}

const FAMILIES: Record<string, string> = {
  mistralai: "Mistral AI",
  "zai-org": "Z.ai (GLM)",
  thudm: "Z.ai (GLM)",
  moonshotai: "Moonshot (Kimi)",
  xiaomimimo: "Xiaomi (MiMo)",
  qwen: "Qwen (Alibaba)",
  nvidia: "NVIDIA",
  google: "Google (Gemma)",
  "deepseek-ai": "DeepSeek",
  openai: "OpenAI (gpt-oss)",
  "meta-llama": "Meta (Llama)",
};

const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
const obj = (v: unknown): Json | undefined => (v && typeof v === "object" && !Array.isArray(v) ? (v as Json) : undefined);
const round = (x: number, d = 2) => Math.round(x * 10 ** d) / 10 ** d;

/** Normalize Mistral params.json keys (dim, n_layers, etc.) to Hugging Face keys. */
function fromMistralParams(p: Json): Json {
  const moe = obj(p.moe);
  const dsa = obj(p.dsa);
  const kvSharing = obj(p.kv_sharing);
  const quantization = obj(p.quantization);
  return {
    hidden_size: p.dim,
    num_hidden_layers: p.n_layers,
    num_attention_heads: p.n_heads,
    num_key_value_heads: p.n_kv_heads,
    head_dim: p.head_dim,
    intermediate_size: p.hidden_dim,
    vocab_size: p.vocab_size,
    tie_word_embeddings: p.tied_embeddings,
    max_position_embeddings: p.max_position_embeddings ?? p.max_seq_len,
    kv_lora_rank: p.kv_lora_rank,
    q_lora_rank: p.q_lora_rank,
    qk_rope_head_dim: p.qk_rope_head_dim,
    qk_nope_head_dim: p.qk_nope_head_dim,
    v_head_dim: p.v_head_dim,
    n_routed_experts: moe?.num_experts,
    num_experts_per_tok: moe?.num_experts_per_tok,
    n_shared_experts: moe?.num_shared_experts,
    moe_intermediate_size: moe?.expert_hidden_dim,
    first_k_dense_replace: moe?.first_k_dense_replace,
    index_head_dim: dsa?.head_dim,
    indexer_types: typeof dsa?.sharing === "string" ? dsa.sharing.split(",") : undefined,
    kv_source_layers: typeof kvSharing?.source_layers_str === "string" ? kvSharing.source_layers_str.split(",").map(Number) : undefined,
    quantization_config: p.quantization_config ?? (String(quantization?.qformat_weight).toLowerCase().startsWith("fp8") ? { quant_method: "fp8" } : undefined),
    vision_config: p.vision_encoder,
  };
}

function detectNativeFormat(q: Json | undefined, dtype: unknown): { format: string; bits?: number } {
  if (q) {
    const method = String(q.quant_method ?? "").toLowerCase();
    const algo = String(q.quant_algo ?? "").toLowerCase();
    if (method === "mxfp4") return { format: "mxfp4" };
    // DeepSeek V4: FP4 experts (E8M0 scale per 32) dominate the weights, the rest is FP8.
    if (String(q.expert_dtype ?? "").toLowerCase() === "fp4") return { format: "mxfp4" };
    if (algo.includes("nvfp4") || method === "nvfp4") return { format: "nvfp4" };
    // ModelOpt mixed precision (e.g. NVFP4 experts, FP8 elsewhere): the experts dominate the weights.
    if (algo === "mixed_precision" && JSON.stringify(q.quantized_layers ?? {}).includes("NVFP4")) return { format: "nvfp4" };
    if (method === "fp8" || q.fmt === "e4m3" || algo.includes("fp8")) return { format: "fp8" };
    if (method === "awq" || method === "gptq") {
      const bits = num(q.bits) ?? 4;
      const gs = num(q.group_size) ?? 128;
      return bits <= 4 ? { format: "int4", bits: round(4 + 32 / gs, 3) } : { format: "int8" };
    }
    const groups = obj(q.config_groups);
    if (groups) {
      for (const g of Object.values(groups)) {
        const gg = obj(g);
        const w = obj(gg?.weights);
        const fmt = String(gg?.format ?? q.format ?? "");
        const bits = num(w?.num_bits);
        if (fmt.includes("mxfp4")) return { format: "mxfp4" };
        if (fmt.includes("nvfp4")) return { format: "nvfp4" };
        if (bits === 8) return { format: w?.type === "int" ? "int8" : "fp8" };
        if (bits === 4) {
          const gs = num(w?.group_size) ?? 128;
          return w?.type === "float" ? { format: "nvfp4" } : { format: "int4", bits: round(4 + 16 / gs, 3) };
        }
      }
    }
  }
  return { format: dtype === "float16" ? "fp16" : "bf16" };
}

interface ParamEstimate {
  total: number;
  active: number;
  /** Routed experts (sharded through expert parallelism). */
  routed: number;
  embeddings: number;
}

/** Estimate parameters from dimensions (attention + MLP/MoE + linear layers). */
function estimateParams(t: Json, groups: LayerGroup[], tied: boolean, linearParamsPerLayer: number): ParamEstimate {
  const H = num(t.hidden_size) ?? 0;
  const vocab = num(t.vocab_size) ?? 0;
  const heads = num(t.num_attention_heads) ?? 1;
  const L = num(t.num_hidden_layers) ?? groups.reduce((a, g) => a + g.count, 0);
  const embeddings = vocab * H * (tied ? 1 : 2);

  let attention = 0;
  for (const g of groups) {
    if (g.kind === "full" || g.kind === "sliding") {
      const dv = g.headDimV ?? g.headDim;
      attention += g.count * (H * heads * g.headDim + H * g.kvHeads * (g.headDim + dv) + heads * dv * H);
    } else if (g.kind === "mla") {
      const qLora = num(t.q_lora_rank) ?? 0;
      const nope = num(t.qk_nope_head_dim) ?? 128;
      const v = num(t.v_head_dim) ?? 128;
      const qk = nope + g.ropeHeadDim;
      const q = qLora ? H * qLora + qLora * heads * qk : H * heads * qk;
      const kvA = g.kvShared ? 0 : H * (g.kvLoraRank + g.ropeHeadDim);
      attention += g.count * (q + kvA + g.kvLoraRank * heads * (nope + v) + heads * v * H);
    } else if (g.kind === "compressed") {
      // DeepSeek V4: low-rank Q, one shared K = V projection, grouped low-rank output.
      const qLora = num(t.q_lora_rank) ?? 0;
      const oLora = num(t.o_lora_rank) ?? 0;
      const q = qLora ? H * qLora + qLora * heads * g.headDim : H * heads * g.headDim;
      const o = oLora ? heads * g.headDim * oLora + (num(t.o_groups) ?? 1) * oLora * H : heads * g.headDim * H;
      attention += g.count * (q + H * g.headDim + o);
    } else {
      attention += g.count * linearParamsPerLayer;
    }
  }

  const inter = num(t.intermediate_size) ?? 4 * H;
  const experts = num(t.n_routed_experts) ?? num(t.num_experts) ?? num(t.num_local_experts) ?? 0;
  const topk = num(t.num_experts_per_tok) ?? num(t.top_k_experts) ?? num(t.num_experts_per_token) ?? 0;
  const moeInter = num(t.moe_intermediate_size) ?? inter;
  const expertHidden = num(t.routed_expert_hidden_size) ?? num(t.moe_latent_size) ?? H;
  const shared = num(t.n_shared_experts) ?? num(t.num_shared_experts) ?? 0;
  const sharedInter = num(t.shared_expert_intermediate_size) ?? num(t.moe_shared_expert_intermediate_size) ?? moeInter;
  const pattern = typeof t.hybrid_override_pattern === "string" ? t.hybrid_override_pattern : undefined;
  let denseLayers = experts ? (num(t.first_k_dense_replace) ?? 0) : L;
  let moeLayers = experts ? L - denseLayers : 0;
  if (pattern) {
    // Nemotron-H: E = MoE, - = dense MLP, M/* = mixers without MLP.
    moeLayers = experts ? [...pattern].filter((c) => c === "E").length : 0;
    denseLayers = [...pattern].filter((c) => c === "-").length;
  }
  const mlpMul = t.mlp_hidden_act === "relu2" ? 2 : 3; // relu squared: no gate
  const dense = denseLayers * mlpMul * H * inter;
  const perExpert = mlpMul * expertHidden * moeInter;
  let routed = moeLayers * experts * perExpert;
  let routedActive = moeLayers * topk * perExpert;
  let router = moeLayers * experts * H;
  // NAS-pruned MoE (Nemotron 3.5 Bolt): expert count, size and top-k vary per block.
  const moeBlocks = Array.isArray(t.block_configs) ? (t.block_configs as Json[]).filter((b) => b.block_type === "moe" && num(b.n_routed_experts)) : [];
  if (moeBlocks.length) {
    const per = (b: Json) => mlpMul * expertHidden * (num(b.moe_intermediate_size) ?? moeInter);
    routed = moeBlocks.reduce((s, b) => s + (num(b.n_routed_experts) as number) * per(b), 0);
    routedActive = moeBlocks.reduce((s, b) => s + (num(b.num_experts_per_tok) ?? topk) * per(b), 0);
    router = moeBlocks.reduce((s, b) => s + (num(b.n_routed_experts) as number) * H, 0);
  }
  const sharedP = moeLayers * shared * mlpMul * H * sharedInter;

  const total = embeddings + attention + dense + routed + sharedP + router;
  const active = embeddings + attention + dense + routedActive + sharedP + router;
  return { total, active, routed, embeddings };
}

function parseNominalParams(text: string): number | undefined {
  const m = /(?:^|[-_/ ])(\d+(?:\.\d+)?)\s*([BT])(?:[-_ ]|$)/i.exec(text);
  if (!m) return undefined;
  return parseFloat(m[1]) * (m[2].toUpperCase() === "T" ? 1000 : 1);
}

export function convertHfConfig(input: unknown, opts: ImportOptions = {}): ImportResult {
  const warnings: string[] = [];
  const t_ = tr(opts.lang ?? "fr");
  /** Warnings that lower the confidence of the result. */
  let uncertain = false;
  const root = obj(input);
  if (!root) throw new Error(t_("Le fichier doit contenir un objet JSON.", "The file must contain a JSON object."));

  // Envelope { repo, revision, config, parameterMetadata } (config/upstream/ snapshots).
  const wrapped = obj(root.config) && (root.repo !== undefined || root.parameterMetadata !== undefined);
  let cfg: Json = wrapped ? (obj(root.config) as Json) : root;
  const meta = wrapped ? obj(root.parameterMetadata) : undefined;
  const repo = opts.source ?? (wrapped && typeof root.repo === "string" ? root.repo : typeof cfg._name_or_path === "string" && cfg._name_or_path.includes("/") ? cfg._name_or_path : undefined);
  const revision = opts.revision ?? (wrapped && typeof root.revision === "string" ? root.revision : undefined);

  if (cfg.dim !== undefined && cfg.n_layers !== undefined) cfg = fromMistralParams(cfg);
  const text = obj(cfg.text_config) ?? obj(cfg.llm_config) ?? obj(cfg.language_config) ?? obj(cfg.thinker_config);
  const t: Json = text ? { ...cfg, ...text } : cfg;
  const multimodal = !!text || !!cfg.vision_config || !!cfg.vision_encoder;
  if (Array.isArray(t.layers_block_type) && typeof t.hybrid_override_pattern !== "string") {
    // Nemotron-H written as block names (Nemotron 3.5): same layout as hybrid_override_pattern.
    const blocks = t.layers_block_type as string[];
    const codes: Record<string, string> = { mamba: "M", attention: "*", moe: "E" };
    t.hybrid_override_pattern = blocks.map((b) => codes[b] ?? "-").join("");
    t.num_hidden_layers ??= blocks.length;
  }

  const H = num(t.hidden_size) ?? num(t.d_model);
  const L = num(t.num_hidden_layers) ?? num(t.num_layers);
  const heads = num(t.num_attention_heads);
  if (!H || !L || !heads) throw new Error(t_("Champs hidden_size, num_hidden_layers ou num_attention_heads introuvables.", "hidden_size, num_hidden_layers or num_attention_heads fields not found."));
  t.hidden_size = H;
  t.num_hidden_layers = L;
  const kvHeads = num(t.num_key_value_heads) ?? heads;
  const headDim = num(t.head_dim) || Math.round(H / heads);
  const vDim = num(t.v_head_dim);
  const vocab = num(t.vocab_size);
  if (!vocab) throw new Error(t_("vocab_size introuvable.", "vocab_size not found."));
  const tied = Boolean(t.tie_word_embeddings ?? cfg.tie_word_embeddings ?? false);
  const window = num(t.sliding_window_size) ?? num(t.sliding_window);
  const gqa = (count: number, kv: number, hd: number, hdv?: number) =>
    ({ kind: "full", count, kvHeads: kv, headDim: hd, ...(hdv && hdv !== hd ? { headDimV: hdv } : {}) }) as LayerGroup;
  const swa = (count: number, kv: number, hd: number, w: number, hdv?: number) =>
    ({ kind: "sliding", count, kvHeads: kv, headDim: hd, ...(hdv && hdv !== hd ? { headDimV: hdv } : {}), window: w }) as LayerGroup;

  const groups: LayerGroup[] = [];
  let ssmStateBytes = 2;
  const ssmDtype = String(t.mamba_ssm_cache_dtype ?? t.mamba_ssm_dtype ?? "");
  if (ssmDtype === "float32") ssmStateBytes = 4;
  const linCfg = obj(t.linear_attn_config);
  const layerTypes = Array.isArray(t.layer_types) ? (t.layer_types as string[]) : undefined;
  const countType = (re: RegExp) => layerTypes?.filter((x) => re.test(x)).length ?? 0;

  // --- Linear layers
  let linearCount = 0;
  let linearParamsPerLayer = 0;
  if (typeof t.hybrid_override_pattern === "string") {
    // Nemotron-H: M = Mamba-2, * = attention, E/- = MLP / MoE.
    const p = t.hybrid_override_pattern;
    linearCount = [...p].filter((c) => c === "M").length;
    const mh = num(t.mamba_num_heads) ?? 0;
    const md = num(t.mamba_head_dim) ?? 0;
    const st = num(t.ssm_state_size) ?? 128;
    const ng = num(t.n_groups) ?? 1;
    const k = num(t.conv_kernel) ?? 4;
    if (linearCount) groups.push({ kind: "linear", count: linearCount, ssmElements: mh * md * st, convElements: (k - 1) * (mh * md + 2 * ng * st), label: { fr: "Mamba-2 (état constant)", en: "Mamba-2 (constant state)" } });
    linearParamsPerLayer = H * (2 * mh * md + 2 * ng * st + mh) + mh * md * H;
    const attn = [...p].filter((c) => c === "*").length;
    if (attn) groups.push(gqa(attn, kvHeads, headDim));
    if (ssmDtype !== "float32") warnings.push(t_("Type des états Mamba non précisé : 16 bits supposés (vLLM --mamba-ssm-cache-dtype).", "Mamba state type not specified: 16 bits assumed (vLLM --mamba-ssm-cache-dtype)."));
  } else if (linCfg && Array.isArray(linCfg.kda_layers)) {
    // Kimi Delta Attention (Kimi Linear / K3, GLM-5.3-Flash).
    linearCount = (linCfg.kda_layers as unknown[]).length;
    const h = num(linCfg.num_heads) ?? heads;
    const d = num(linCfg.head_dim) ?? 128;
    const k = num(linCfg.short_conv_kernel_size) ?? 4;
    groups.push({ kind: "linear", count: linearCount, ssmElements: h * d * d, convElements: (k - 1) * 3 * h * d, label: { fr: "KDA (état constant)", en: "KDA (constant state)" } });
    linearParamsPerLayer = 6 * H * h * d; // q, k, v, o + gates
    if (ssmDtype !== "float32") {
      ssmStateBytes = 4;
      warnings.push(t_("États KDA supposés en FP32 (hypothèse prudente).", "KDA states assumed to be FP32 (conservative assumption)."));
    }
  } else if (countType(/linear/) > 0 || (num(t.linear_num_value_heads) && num(t.full_attention_interval))) {
    // Gated DeltaNet (Qwen3-Next, Qwen3.5).
    linearCount = countType(/linear/) || L - Math.floor(L / (num(t.full_attention_interval) as number));
    const kh = num(t.linear_num_key_heads) ?? 16;
    const vh = num(t.linear_num_value_heads) ?? 32;
    const kd = num(t.linear_key_head_dim) ?? 128;
    const vd = num(t.linear_value_head_dim) ?? 128;
    const k = num(t.linear_conv_kernel_dim) ?? 4;
    groups.push({ kind: "linear", count: linearCount, ssmElements: vh * kd * vd, convElements: (k - 1) * (2 * kh * kd + vh * vd), label: { fr: "Gated DeltaNet (état constant)", en: "Gated DeltaNet (constant state)" } });
    linearParamsPerLayer = H * (2 * kh * kd + 2 * vh * vd + 2 * vh) + vh * vd * H;
  }

  // --- Attention layers
  if (typeof t.hybrid_override_pattern === "string") {
    // Already handled
  } else if (Array.isArray(t.compress_ratios)) {
    // DeepSeek V4 CSA/HCA: each layer keeps a window of raw K = V vectors; a layer with ratio r also
    // stores one pooled vector per r tokens. Entries after num_hidden_layers are MTP layers. V4.1
    // compresses only in kv_source_layer_ids (later layers read those entries); V4 compresses in every
    // layer with a ratio and indexes the ratio-4 ones, as in the reference inference code.
    const ratios = (t.compress_ratios as number[]).slice(0, L);
    const sources = Array.isArray(t.kv_source_layer_ids) ? (t.kv_source_layer_ids as number[]) : undefined;
    const indexSources = Array.isArray(t.index_source_layer_ids) ? (t.index_source_layer_ids as number[]) : undefined;
    const idxDim = num(t.index_head_dim);
    // FP4 indexer keys + one 4-byte scale per entry, as in DeepSeek's published per-token KV sizes.
    const idxBytes = idxDim ? idxDim / 2 + 4 : 0;
    const buckets = new Map<string, LayerGroup & { kind: "compressed" }>();
    ratios.forEach((r, i) => {
      const owns = r > 0 && (!sources || sources.includes(i));
      const indexed = owns && idxBytes > 0 && (sources ? !indexSources || indexSources.includes(i) : r === 4);
      const key = `${owns ? r : 0}/${indexed}`;
      const g = buckets.get(key);
      if (g) g.count++;
      else buckets.set(key, { kind: "compressed", count: 1, headDim, ...(owns ? { compressRatio: r } : {}), window: window ?? 128, ...(indexed ? { extraBytesPerEntry: idxBytes } : {}) });
    });
    groups.push(...buckets.values());
  } else if (num(t.kv_lora_rank)) {
    const attnCount = linCfg && Array.isArray(linCfg.full_attn_layers) ? (linCfg.full_attn_layers as unknown[]).length : L - linearCount;
    const rope = t.mla_use_nope === true ? 0 : (num(t.qk_rope_head_dim) ?? 64);
    const rank = num(t.kv_lora_rank) as number;
    const idxDim = num(t.index_head_dim);
    const types = Array.isArray(t.indexer_types) ? (t.indexer_types as string[]) : undefined;
    const sources = Array.isArray(t.kv_source_layers) ? (t.kv_source_layers as number[]) : undefined;
    let extra = idxDim ? idxDim + 4 : 0; // FP8 keys + FP32 scale
    if (idxDim && t.index_kpool_compress === true && num(t.index_kpool)) extra = Math.round(extra / (num(t.index_kpool) as number));
    if (sources && sources.length === attnCount && sources.some((s, i) => s !== i)) {
      // Cross-layer KV sharing (Mistral Large 4): a layer reading an earlier layer's latent
      // stores only its own indexer keys. Group layers by (shared latent, own indexer).
      const buckets = new Map<string, LayerGroup & { kind: "mla" }>();
      sources.forEach((s, i) => {
        const kvShared = s !== i;
        const indexed = extra > 0 && (!types || types[i] === "full");
        const key = `${kvShared}/${indexed}`;
        const g = buckets.get(key);
        if (g) g.count++;
        else buckets.set(key, { kind: "mla", count: 1, kvLoraRank: rank, ropeHeadDim: rope, ...(indexed ? { extraBytesPerToken: extra } : {}), ...(kvShared ? { kvShared: true } : {}) });
      });
      groups.push(...buckets.values());
    } else if (idxDim) {
      const fullIdx = Math.min(attnCount, types ? types.filter((x) => x === "full").length : attnCount);
      groups.push({ kind: "mla", count: fullIdx, kvLoraRank: rank, ropeHeadDim: rope, extraBytesPerToken: extra });
      if (attnCount - fullIdx > 0) groups.push({ kind: "mla", count: attnCount - fullIdx, kvLoraRank: rank, ropeHeadDim: rope });
    } else {
      groups.push({ kind: "mla", count: attnCount, kvLoraRank: rank, ropeHeadDim: rope });
    }
  } else if (Array.isArray(t.block_configs)) {
    // NAS-pruned models (DeciLM / Nemotron Super).
    const blocks = t.block_configs as Json[];
    const byKv = new Map<number, number>();
    for (const b of blocks) {
      const a = obj(b.attention);
      if (!a || a.no_op === true || a.replace_with_linear === true) continue;
      const kv = Math.round(heads / (num(a.n_heads_in_group) ?? 1));
      byKv.set(kv, (byKv.get(kv) ?? 0) + 1);
    }
    for (const [kv, count] of byKv) groups.push(gqa(count, kv, headDim));
  } else if (obj(t.hybrid_layer_pattern) || Array.isArray(t.hybrid_layer_pattern)) {
    // MiMo-V2: 0 = global attention, 1 = sliding window.
    const p = t.hybrid_layer_pattern;
    const values = Array.isArray(p) ? (p as number[]) : Object.entries(p as Json).flatMap(([k, v]) => Array(num(v) ?? 0).fill(Number(k)));
    const full = values.filter((x) => x === 0).length;
    const slide = values.filter((x) => x === 1).length;
    if (slide) groups.push(swa(slide, num(t.swa_num_key_value_heads) ?? kvHeads, num(t.swa_head_dim) ?? headDim, window ?? 4096, num(t.swa_v_head_dim) ?? vDim));
    if (full) groups.push(gqa(full, kvHeads, headDim, vDim));
  } else if (countType(/sliding/) > 0) {
    // Gemma 3/4, gpt-oss: explicit layer_types.
    const slide = countType(/sliding/);
    const full = countType(/full/);
    groups.push(swa(slide, kvHeads, headDim, window ?? 4096, vDim));
    if (full) groups.push(gqa(full, num(t.num_global_key_value_heads) ?? kvHeads, num(t.global_head_dim) ?? headDim, num(t.global_head_dim) ? undefined : vDim));
  } else if (num(t.sliding_window_pattern) && window) {
    const every = num(t.sliding_window_pattern) as number;
    const full = Math.floor(L / every);
    groups.push(swa(L - full, kvHeads, headDim, window, vDim), gqa(full, kvHeads, headDim, vDim));
  } else if (num(t.attention_chunk_size) && (t.model_type === "llama4_text" || t.model_type === "llama4" || t.no_rope_layers !== undefined)) {
    // Llama 4: chunked attention in 3 out of 4 layers.
    const interval = num(t.nope_layer_interval) ?? 4;
    const full = Math.floor(L / interval);
    groups.push(swa(L - full, kvHeads, headDim, num(t.attention_chunk_size) as number, vDim), gqa(full, kvHeads, headDim, vDim));
    warnings.push(t_("Attention par chunks modélisée comme une fenêtre glissante de la taille du chunk.", "Chunked attention modeled as a sliding window of the chunk size."));
  } else if (L - linearCount > 0) {
    groups.push(gqa(L - linearCount, kvHeads, headDim, vDim));
    if (window && t.use_sliding_window === undefined && !layerTypes) {
      uncertain = true;
      warnings.push(
        t_(
          `sliding_window = ${window} présent sans layer_types : vérifiez si toutes les couches utilisent une fenêtre glissante.`,
          `sliding_window = ${window} present without layer_types: check whether every layer uses a sliding window.`,
        ),
      );
    }
  }
  if (!groups.length) throw new Error(t_("Aucune couche à cache détectée.", "No cache-bearing layer detected."));

  // --- Parameters
  const est = estimateParams(t, groups, tied, linearParamsPerLayer);
  let paramsSource: ImportResult["paramsSource"] = "estimate";
  let paramsB = est.total / 1e9;
  const metaParams = obj(meta?.parameters);
  const packed = metaParams && Object.keys(metaParams).some((k) => /^(U8|I8|I32|I64)$/.test(k));
  const nominal = repo ? parseNominalParams(repo.split("/").pop() ?? "") : undefined;
  if (opts.paramsB) {
    paramsB = opts.paramsB;
    paramsSource = "option";
  } else if (meta && num(meta.total) && !packed) {
    paramsB = (num(meta.total) as number) / 1e9;
    paramsSource = "safetensors";
  } else if (nominal && (packed || Math.abs(nominal - paramsB) / nominal > 0.35)) {
    // Unreliable estimate (packed tensors or unrecognized architecture): keep the nominal value.
    paramsB = nominal;
    paramsSource = "repo-name";
    warnings.push(
      t_(
        `Nombre de paramètres nominal pris dans le nom du dépôt (${nominal} B) ; l'estimation par dimensions donne ${round(est.total / 1e9, 1)} B.`,
        `Nominal parameter count taken from the repository name (${nominal} B); the estimate from dimensions gives ${round(est.total / 1e9, 1)} B.`,
      ),
    );
  } else if (!nominal) {
    uncertain = true;
    warnings.push(
      t_(
        `Nombre de paramètres estimé à partir des dimensions (${round(paramsB, 1)} B) : précisez --params si vous le connaissez.`,
        `Parameter count estimated from dimensions (${round(paramsB, 1)} B): pass --params if you know it.`,
      ),
    );
  }
  const ratio = paramsB / (est.total / 1e9);
  const experts = num(t.n_routed_experts) ?? num(t.num_experts) ?? num(t.num_local_experts) ?? 0;
  const activeB = opts.activeParamsB ?? (experts ? (est.active / 1e9) * Math.min(1.5, Math.max(0.67, ratio)) : undefined);
  if (experts && !opts.activeParamsB) {
    uncertain = true;
    warnings.push(t_(`Paramètres actifs estimés (${round(activeB as number, 1)} B).`, `Active parameters estimated (${round(activeB as number, 1)} B).`));
  }
  if (Math.abs(ratio - 1) > 0.15 && paramsSource !== "estimate") {
    uncertain = true;
    warnings.push(
      t_(
        `L'estimation par dimensions (${round(est.total / 1e9, 1)} B) s'écarte de ${Math.round((ratio - 1) * 100)} % du nombre retenu : architecture partiellement reconnue.`,
        `The estimate from dimensions (${round(est.total / 1e9, 1)} B) differs by ${Math.round((ratio - 1) * 100)}% from the retained count: architecture partially recognized.`,
      ),
    );
  }

  // --- Native format
  const q = obj(t.quantization_config) ?? obj(cfg.quantization_config);
  const nativeFmt = detectNativeFormat(q && t.expert_dtype ? { ...q, expert_dtype: t.expert_dtype } : q, t.torch_dtype ?? t.dtype ?? cfg.torch_dtype ?? cfg.dtype);
  let extraUnq = 0;
  if (metaParams && nativeFmt.format !== "bf16" && nativeFmt.format !== "fp16") {
    const bf16 = ((num(metaParams.BF16) ?? 0) + (num(metaParams.F16) ?? 0) + (num(metaParams.F32) ?? 0)) / 1e9;
    extraUnq = Math.max(0, bf16 - est.embeddings / 1e9);
  }

  // --- Context
  const maxPos = num(t.max_position_embeddings) ?? num(t.max_seq_len) ?? 32768;
  const rope = obj(t.rope_scaling) ?? obj(t.rope_parameters);
  if (!rope?.factor && maxPos <= 65536) warnings.push(
      t_(
        "Pas d'extension RoPE déclarée : renseignez maxContextExtended si le modèle supporte YaRN.",
        "No RoPE extension declared: set maxContextExtended if the model supports YaRN.",
      ),
    );

  const repoName = repo?.split("/").pop();
  const name = opts.name ?? repoName ?? String(t.model_type ?? "modèle importé");
  const id = opts.id ?? name.toLowerCase().replace(/[^a-z0-9.]+/g, "-").replace(/^-|-$/g, "");
  const org = repo?.split("/")[0]?.toLowerCase();
  const tags: string[] = [];
  if (experts) tags.push("moe");
  if (multimodal) tags.push("vision");
  if (groups.some((g) => g.kind === "linear")) tags.push("hybrid");
  if (groups.some((g) => g.kind === "sliding")) tags.push("swa");

  const model: ModelInput = {
    id,
    name,
    family: opts.family ?? (org && FAMILIES[org]) ?? "Importé",
    ...(repo ? { source: repo } : {}),
    ...(revision ? { revision } : {}),
    paramsB: round(paramsB, 2),
    ...(activeB ? { activeParamsB: round(activeB, 1) } : {}),
    hiddenSize: H,
    vocabSize: vocab,
    ...(tied ? { tiedEmbeddings: true } : {}),
    ...(extraUnq > 0.05 ? { extraUnquantizedParamsB: round(extraUnq, 2) } : {}),
    attentionHeads: heads,
    ...(L > groups.reduce((a, g) => a + g.count, 0) ? { numLayers: L } : {}),
    ...(experts && est.routed > 0 ? { expertParamsB: round((est.routed / 1e9) * Math.min(1.5, Math.max(0.67, ratio)), 1) } : {}),
    layers: groups,
    maxContext: maxPos,
    nativeWeightFormat: nativeFmt.format,
    ...(nativeFmt.bits ? { nativeBitsPerParam: nativeFmt.bits } : {}),
    officialWeightFormats: [nativeFmt.format],
    ...(ssmStateBytes !== 2 ? { ssmStateBytes } : {}),
    tags,
    notes: [{ fr: "Importé depuis config.json.", en: "Imported from config.json." }],
    confidence: uncertain ? "medium" : "high",
  };
  return { model, warnings, paramsSource };
}

/** Serialize a model entry for insertion under `models:` (compact layer formatting). */
export function modelToYaml(model: ModelInput): string {
  const doc = new Document([model]);
  const seq = doc.contents;
  if (isSeq(seq)) {
    const m = seq.items[0];
    if (isMap(m)) {
      const layers = m.get("layers", true);
      if (isSeq(layers)) for (const it of layers.items) if (isMap(it)) it.flow = true;
      for (const key of ["officialWeightFormats", "tags", "notes"]) {
        const v = m.get(key, true);
        if (!isSeq(v)) continue;
        // Per-language notes ({ fr, en }) go one per line.
        if (v.items.some(isMap)) {
          for (const it of v.items) if (isMap(it)) it.flow = true;
        } else v.flow = true;
      }
    }
  }
  return doc
    .toString({ lineWidth: 0 })
    .split("\n")
    .map((l) => (l ? `  ${l}` : l))
    .join("\n");
}

