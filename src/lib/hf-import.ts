/**
 * Conversion d'un config.json Hugging Face (ou params.json Mistral) en entrée `models:` du YAML.
 * Utilisé par le script `pnpm run import-model` et par le panneau d'import de l'interface.
 */
import { Document, isMap, isSeq } from "yaml";
import type { LayerGroup, ModelInput } from "./config-schema";

type Json = Record<string, unknown>;

export interface ImportOptions {
  id?: string;
  name?: string;
  family?: string;
  /** Dépôt Hugging Face (org/nom). */
  source?: string;
  revision?: string;
  /** Paramètres totaux, en milliards (prioritaire sur les métadonnées et l'estimation). */
  paramsB?: number;
  activeParamsB?: number;
}

export interface ImportResult {
  model: ModelInput;
  warnings: string[];
  /** Origine du nombre de paramètres. */
  paramsSource: "option" | "métadonnées safetensors" | "nom du dépôt" | "estimation";
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

/** Normalise un params.json Mistral (dim, n_layers…) vers les clés Hugging Face. */
function fromMistralParams(p: Json): Json {
  const moe = obj(p.moe);
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
    quantization_config: p.quantization_config,
    vision_config: p.vision_encoder,
  };
}

function detectNativeFormat(q: Json | undefined, dtype: unknown): { format: string; bits?: number } {
  if (q) {
    const method = String(q.quant_method ?? "").toLowerCase();
    const algo = String(q.quant_algo ?? "").toLowerCase();
    if (method === "mxfp4") return { format: "mxfp4" };
    if (algo.includes("nvfp4") || method === "nvfp4") return { format: "nvfp4" };
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
  /** Experts routés (répartis en expert parallel). */
  routed: number;
  embeddings: number;
}

/** Estimation des paramètres à partir des dimensions (attention + MLP/MoE + couches linéaires). */
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
      attention += g.count * (q + H * (g.kvLoraRank + g.ropeHeadDim) + g.kvLoraRank * heads * (nope + v) + heads * v * H);
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
    // Nemotron-H : E = MoE, - = MLP dense, M/* = mixeurs sans MLP.
    moeLayers = experts ? [...pattern].filter((c) => c === "E").length : 0;
    denseLayers = [...pattern].filter((c) => c === "-").length;
  }
  const mlpMul = t.mlp_hidden_act === "relu2" ? 2 : 3; // relu² : pas de gate
  const dense = denseLayers * mlpMul * H * inter;
  const perExpert = mlpMul * expertHidden * moeInter;
  const routed = moeLayers * experts * perExpert;
  const sharedP = moeLayers * shared * mlpMul * H * sharedInter;
  const router = moeLayers * experts * H;

  const total = embeddings + attention + dense + routed + sharedP + router;
  const active = embeddings + attention + dense + moeLayers * topk * perExpert + sharedP + router;
  return { total, active, routed, embeddings };
}

function parseNominalParams(text: string): number | undefined {
  const m = /(?:^|[-_/ ])(\d+(?:\.\d+)?)\s*([BT])(?:[-_ ]|$)/i.exec(text);
  if (!m) return undefined;
  return parseFloat(m[1]) * (m[2].toUpperCase() === "T" ? 1000 : 1);
}

export function convertHfConfig(input: unknown, opts: ImportOptions = {}): ImportResult {
  const warnings: string[] = [];
  const root = obj(input);
  if (!root) throw new Error("Le fichier doit contenir un objet JSON.");

  // Enveloppe { repo, revision, config, parameterMetadata } (copies de config/upstream/).
  const wrapped = obj(root.config) && (root.repo !== undefined || root.parameterMetadata !== undefined);
  let cfg: Json = wrapped ? (obj(root.config) as Json) : root;
  const meta = wrapped ? obj(root.parameterMetadata) : undefined;
  const repo = opts.source ?? (wrapped && typeof root.repo === "string" ? root.repo : typeof cfg._name_or_path === "string" && cfg._name_or_path.includes("/") ? cfg._name_or_path : undefined);
  const revision = opts.revision ?? (wrapped && typeof root.revision === "string" ? root.revision : undefined);

  if (cfg.dim !== undefined && cfg.n_layers !== undefined) cfg = fromMistralParams(cfg);
  const text = obj(cfg.text_config) ?? obj(cfg.llm_config) ?? obj(cfg.language_config) ?? obj(cfg.thinker_config);
  const t: Json = text ? { ...cfg, ...text } : cfg;
  const multimodal = !!text || !!cfg.vision_config || !!cfg.vision_encoder;

  const H = num(t.hidden_size) ?? num(t.d_model);
  const L = num(t.num_hidden_layers) ?? num(t.num_layers);
  const heads = num(t.num_attention_heads);
  if (!H || !L || !heads) throw new Error("Champs hidden_size, num_hidden_layers ou num_attention_heads introuvables.");
  t.hidden_size = H;
  t.num_hidden_layers = L;
  const kvHeads = num(t.num_key_value_heads) ?? heads;
  const headDim = num(t.head_dim) || Math.round(H / heads);
  const vDim = num(t.v_head_dim);
  const vocab = num(t.vocab_size);
  if (!vocab) throw new Error("vocab_size introuvable.");
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

  // --- Couches linéaires
  let linearCount = 0;
  let linearParamsPerLayer = 0;
  if (typeof t.hybrid_override_pattern === "string") {
    // Nemotron-H : M = Mamba-2, * = attention, E/- = MLP / MoE.
    const p = t.hybrid_override_pattern;
    linearCount = [...p].filter((c) => c === "M").length;
    const mh = num(t.mamba_num_heads) ?? 0;
    const md = num(t.mamba_head_dim) ?? 0;
    const st = num(t.ssm_state_size) ?? 128;
    const ng = num(t.n_groups) ?? 1;
    const k = num(t.conv_kernel) ?? 4;
    if (linearCount) groups.push({ kind: "linear", count: linearCount, ssmElements: mh * md * st, convElements: (k - 1) * (mh * md + 2 * ng * st), label: "Mamba-2 (état constant)" });
    linearParamsPerLayer = H * (2 * mh * md + 2 * ng * st + mh) + mh * md * H;
    const attn = [...p].filter((c) => c === "*").length;
    if (attn) groups.push(gqa(attn, kvHeads, headDim));
    if (ssmDtype !== "float32") warnings.push("Type des états Mamba non précisé : 16 bits supposés (vLLM --mamba-ssm-cache-dtype).");
  } else if (linCfg && Array.isArray(linCfg.kda_layers)) {
    // Kimi Delta Attention (Kimi Linear / K3, GLM-5.3-Flash).
    linearCount = (linCfg.kda_layers as unknown[]).length;
    const h = num(linCfg.num_heads) ?? heads;
    const d = num(linCfg.head_dim) ?? 128;
    const k = num(linCfg.short_conv_kernel_size) ?? 4;
    groups.push({ kind: "linear", count: linearCount, ssmElements: h * d * d, convElements: (k - 1) * 3 * h * d, label: "KDA (état constant)" });
    linearParamsPerLayer = 6 * H * h * d; // q, k, v, o + portes
    if (ssmDtype !== "float32") {
      ssmStateBytes = 4;
      warnings.push("États KDA supposés en FP32 (hypothèse prudente).");
    }
  } else if (countType(/linear/) > 0 || (num(t.linear_num_value_heads) && num(t.full_attention_interval))) {
    // Gated DeltaNet (Qwen3-Next, Qwen3.5).
    linearCount = countType(/linear/) || L - Math.floor(L / (num(t.full_attention_interval) as number));
    const kh = num(t.linear_num_key_heads) ?? 16;
    const vh = num(t.linear_num_value_heads) ?? 32;
    const kd = num(t.linear_key_head_dim) ?? 128;
    const vd = num(t.linear_value_head_dim) ?? 128;
    const k = num(t.linear_conv_kernel_dim) ?? 4;
    groups.push({ kind: "linear", count: linearCount, ssmElements: vh * kd * vd, convElements: (k - 1) * (2 * kh * kd + vh * vd), label: "Gated DeltaNet (état constant)" });
    linearParamsPerLayer = H * (2 * kh * kd + 2 * vh * vd + 2 * vh) + vh * vd * H;
  }

  // --- Couches d'attention
  if (typeof t.hybrid_override_pattern === "string") {
    // déjà traité
  } else if (num(t.kv_lora_rank)) {
    const attnCount = linCfg && Array.isArray(linCfg.full_attn_layers) ? (linCfg.full_attn_layers as unknown[]).length : L - linearCount;
    const rope = t.mla_use_nope === true ? 0 : (num(t.qk_rope_head_dim) ?? 64);
    const rank = num(t.kv_lora_rank) as number;
    const idxDim = num(t.index_head_dim);
    if (idxDim) {
      let extra = idxDim + 4; // clés FP8 + échelle FP32
      if (t.index_kpool_compress === true && num(t.index_kpool)) extra = Math.round(extra / (num(t.index_kpool) as number));
      const types = Array.isArray(t.indexer_types) ? (t.indexer_types as string[]) : undefined;
      const fullIdx = Math.min(attnCount, types ? types.filter((x) => x === "full").length : attnCount);
      groups.push({ kind: "mla", count: fullIdx, kvLoraRank: rank, ropeHeadDim: rope, extraBytesPerToken: extra });
      if (attnCount - fullIdx > 0) groups.push({ kind: "mla", count: attnCount - fullIdx, kvLoraRank: rank, ropeHeadDim: rope });
    } else {
      groups.push({ kind: "mla", count: attnCount, kvLoraRank: rank, ropeHeadDim: rope });
    }
  } else if (Array.isArray(t.block_configs)) {
    // Modèles élagués par NAS (DeciLM / Nemotron Super).
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
    // MiMo-V2 : 0 = attention globale, 1 = fenêtre glissante.
    const p = t.hybrid_layer_pattern;
    const values = Array.isArray(p) ? (p as number[]) : Object.entries(p as Json).flatMap(([k, v]) => Array(num(v) ?? 0).fill(Number(k)));
    const full = values.filter((x) => x === 0).length;
    const slide = values.filter((x) => x === 1).length;
    if (slide) groups.push(swa(slide, num(t.swa_num_key_value_heads) ?? kvHeads, num(t.swa_head_dim) ?? headDim, window ?? 4096, num(t.swa_v_head_dim) ?? vDim));
    if (full) groups.push(gqa(full, kvHeads, headDim, vDim));
  } else if (countType(/sliding/) > 0) {
    // Gemma 3/4, gpt-oss : layer_types explicite.
    const slide = countType(/sliding/);
    const full = countType(/full/);
    groups.push(swa(slide, kvHeads, headDim, window ?? 4096, vDim));
    if (full) groups.push(gqa(full, num(t.num_global_key_value_heads) ?? kvHeads, num(t.global_head_dim) ?? headDim, num(t.global_head_dim) ? undefined : vDim));
  } else if (num(t.sliding_window_pattern) && window) {
    const every = num(t.sliding_window_pattern) as number;
    const full = Math.floor(L / every);
    groups.push(swa(L - full, kvHeads, headDim, window, vDim), gqa(full, kvHeads, headDim, vDim));
  } else if (num(t.attention_chunk_size) && (t.model_type === "llama4_text" || t.model_type === "llama4" || t.no_rope_layers !== undefined)) {
    // Llama 4 : attention par chunks 3 couches sur 4.
    const interval = num(t.nope_layer_interval) ?? 4;
    const full = Math.floor(L / interval);
    groups.push(swa(L - full, kvHeads, headDim, num(t.attention_chunk_size) as number, vDim), gqa(full, kvHeads, headDim, vDim));
    warnings.push("Attention par chunks modélisée comme une fenêtre glissante de la taille du chunk.");
  } else if (L - linearCount > 0) {
    groups.push(gqa(L - linearCount, kvHeads, headDim, vDim));
    if (window && t.use_sliding_window === undefined && !layerTypes)
      warnings.push(`sliding_window = ${window} présent sans layer_types : vérifiez si toutes les couches utilisent une fenêtre glissante.`);
  }
  if (!groups.length) throw new Error("Aucune couche à cache détectée.");

  // --- Paramètres
  const est = estimateParams(t, groups, tied, linearParamsPerLayer);
  let paramsSource: ImportResult["paramsSource"] = "estimation";
  let paramsB = est.total / 1e9;
  const metaParams = obj(meta?.parameters);
  const packed = metaParams && Object.keys(metaParams).some((k) => /^(U8|I8|I32|I64)$/.test(k));
  const nominal = repo ? parseNominalParams(repo.split("/").pop() ?? "") : undefined;
  if (opts.paramsB) {
    paramsB = opts.paramsB;
    paramsSource = "option";
  } else if (meta && num(meta.total) && !packed) {
    paramsB = (num(meta.total) as number) / 1e9;
    paramsSource = "métadonnées safetensors";
  } else if (nominal && (packed || Math.abs(nominal - paramsB) / nominal > 0.35)) {
    // Estimation non fiable (tenseurs empaquetés ou architecture mal reconnue) : on garde la valeur nominale.
    paramsB = nominal;
    paramsSource = "nom du dépôt";
    warnings.push(`Nombre de paramètres nominal pris dans le nom du dépôt (${nominal} B) ; l'estimation par dimensions donne ${round(est.total / 1e9, 1)} B.`);
  } else if (!nominal) {
    warnings.push(`Nombre de paramètres estimé à partir des dimensions (${round(paramsB, 1)} B) : précisez --params si vous le connaissez.`);
  }
  const ratio = paramsB / (est.total / 1e9);
  const experts = num(t.n_routed_experts) ?? num(t.num_experts) ?? num(t.num_local_experts) ?? 0;
  const activeB = opts.activeParamsB ?? (experts ? (est.active / 1e9) * Math.min(1.5, Math.max(0.67, ratio)) : undefined);
  if (experts && !opts.activeParamsB) warnings.push(`Paramètres actifs estimés (${round(activeB as number, 1)} B).`);
  if (Math.abs(ratio - 1) > 0.15 && paramsSource !== "estimation")
    warnings.push(`L'estimation par dimensions (${round(est.total / 1e9, 1)} B) s'écarte de ${Math.round((ratio - 1) * 100)} % du nombre retenu : architecture partiellement reconnue.`);

  // --- Format natif
  const q = obj(t.quantization_config) ?? obj(cfg.quantization_config);
  const nativeFmt = detectNativeFormat(q, t.torch_dtype ?? t.dtype ?? cfg.torch_dtype ?? cfg.dtype);
  let extraUnq = 0;
  if (metaParams && nativeFmt.format !== "bf16" && nativeFmt.format !== "fp16") {
    const bf16 = ((num(metaParams.BF16) ?? 0) + (num(metaParams.F16) ?? 0) + (num(metaParams.F32) ?? 0)) / 1e9;
    extraUnq = Math.max(0, bf16 - est.embeddings / 1e9);
  }

  // --- Contexte
  const maxPos = num(t.max_position_embeddings) ?? num(t.max_seq_len) ?? 32768;
  const rope = obj(t.rope_scaling) ?? obj(t.rope_parameters);
  if (!rope?.factor && maxPos <= 65536) warnings.push("Pas d'extension RoPE déclarée : renseignez maxContextExtended si le modèle supporte YaRN.");

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
    notes: ["Importé depuis config.json."],
    confidence: warnings.some((w) => w.includes("estimé") || w.includes("partiellement") || w.includes("vérifiez")) ? "medium" : "high",
  };
  return { model, warnings, paramsSource };
}

/** Sérialise une entrée de modèle pour la coller sous `models:` (couches en style compact). */
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
        if (isSeq(v)) v.flow = true;
      }
    }
  }
  return doc
    .toString({ lineWidth: 0 })
    .split("\n")
    .map((l) => (l ? `  ${l}` : l))
    .join("\n");
}

