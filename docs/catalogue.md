# Catalogue

The catalogue (`config/estimator.yaml`) lists formats, engines, GPUs and models. This page covers its schema and how to add models. Back to the [README](../README.md).

## Loading and validation

The server reads `config/estimator.yaml` at startup. In development, it reads the file again on each request. Use a YAML or JSON override without rebuilding the application:

```bash
KV_ESTIMATOR_CONFIG=/etc/kv/estimator.yaml pnpm start
pnpm run check-config                         # Validate the default catalogue
pnpm run check-config /etc/kv/estimator.yaml    # Validate another file
```

Invalid configuration produces an error page. The schema is strict: unknown or misspelled keys are rejected.

## Configuration schema

Top-level sections are `defaults`, `performance`, `largeGpuCounts`, `intermediateGpuCounts`, `weightFormats`, `kvCacheFormats`, `engines`, `gpus` and `models`. See the English comments in `config/estimator.yaml` and the strict schema in `src/lib/config-schema.ts`.

`largeGpuCounts` (default `[16, 32, 64, 72]`) are always offered above 8 GPUs, 72 only for NVL72 domains. `intermediateGpuCounts` (default `[12, 24, 48]`, optional) are offered only for cards whose servers or interconnect domain they fill, and when the model and engine have a valid layout at that size; set it to `[]` to keep the previous list. See [GPU counts](calculation.md#gpu-counts).

Example model entry to add under `models` in a complete catalogue:

```yaml
- id: qwen3-32b
  name: Qwen3-32B
  family: Qwen (Alibaba)
  source: Qwen/Qwen3-32B
  revision: 9216db5781bf21249d130ec9da846c4624c16137
  paramsB: 32.76
  hiddenSize: 5120
  vocabSize: 151936
  attentionHeads: 64
  layers:
    - { kind: full, count: 64, kvHeads: 8, headDim: 128 }
  maxContext: 32768
  maxContextExtended: 131072
  nativeWeightFormat: bf16
  officialWeightFormats: [bf16, fp8, int4]
```

Layer kinds are `full`, `sliding` (`window`), `mla` (`kvLoraRank`, `ropeHeadDim`, `extraBytesPerToken`, `kvShared`) and `linear` (`ssmElements`, `convElements`).

GPU entries can set `family` and `variant` (picker grouping: the family is chosen first, then the variant), `segment` (`datacenter` or `pcie`), `vendor` (`nvidia` or `amd`), `interconnect.fabric` (the scale-up fabric name used in messages, `NVLink` by default) and `priceUsdPerHour`. `defaults.workload` sets the initial workload. All new fields are optional, so catalogues written for earlier versions still validate.

Text shown in the interface (descriptions, notes, support notes, labels, GPU names) accepts either one string for both languages or one string per language:

```yaml
notes:
  - Same text in both languages.
  - { fr: "Licence Mistral Research (usage commercial soumis à accord).", en: "Mistral Research License (commercial use requires an agreement)." }
```

The catalogue contains 91 models across Mistral, Z.ai (GLM), Moonshot (Kimi), Xiaomi (MiMo), Qwen, NVIDIA, Google (Gemma), DeepSeek, OpenAI (gpt-oss), Meta (Llama and Muse), MiniMax, Reflection AI, Aleph Alpha, StepFun, Meituan (LongCat), inclusionAI (Ling), Tencent (Hunyuan), Baidu (ERNIE), Cohere, IBM (Granite) and Microsoft (Phi). Entries with a verified configuration revision have upstream snapshots in `config/upstream/`, originally collected on `feat/kv-cache-estimator`. Other entries use public architecture information with a `confidence` level (`high`, `medium`, `low`); the UI marks medium and low confidence. Some active parameter counts, including Kimi K3, GLM-5.3, MiMo-V2.6-Pro and Ring-2.6-1T, are estimates. Muse Spark Open's official repository is gated, so its architecture comes from a public republication of its `config.json`.

## Import a model

Convert a Hugging Face `config.json` or Mistral `params.json` into a catalogue entry:

```bash
pnpm run import-model --hf Qwen/Qwen3-8B          # download config.json and the parameter count from the Hub
pnpm run import-model path/config.json
pnpm run import-model path/config.json --append
pnpm run import-model config.json --source org/repo --params 235 --active 22 --id my-model
```

Without `--append`, the command prints YAML. With `--append`, it adds the entry to the catalogue and validates the result. The UI's model import tab fetches a repository by name through the server (`/api/hf-model?repo=org/name`, which only contacts `huggingface.co`) or accepts pasted or uploaded JSON. Imported models can be tested and saved in the current browser. Gated repositories need a token: set `HF_TOKEN` for the server or the CLI. Without network access, paste the file instead.

The converter recognizes GQA/MHA, sliding windows (`layer_types`, `hybrid_layer_pattern`, `sliding_window_pattern`), Llama 4 chunked attention, MLA and DSA indexers (including IndexShare and pooling), cross-layer KV sharing (`kv_source_layers`), Mamba-2, Gated DeltaNet, KDA, NAS-pruned `block_configs`, and Mistral's `params.json` structure.

It also recognizes DeepSeek V4 compressed attention (`compress_ratios`, including V4.1 shared compressed caches) and Nemotron-H `layers_block_type`.

Parameter counts use this priority:

1. The explicit `--params` option.
2. Safetensors metadata from a `{repo, revision, config, parameterMetadata}` envelope, as stored in `config/upstream/`.
3. An estimate from architecture dimensions: attention, MLPs, experts and linear layers.
4. The repository name when a dimension-based estimate is unreliable, for example with packed tensors.

Active MoE parameters are estimates. The native format comes from `quantization_config`; SSM state uses FP32 when the configuration requires it (`mamba_ssm_cache_dtype`). The 42 upstream snapshots include source revisions. Tests check that the converter reproduces the catalogue's layer definitions for each snapshot.
