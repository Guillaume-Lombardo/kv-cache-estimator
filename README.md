# KV cache estimator

A single-page Next.js / TypeScript application for estimating the memory required to serve open-weight LLMs on NVIDIA GPUs: model weights, KV cache per request, concurrent sessions and approximate throughput. It also supports inverse sizing and scenario comparison. The reference engine is vLLM, including deployments orchestrated by llm-d.

The application interface is in French. Repository documentation, Markdown files, source comments and docstrings are written in English.

## Quick start

Requirements: Node.js 22.13 or newer and Corepack. `package.json` pins the pnpm version.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm run dev                 # http://localhost:3000
# Production
pnpm run build && pnpm start
```

The server reads `config/estimator.yaml` at startup. In development, it reads the file again on each request. Use a YAML or JSON override without rebuilding the application:

```bash
KV_ESTIMATOR_CONFIG=/etc/kv/estimator.yaml pnpm start
pnpm run check-config                         # Validate the default catalogue
pnpm run check-config /etc/kv/estimator.yaml    # Validate another file
```

Invalid configuration produces an error page. The schema is strict: unknown or misspelled keys are rejected.

## Ready-to-run container

Once a release has completed its publication workflow, its image is available at `ghcr.io/guillaume-lombardo/kv-cache-estimator`. Images include the default catalogue and support `linux/amd64` and `linux/arm64`. No GPU is required to run the estimator.

```bash
# latest follows the most recently published stable release.
docker run --rm -p 3000:3000 \
  ghcr.io/guillaume-lombardo/kv-cache-estimator:latest
```

Open <http://localhost:3000>. For reproducible deployments, replace `latest` with a published release tag such as `v0.1.0`, or use the image digest recorded in the workflow summary. `v0.1.0` is an example, not a claim that this release already exists.

Mount a custom catalogue read-only:

```bash
docker run --rm -p 3000:3000 \
  --mount "type=bind,src=$PWD/my-catalogue.yaml,dst=/config/estimator.yaml,readonly" \
  -e KV_ESTIMATOR_CONFIG=/config/estimator.yaml \
  ghcr.io/guillaume-lombardo/kv-cache-estimator:latest
```

The image runs as the non-root `node` user. Mounted files must be readable by that user. The default port is 3000; `PORT`, `HOSTNAME` and `KV_ESTIMATOR_CONFIG` can be overridden at runtime.

To build and test an image locally:

```bash
docker build -t kv-cache-estimator:local .
bash scripts/smoke-container.sh kv-cache-estimator:local
```

## Releases and image publication

The GitHub Actions workflow validates the catalogue, runs TypeScript checks and tests, builds the application, then builds and smoke-tests the production container. The smoke test checks both the bundled catalogue and a mounted YAML override, and verifies that the container runs without root privileges.

Publishing a GitHub Release triggers image publication after those checks pass:

1. Merge the intended changes into `main`.
2. Create a release with a tag such as `v0.1.0` or `v0.2.0-rc.1`, pointing to the intended commit. Use the GitHub prerelease flag for release candidates.
3. Publish the release and wait for the `publish` job to succeed.
4. Use the version tag or digest from the workflow summary.

Stable releases publish their exact tag and `latest`. Prereleases publish only their exact tag and never replace `latest`. Draft releases and ordinary branch pushes do not publish images. Release tags are the image version authority; `package.json` is not used to generate image tags. Avoid republishing a version tag; use digests when immutability matters.

The workflow authenticates to GHCR with `GITHUB_TOKEN` and grants `packages: write` only to the publication job. It attaches OCI metadata, build provenance and an SBOM. The runtime smoke test runs on amd64; the publication job builds both architectures.

After the first publication, check the package's visibility in GitHub Packages. GHCR may create a private package initially. Set it to public for anonymous pulls, or authenticate Docker with an account that can read it. Repository or organization policies must allow Actions to publish packages.

## Import a model

Convert a Hugging Face `config.json` or Mistral `params.json` into a catalogue entry:

```bash
pnpm run import-model path/config.json
pnpm run import-model path/config.json --append
pnpm run import-model config.json --source org/repo --params 235 --active 22 --id my-model
```

Without `--append`, the command prints YAML. With `--append`, it adds the entry to the catalogue and validates the result. The UI's model import tab also accepts pasted or uploaded JSON. Imported models can be tested and saved in the current browser.

The converter recognizes GQA/MHA, sliding windows (`layer_types`, `hybrid_layer_pattern`, `sliding_window_pattern`), Llama 4 chunked attention, MLA and DSA indexers (including IndexShare and pooling), Mamba-2, Gated DeltaNet, KDA, NAS-pruned `block_configs`, and Mistral's `params.json` structure.

Parameter counts use this priority:

1. The explicit `--params` option.
2. Safetensors metadata from a `{repo, revision, config, parameterMetadata}` envelope, as stored in `config/upstream/`.
3. An estimate from architecture dimensions: attention, MLPs, experts and linear layers.
4. The repository name when a dimension-based estimate is unreliable, for example with packed tensors.

Active MoE parameters are estimates. The native format comes from `quantization_config`; SSM state uses FP32 when the configuration requires it (`mamba_ssm_cache_dtype`). The 24 upstream snapshots include source revisions. Tests check that the converter reproduces the catalogue's layer definitions for each snapshot.

## Inputs

| Input | Control |
|---|---|
| Model | Search by name, family or architecture keyword |
| Context | Doubling steps from 8k to 1M; extended and unsupported contexts are marked |
| GPU | V100, L40S, H100, H200, GH200 (96 / 144 GB), B200, GB200, B300, GB300 |
| GPU count | 1 to 8, then 16, 32, 64; 72 for NVL72 domains |
| Parallelism | TP/PP with selectable TP degree (PP = GPU count / TP), or DP-attention + EP for MoE |
| Engine | vLLM / llm-d or ZML / LLMD; both GPU and engine must support the format |
| Formats | BF16, FP16, FP8, INT8, INT4, MXFP4, NVFP4 weights; BF16, FP16, FP8, NVFP4 KV |
| Reserve | Percentage of each GPU reserved for activations and CUDA graphs, similar to `--gpu-memory-utilization` |
| Advanced | Visible memory per GPU from `nvidia-smi`, KV offload to CPU RAM |

The URL stores the scenario settings for sharing. Models imported only into browser storage must also be available in the receiving browser.

## Results and limitations

The main indicators show model weights (total and per GPU), logical and replicated KV per request, model plus one KV cache, free KV memory in GiB, and concurrent sessions. Other panels provide:

- Per-layer KV details, sessions at each context step and an indicative `vllm serve` command.
- Estimated per-session and aggregate throughput, prefill time, memory/compute limits and CPU offload.
- Inverse sizing: minimum GPU count and parallelism for a target session count, plus the maximum context on the current configuration.
- Up to six pinned scenarios for comparison, and a JSON export.

Unsupported options are disabled with a reason on hover. Limited or unverified options remain selectable with warnings. Format choices adjust when switching model, GPU or engine, with a notice explaining the fallback. Warnings are ordered by severity.

Any blocking error marks the configuration invalid. The UI retains theoretical memory figures but removes success indicators. Invalid configurations are excluded from sizing recommendations and comparison winners.

## Engine profiles

Each profile declares weight and KV format support (`native`, `limited`, `unverified`, `unsupported`). An unverified format remains selectable but carries a warning.

`memoryFormat` describes a conversion performed at load time, such as an FP8 file dequantized into BF16 GPU memory. Memory estimates use the loaded format. When the loaded precision is unverified, the estimate assumes the file format and also reports the BF16 alternative.

Capabilities use three states: `true`, `false`, or `unverified`. They cover KV paging, pool preallocation, prefix caching, continuous batching, sliding-window attention, block reclamation, TP, PP, DP-attention and offload. Paging and pool preallocation can coexist.

Sliding-window memory savings require both `slidingWindowAttention: true` and `slidingWindowFreesBlocks: true`. Missing attention support is a blocking error because model outputs would be incorrect. Unverified block reclamation means allocating the full context.

### ZML / LLMD

The profile distinguishes ZML's LLMD server from llm-d, the vLLM orchestrator.

Server capabilities (PagedAttention, continuous batching, tensor parallelism and prefix caching) come from the [LLMD documentation](https://zml.ai/llmd/). Format, KV precision and sliding-window observations refer to the CUDA backend of [ZML at revision 3500e88](https://github.com/zml/zml/tree/3500e88cf84e0e98d37b0f6048e6af9d33bc6108), dated October 9, 2026. They do not establish that LLMD exposes every library capability:

- FP8, MXFP4 and NVFP4 weights exist in the library and remain quantized in memory; LLMD loading support is unverified.
- INT4 and INT8 are TODOs in `zml/nn.zig`.
- FP8 KV is rejected by FA2/FA3 and disabled in the Triton wrapper.
- Kernels apply sliding-window attention, but the library does not reclaim pages outside the window, so the profile applies no VRAM savings.

Pool preallocation, PP, DP-attention and offload remain unverified in this profile.

## Calculation model

- Weights use parameter counts and effective bits, including quantization scales (for example INT4 g128 = 4.25 bits and NVFP4 = 4.5 bits). Embeddings, lm_head and designated unquantized parameters stay at 16 bits. `nativeBitsPerParam` handles specific native checkpoints, such as Kimi K2.5 INT4 g32 at 4.5 bits.
- GQA/MHA KV = layers x KV heads x (K dimension + V dimension) x bytes x tokens.
- Sliding-window KV is capped only when the engine both applies the window and frees expired blocks.
- MLA KV = layers x (latent rank + RoPE dimension) x bytes x tokens, plus DSA indexer storage and scales where applicable.
- Linear/SSM cache is recurrent state plus convolution state, constant per request. `ssmStateBytes` can require FP32.
- TP shards KV heads and replicates them when TP exceeds their count. Each TP rank stores a full MLA latent.
- PP uses whole layers and sizes for the most heavily loaded stage: ceil(layers / PP), selecting the most expensive cache layers because exact ordering is unknown. Input/output embeddings are included; tied embeddings are copied at both ends. PP greater than the decoder layer count is invalid.
- DP-attention + EP shards only routed experts (`expertParamsB`). All non-expert tensors remain on every GPU at their respective precisions. Unknown expert parameter counts block capacity validation. Each request resides on one GPU.
- Sessions = floor((usable memory on the limiting GPU - its weights) / its KV per request). DP-attention sums the per-GPU session counts. Every session uses its full selected context; prefix reuse is ignored.
- Total physical free KV memory = total usable memory - physical weights, including replicas. A separate conservative budget multiplies the limiting GPU's free memory by GPU count. This budget drives session capacity and is shown when it differs from physical free memory.
- Model plus one KV reports physical usage, including replication. Its fit verdict uses the limiting GPU; a logical sum is also shown when different.
- Recommendations exclude blocking format, parallelism, context and capacity errors. A visible-memory override applies only to its original GPU type.
- Launch commands use the native repository or a format-specific `checkpoints` mapping. Missing checkpoints are explicitly marked as required input.
- CPU offload parks queued, preempted or reusable sessions. It does not increase simultaneous decoding capacity. Reload time = per-GPU KV / CPU-to-GPU bandwidth.
- Decode step time = max(bytes read / effective bandwidth, FLOPs / effective compute throughput) + collective latency. MoE reads account for experts touched by the batch. PP traverses stages. Prefill uses FLOPs including causal attention. See `performance` in the catalogue for efficiency assumptions.

These are capacity and first-order performance estimates, not deployment benchmarks. Validate the chosen model, backend and hardware combination before relying on the result for production sizing.

## Configuration schema

Top-level sections are `defaults`, `performance`, `largeGpuCounts`, `weightFormats`, `kvCacheFormats`, `engines`, `gpus` and `models`. See the English comments in `config/estimator.yaml` and the strict schema in `src/lib/config-schema.ts`.

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

Layer kinds are `full`, `sliding` (`window`), `mla` (`kvLoraRank`, `ropeHeadDim`, `extraBytesPerToken`) and `linear` (`ssmElements`, `convElements`).

The catalogue contains 64 models across Mistral, Z.ai (GLM), Moonshot (Kimi), Xiaomi (MiMo), Qwen, NVIDIA, Google (Gemma), DeepSeek, OpenAI (gpt-oss) and Meta (Llama). Entries with a verified configuration revision have upstream snapshots in `config/upstream/`, originally collected on `feat/kv-cache-estimator`. Other entries use public architecture information with a `confidence` level (`high`, `medium`, `low`); the UI marks medium and low confidence. Some active parameter counts, including Kimi K3, GLM-5.3 and MiMo-V2.6-Pro, are estimates.

## Development checks

```bash
pnpm run check-config
pnpm run typecheck
pnpm test
pnpm run build
```

Tests cover formulas, TP/PP/DP replication, offload, throughput, sizing, the 24 upstream configuration imports and a configuration matrix. GitHub Actions also builds and smoke-tests the production image.

Keep source comments, docstrings and Markdown artifacts in English. Preserve French UI text and catalogue messages unless a change explicitly includes interface translation. Do not edit upstream configuration snapshots to translate external source data.
