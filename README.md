# KV cache estimator

A single-page Next.js / TypeScript application for estimating the memory required to serve open-weight LLMs on NVIDIA GPUs: model weights, KV cache per request, concurrent sessions and approximate throughput. It also supports inverse sizing and scenario comparison. The reference engine is vLLM, including deployments orchestrated by llm-d.

The interface is available in French and English, with a light, dark or system theme. The buttons at the top right switch both; the choice is stored in cookies (`kv-lang`, `kv-theme`) so the server renders the right language and theme directly. Without a cookie, the language follows the browser's `Accept-Language` header (French by default). Repository documentation, Markdown files, source comments and docstrings are written in English.

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

To serve the application under a URL prefix instead of the root, set `KV_ESTIMATOR_BASE_PATH` for both the build and the server, for example `KV_ESTIMATOR_BASE_PATH=/kv-estimator pnpm run build && KV_ESTIMATOR_BASE_PATH=/kv-estimator pnpm start`, then open <http://localhost:3000/kv-estimator>. See [Serving under a URL prefix](#serving-under-a-url-prefix).

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

The image runs as the non-root `node` user. Mounted files must be readable by that user. The default port is 3000; `PORT`, `HOSTNAME` and `KV_ESTIMATOR_CONFIG` can be overridden at runtime. `KV_ESTIMATOR_BASE_PATH` is a build argument; see [Serving under a URL prefix](#serving-under-a-url-prefix).

To build and test an image locally:

```bash
docker build -t kv-cache-estimator:local .
bash scripts/smoke-container.sh kv-cache-estimator:local
```

## Serving under a URL prefix

By default the application is served at the root (`/`). To expose it under a prefix such as `/kv-estimator`, for example behind a reverse proxy that shares a host name with other services, build it with `KV_ESTIMATOR_BASE_PATH`. The value is passed to Next.js as [`basePath`](https://nextjs.org/docs/app/api-reference/config/next-config-js/basePath), so pages, `/_next/static` CSS and JavaScript, and redirects all carry the prefix.

- The value must be empty (root, the default) or a path that starts with `/` and has no trailing slash, such as `/kv-estimator` or `/tools/kv-estimator`. Any other value, such as `/kv-estimator/`, `kv-estimator` or `/`, stops the build with an `Invalid KV_ESTIMATOR_BASE_PATH` error.
- The prefix is compiled into the standalone server at build time. Setting `KV_ESTIMATOR_BASE_PATH` when starting a container has no effect: **changing the prefix requires rebuilding the image.** The published GHCR images are built for the root.
- With a prefix, `/kv-estimator` serves the estimator, `/kv-estimator/` redirects to it (HTTP 308, relative `Location: /kv-estimator`), and `/` returns 404.

### Build and run with Docker

```bash
# Root (default): http://localhost:3000/
docker build -t kv-cache-estimator:root .
docker run --rm -p 3000:3000 kv-cache-estimator:root

# Under /kv-estimator: http://localhost:3000/kv-estimator
docker build --build-arg KV_ESTIMATOR_BASE_PATH=/kv-estimator -t kv-cache-estimator:kv-estimator .
docker run --rm -p 3000:3000 kv-cache-estimator:kv-estimator
```

### Build and run with Podman

```bash
# Root (default): http://localhost:3000/
podman build -t kv-cache-estimator:root .
podman run --rm -p 3000:3000 localhost/kv-cache-estimator:root

# Under /kv-estimator: http://localhost:3000/kv-estimator
podman build --build-arg KV_ESTIMATOR_BASE_PATH=/kv-estimator -t kv-cache-estimator:kv-estimator .
podman run --rm -p 3000:3000 localhost/kv-cache-estimator:kv-estimator
```

A custom catalogue is mounted the same way in both modes (add `:Z` or `,relabel=private` to the mount on SELinux hosts with Podman):

```bash
docker run --rm -p 3000:3000 \
  --mount "type=bind,src=$PWD/my-catalogue.yaml,dst=/config/estimator.yaml,readonly" \
  -e KV_ESTIMATOR_CONFIG=/config/estimator.yaml \
  kv-cache-estimator:kv-estimator
```

The images still run as the non-root `node` user. To check which prefix an image was built with:

```bash
docker run --rm kv-cache-estimator:kv-estimator \
  node -p "require('/app/.next/required-server-files.json').config.basePath"
# /kv-estimator   (empty output means the root)
```

### Health checks

Under a prefix, the root no longer answers with HTTP 200, so point health checks at the prefixed path. Probe `/kv-estimator` without a trailing slash to get a direct 200 rather than a 308 redirect (or make sure the probe follows redirects). The image is based on Alpine and has no `curl`; use BusyBox `wget` or Node.js:

```bash
# Docker or Podman
docker run -d -p 3000:3000 \
  --health-cmd 'wget -q --spider http://127.0.0.1:3000/kv-estimator || exit 1' \
  --health-interval 30s \
  kv-cache-estimator:kv-estimator
```

```yaml
# Docker Compose
healthcheck:
  test: ["CMD", "node", "-e", "fetch('http://127.0.0.1:3000/kv-estimator').then(r => process.exit(r.ok ? 0 : 1), () => process.exit(1))"]
  interval: 30s
  timeout: 5s
  retries: 3

# Kubernetes
readinessProbe:
  httpGet:
    path: /kv-estimator
    port: 3000
```

At the root, use `http://127.0.0.1:3000/` and `path: /` instead. `podman run` accepts the same `--health-cmd` and `--health-interval` options.

### Traefik

Route the prefix to the Next.js server **without stripping it**: the server expects to receive `/kv-estimator/...` paths. Do not attach a `StripPrefix` middleware to this router; with one, Next.js receives paths without the prefix and returns 404.

Docker or Podman labels (Traefik Docker provider):

```yaml
services:
  kv-estimator:
    image: kv-cache-estimator:kv-estimator
    labels:
      - traefik.enable=true
      - traefik.http.routers.kv-estimator.rule=PathPrefix(`/kv-estimator`)
      - traefik.http.routers.kv-estimator.entrypoints=websecure
      - traefik.http.services.kv-estimator.loadbalancer.server.port=3000
      # No traefik.http.routers.kv-estimator.middlewares=...stripprefix...
```

File provider:

```yaml
http:
  routers:
    kv-estimator:
      rule: PathPrefix(`/kv-estimator`)
      entryPoints: [websecure]
      service: kv-estimator
      # No StripPrefix middleware.
  services:
    kv-estimator:
      loadBalancer:
        servers:
          - url: http://kv-estimator:3000
```

``PathPrefix(`/kv-estimator`)`` matches both `/kv-estimator` and `/kv-estimator/...`. To exclude unrelated paths such as `/kv-estimator-old`, use ``Path(`/kv-estimator`) || PathPrefix(`/kv-estimator/`)`` instead. Add a host rule (``Host(`example.com`) && ...``) when Traefik serves several host names.

After deploying, check that the page and its redirect stay under the prefix and that the assets load:

```bash
curl -sI https://example.com/kv-estimator/ | grep -i -E '^(HTTP|location)'   # 308, location: /kv-estimator
curl -sL https://example.com/kv-estimator/ | grep -o '/kv-estimator/_next/static/[^"]*\.css' | head -1
```

The second command must print a stylesheet path under `/kv-estimator/_next/static/`; requesting it must return HTTP 200 with `Content-Type: text/css`.

### Smoke tests

`scripts/smoke-container.sh IMAGE [BASE_PATH]` tests an image built for the given prefix (empty for the root):

```bash
bash scripts/smoke-container.sh kv-cache-estimator:root
bash scripts/smoke-container.sh kv-cache-estimator:kv-estimator /kv-estimator
CONTAINER_ENGINE=podman bash scripts/smoke-container.sh localhost/kv-cache-estimator:kv-estimator /kv-estimator
```

For each mode it checks that the image was built with the expected prefix and runs as a non-root user, that the estimator is served in French and English (`Accept-Language` and the `kv-lang` cookie), that `BASE_PATH/` stays under the prefix after any redirect, and that every CSS and JavaScript URL in the HTML is under `BASE_PATH/_next/static/` and returns HTTP 200 with a CSS or JavaScript `Content-Type`. With a prefix, it also checks that neither the page nor the assets are served without it. It then restarts the image with a mounted catalogue and checks that the catalogue is used.

When `playwright-core` is installed (`pnpm install`) and a Chromium or Google Chrome browser is available, `scripts/smoke-browser.mjs` also opens the page, checks that the stylesheet is applied, toggles the language to confirm that the JavaScript has hydrated, and reloads to confirm that the server renders the chosen language. Set `KV_SMOKE_BROWSER=require` to fail without a browser, `KV_SMOKE_BROWSER=skip` to skip the check, and `KV_SMOKE_BROWSER_PATH` to use a specific browser executable.

## Releases and image publication

The GitHub Actions workflow validates the catalogue, runs TypeScript checks and tests, builds the application, then builds and smoke-tests the production container twice: at the root and under `/kv-estimator`. The smoke tests check both the bundled catalogue and a mounted YAML override, the French and English pages, the CSS and JavaScript assets and their loading in a browser, and verify that the container runs without root privileges. Published images are built for the root.

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

The converter recognizes GQA/MHA, sliding windows (`layer_types`, `hybrid_layer_pattern`, `sliding_window_pattern`), Llama 4 chunked attention, MLA and DSA indexers (including IndexShare and pooling), cross-layer KV sharing (`kv_source_layers`), Mamba-2, Gated DeltaNet, KDA, NAS-pruned `block_configs`, and Mistral's `params.json` structure.

Parameter counts use this priority:

1. The explicit `--params` option.
2. Safetensors metadata from a `{repo, revision, config, parameterMetadata}` envelope, as stored in `config/upstream/`.
3. An estimate from architecture dimensions: attention, MLPs, experts and linear layers.
4. The repository name when a dimension-based estimate is unreliable, for example with packed tensors.

Active MoE parameters are estimates. The native format comes from `quantization_config`; SSM state uses FP32 when the configuration requires it (`mamba_ssm_cache_dtype`). The 25 upstream snapshots include source revisions. Tests check that the converter reproduces the catalogue's layer definitions for each snapshot.

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
- MLA KV = layers x (latent rank + RoPE dimension) x bytes x tokens, plus DSA indexer storage and scales where applicable. Layers that read an earlier layer's latent (cross-layer KV sharing, `kvShared`) store only their indexer keys.
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

Layer kinds are `full`, `sliding` (`window`), `mla` (`kvLoraRank`, `ropeHeadDim`, `extraBytesPerToken`, `kvShared`) and `linear` (`ssmElements`, `convElements`).

Text shown in the interface (descriptions, notes, support notes, labels, GPU names) accepts either one string for both languages or one string per language:

```yaml
notes:
  - Same text in both languages.
  - { fr: "Licence Mistral Research (usage commercial soumis à accord).", en: "Mistral Research License (commercial use requires an agreement)." }
```

The catalogue contains 65 models across Mistral, Z.ai (GLM), Moonshot (Kimi), Xiaomi (MiMo), Qwen, NVIDIA, Google (Gemma), DeepSeek, OpenAI (gpt-oss) and Meta (Llama). Entries with a verified configuration revision have upstream snapshots in `config/upstream/`, originally collected on `feat/kv-cache-estimator`. Other entries use public architecture information with a `confidence` level (`high`, `medium`, `low`); the UI marks medium and low confidence. Some active parameter counts, including Kimi K3, GLM-5.3 and MiMo-V2.6-Pro, are estimates.

## Development checks

```bash
pnpm run check-config
pnpm run typecheck
pnpm test
pnpm run build
```

Tests cover formulas, TP/PP/DP replication, offload, throughput, sizing, the 25 upstream configuration imports and a configuration matrix. GitHub Actions also builds and smoke-tests the production image.

Keep source comments, docstrings and Markdown artifacts in English. Write every new UI string and catalogue message in both French and English: `t("…", "…")` from `usePrefs()` in components, `tr(lang)` in `src/lib`, `{ fr, en }` in the configuration. Do not edit upstream configuration snapshots to translate external source data.
