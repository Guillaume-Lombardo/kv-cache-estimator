# Calculation model and engine profiles

What the estimator computes, the formulas behind each figure, and how engine profiles change them. Back to the [README](../README.md).

## Inputs

| Input | Control |
|---|---|
| Model | Search by name, family or architecture keyword |
| Context | Any length (`200k`, `1.5M`, `131072`), doubling presets from 8k to 1M, and the model's native and extended limits; extended and unsupported contexts are marked |
| GPU | A family, then a variant. NVIDIA data center: V100, A100 (SXM 40 / 80 GB, PCIe 80 GB), H100 (SXM, NVL), H200 (SXM, NVL), GH200 (96 / 144 GB), B200 (HGX, GB200 NVL72), B300 (HGX, GB300 NVL72). AMD data center: MI300X, MI325X, MI355X. PCIe: L4, A10 / A10G, L40S, RTX PRO 6000, RTX 4090, RTX 5090 |
| GPU count | 1 to 8, then 16, 32, 64; 72 for NVL72 domains |
| Parallelism | TP/PP with selectable TP degree (PP = GPU count / TP), or DP-attention + EP for MoE |
| Engine | vLLM / llm-d, SGLang or ZML / LLMD; both GPU and engine must support the format |
| Formats | BF16, FP16, FP8, INT8, INT4, MXFP4, NVFP4 weights; BF16, FP16, FP8, NVFP4 KV |
| Reserve | Percentage of each GPU left outside the engine: `1 − --gpu-memory-utilization` (vLLM) or `1 − --mem-fraction-static` (SGLang) |
| Advanced | Visible memory per GPU from `nvidia-smi` / `amd-smi`, tokens per scheduling step, measured activation and runtime memory, KV offload to CPU RAM |
| Workload | Average prompt and output tokens, request rate, TTFT and ITL targets, price per GPU-hour (Load and SLOs tab) |

The URL stores the scenario settings for sharing. Models imported only into browser storage must also be available in the receiving browser.

## Results and limitations

The main indicators show model weights (total and per GPU), logical and replicated KV per request, model plus one KV cache, free KV memory in GiB, and concurrent sessions. Other panels provide:

- Per-layer KV details, sessions at each context step and an indicative `vllm serve` command.
- Estimated per-session and aggregate throughput, prefill time, memory/compute limits and CPU offload.
- Load and SLOs: concurrency needed against capacity at the average request length, TTFT, ITL, replicas, aggregated versus prefill/decode-disaggregated serving with the KV traffic between pools, cost per hour and per million tokens, and the cheapest instance shape on every card.
- Inverse sizing: minimum GPU count and parallelism for a target session count, plus the maximum context on the current configuration. Both sizing tables filter by card group.
- Up to six pinned scenarios for comparison, and a JSON export.

Unsupported options are disabled with a reason on hover. Limited or unverified options remain selectable with warnings. Format choices adjust when switching model, GPU or engine, with a notice explaining the fallback. Warnings are ordered by severity.

Any blocking error marks the configuration invalid. The UI retains theoretical memory figures but removes success indicators. Invalid configurations are excluded from sizing recommendations and comparison winners.

## Calculation model

- Weights use parameter counts and effective bits, including quantization scales (for example INT4 g128 = 4.25 bits and NVFP4 = 4.5 bits). Embeddings, lm_head and designated unquantized parameters stay at 16 bits. `nativeBitsPerParam` handles specific native checkpoints, such as Kimi K2.5 INT4 g32 at 4.5 bits. `nativeNonExpertBitsPerParam` keeps the other quantized weights at their own precision when a native checkpoint quantizes only the experts further, such as DeepSeek V4 (FP4 experts, FP8 elsewhere); DP-attention replicates them at that precision.
- GQA/MHA KV = layers x KV heads x (K dimension + V dimension) x bytes x tokens.
- Sliding-window KV is capped only when the engine both applies the window and frees expired blocks.
- MLA KV = layers x (latent rank + RoPE dimension) x bytes x tokens, plus DSA indexer storage and scales where applicable. Layers that read an earlier layer's latent (cross-layer KV sharing, `kvShared`) store only their indexer keys.
- Linear/SSM cache is recurrent state plus convolution state, constant per request. `ssmStateBytes` can require FP32.
- Compressed attention (DeepSeek V4 CSA/HCA, `compressed` layers) stores one shared K = V vector per `compressRatio` tokens, plus the most recent `window` tokens uncompressed and the sparse indexer keys per compressed entry (`extraBytesPerEntry`). TP replicates it like an MLA latent. The compressors' FP32 state is not counted.
- TP shards KV heads and replicates them when TP exceeds their count. Each TP rank stores a full MLA latent.
- PP uses whole layers and sizes for the most heavily loaded stage: ceil(layers / PP), selecting the most expensive cache layers because exact ordering is unknown. Input/output embeddings are included; tied embeddings are copied at both ends. PP greater than the decoder layer count is invalid.
- DP-attention + EP shards only routed experts (`expertParamsB`). All non-expert tensors remain on every GPU at their respective precisions. Unknown expert parameter counts block capacity validation. Each request resides on one GPU.
- Usable memory per GPU = visible memory − reserve − runtime overhead when the engine budget includes it (vLLM). Runtime overhead = max(activation peak, sampler logits) + fixed runtime memory + CUDA graphs. The activation peak covers one scheduling step of `maxNumBatchedTokens` tokens: the gated MLP's gate/up and activated outputs (3 × width × 2 bytes per token, sharded by TP) plus a few hidden-size buffers. The MLP width comes from the parameters per decoder layer (active experts for MoE). Sampler logits are `maxNumSeqs` × vocabulary × 4 bytes. A measured value from the engine's startup log can replace the estimate.
- Sessions = floor((usable memory on the limiting GPU - its weights) / its KV per request). DP-attention sums the per-GPU session counts. Every session uses its full selected context; prefix reuse is ignored.
- Total physical free KV memory = total usable memory - physical weights, including replicas. A separate conservative budget multiplies the limiting GPU's free memory by GPU count. This budget drives session capacity and is shown when it differs from physical free memory.
- Model plus one KV reports physical usage, including replication. Its fit verdict uses the limiting GPU; a logical sum is also shown when different.
- Recommendations exclude blocking format, parallelism, context and capacity errors. A visible-memory override applies only to its original GPU type.
- Launch commands use the native repository or a format-specific `checkpoints` mapping. Missing checkpoints are explicitly marked as required input.
- CPU offload parks queued, preempted or reusable sessions. It does not increase simultaneous decoding capacity. Reload time = per-GPU KV / CPU-to-GPU bandwidth.
- Decode step time = max(bytes read / effective bandwidth, FLOPs / effective compute throughput) + collective latency. MoE reads account for experts touched by the batch. PP traverses stages. Prefill uses FLOPs including causal attention; one prompt runs on the GPUs of one PP stage. See `performance` in the catalogue for efficiency assumptions.
- Workload (first order, no queueing): each request holds input + output/2 KV tokens on average, and concurrency is capped by that capacity and `maxNumSeqs`. Prefills take a share u = rate × prefill FLOPs / instance compute; compute-bound steps slow down by 1/(1 − u). TTFT = single-prompt prefill / (1 − u), ITL = step time at the solved concurrency, concurrency = rate × (TTFT + output × ITL) by Little's law. Replicas share the rate evenly. In prefill/decode disaggregation, the prefill pool runs prompts and sends their KV to the decode pool over `performance.networkGBsPerGpu`; the decode pool runs without prefill interference. Costs use `priceUsdPerHour` from the catalogue (indicative on-demand prices, October 2026) or the price entered for the selected card.
- Not modelled: prefix-cache reuse, speculative decoding, LoRA adapters, vision/audio encoder activations, DeepSeek V4 compressor state, allocator fragmentation beyond the reserve, queueing before prefill, and network latency between prefill and decode pools. The report and the UI list these items.

These are capacity and first-order performance estimates, not deployment benchmarks. Validate the chosen model, backend and hardware combination before relying on the result for production sizing.

## Engine profiles

Each profile declares weight and KV format support (`native`, `limited`, `unverified`, `unsupported`). An unverified format remains selectable but carries a warning.

`memoryFormat` describes a conversion performed at load time, such as an FP8 file dequantized into BF16 GPU memory. Memory estimates use the loaded format. When the loaded precision is unverified, the estimate assumes the file format and also reports the BF16 alternative.

Capabilities use three states: `true`, `false`, or `unverified`. They cover KV paging, pool preallocation, prefix caching, continuous batching, sliding-window attention, block reclamation, TP, PP, DP-attention and offload. Paging and pool preallocation can coexist.

Sliding-window memory savings require both `slidingWindowAttention: true` and `slidingWindowFreesBlocks: true`. Missing attention support is a blocking error because model outputs would be incorrect. Unverified block reclamation means allocating the full context.

`vendorFormats` restricts formats for one GPU vendor within an engine, for example AWQ/GPTQ, which vLLM does not support on ROCm.

`memory` describes how the engine sizes its KV pool. With `overheadInBudget: true` (vLLM), the memory fraction also pays for the profiled activation peak, non-PyTorch memory and CUDA graphs; with `false` (SGLang), `--mem-fraction-static` covers weights and KV only, and the overhead must fit in the reserve. `runtime` gives the scheduler defaults, optionally by GPU memory (`byMemory`); vLLM uses 16384 tokens per step and 1024 sequences from 160 GiB, 8192 / 1024 from 70 GiB except A100, and 2048 / 256 otherwise. `flags` provides the launch command and options for the indicative command line.

### SGLang

The profile follows SGLang 0.5.22: server arguments, quantization documentation and `memory_hook.py` for memory accounting. RadixAttention, the hybrid sliding-window pool, pipeline parallelism, DP-attention with expert parallelism (`--attn-dp-size`, `--ep-size`) and HiCache offloading are supported. The chunked-prefill size follows GPU memory (2048 to 16384 tokens). By default SGLang reserves at least 10 GiB outside its static fraction on GPUs above 60 GiB; a reserve smaller than the estimated activations raises a warning. The profile has medium confidence: it is sourced from documentation and code, not measured.

### ZML / LLMD

The profile distinguishes ZML's LLMD server from llm-d, the vLLM orchestrator.

Server capabilities (PagedAttention, continuous batching, tensor parallelism and prefix caching) come from the [LLMD documentation](https://zml.ai/llmd/). Format, KV precision and sliding-window observations refer to the CUDA backend of [ZML at revision 3500e88](https://github.com/zml/zml/tree/3500e88cf84e0e98d37b0f6048e6af9d33bc6108), dated October 9, 2026. They do not establish that LLMD exposes every library capability:

- FP8, MXFP4 and NVFP4 weights exist in the library and remain quantized in memory; LLMD loading support is unverified.
- INT4 and INT8 are TODOs in `zml/nn.zig`.
- FP8 KV is rejected by FA2/FA3 and disabled in the Triton wrapper.
- Kernels apply sliding-window attention, but the library does not reclaim pages outside the window, so the profile applies no VRAM savings.

Pool preallocation, PP, DP-attention and offload remain unverified in this profile.
