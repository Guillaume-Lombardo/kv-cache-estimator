import { describe, expect, it } from "vitest";
import path from "node:path";
import { collectIssues, contextLabel, estimate, GIB, kvSupport, parseContext, runtimeDefaults, weightSupport } from "./calc";
import type { AppConfig } from "./config-schema";
import { gpuFamilies, gpuGroupOf } from "./gpu-groups";
import { fetchHfModel, HfFetchError } from "./hf-fetch";
import { localize } from "./i18n";
import { parseConfigFile } from "./load-config";
import { estimatePerf } from "./perf";
import { maxContextFor } from "./planning";
import { buildReport, unknownIdentifiers } from "./report";
import { resolveScenario, scenarioFromQuery, scenarioToQuery, type QueryParams, type Workload } from "./scenario";
import { analyzeWorkload, instanceLoad, sizeForWorkload } from "./workload";

const load = async (): Promise<AppConfig> => {
  const r = await parseConfigFile(path.join(__dirname, "../../config/estimator.yaml"));
  if (!r.ok) throw new Error(r.error);
  return localize(r.config, "en");
};

describe("runtime overhead", async () => {
  const cfg = await load();
  const inputFor = (q: QueryParams) => resolveScenario(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, { n: "1", ...q }), "en")!;

  it("vLLM takes the activation peak and runtime memory from its own memory fraction", () => {
    const e = estimate(inputFor({ model: "qwen3-32b", gpu: "h100" }));
    const ov = e.memory.overhead;
    expect(ov.inBudget).toBe(true);
    expect(ov.totalBytes).toBeGreaterThan(GIB);
    expect(e.memory.usablePerGpuBytes).toBeCloseTo(e.memory.gpuBytes - e.memory.reservePerGpuBytes - ov.totalBytes, 0);
    // The estimate is optimistic without it: one 32k session fits either way, but less free memory remains.
    const without = estimate(inputFor({ model: "qwen3-32b", gpu: "h100", ovh: "0" }));
    expect(without.memory.freeForKvPerGpuBytes - e.memory.freeForKvPerGpuBytes).toBeCloseTo(ov.totalBytes, 0);
  });

  it("the activation peak grows with the tokens per step and shrinks with TP", () => {
    const a = estimate(inputFor({ model: "qwen3-32b", gpu: "h100", mbt: "2048" })).memory.overhead;
    const b = estimate(inputFor({ model: "qwen3-32b", gpu: "h100", mbt: "8192" })).memory.overhead;
    expect(b.activationBytes).toBeCloseTo(4 * a.activationBytes, 0);
    const tp4 = estimate(inputFor({ model: "qwen3-32b", gpu: "h100", n: "4", tp: "4", mbt: "8192" })).memory.overhead;
    expect(tp4.activationBytes).toBeLessThan(b.activationBytes / 2);
  });

  it("estimates the MLP width from parameter counts (Qwen3-32B: 25600 actual)", () => {
    const ov = estimate(inputFor({ model: "qwen3-32b", gpu: "h100" })).memory.overhead;
    expect(ov.mlpWidth).toBeGreaterThan(25600 * 0.9);
    expect(ov.mlpWidth).toBeLessThan(25600 * 1.25);
  });

  it("an entered overhead replaces the estimate", () => {
    const ov = estimate(inputFor({ model: "qwen3-32b", gpu: "h100", ovh: "3.5" })).memory.overhead;
    expect(ov.overridden).toBe(true);
    expect(ov.totalBytes).toBe(3.5 * GIB);
  });
});

describe("context length", async () => {
  const cfg = await load();
  it("labels binary and decimal lengths", () => {
    expect(contextLabel(131072)).toBe("128k");
    expect(contextLabel(200000)).toBe("200k");
    expect(contextLabel(1048576)).toBe("1M");
    expect(contextLabel(40960)).toBe("40k");
  });
  it("parses typed lengths", () => {
    expect(parseContext("128k")).toBe(131072);
    expect(parseContext("200k")).toBe(200000);
    expect(parseContext("1M")).toBe(1048576);
    expect(parseContext("1,5M")).toBe(1500000);
    expect(parseContext("40960")).toBe(40960);
    expect(parseContext("abc")).toBeNull();
  });
  it("keeps an exact context from the URL instead of rounding it to a preset", () => {
    const s = scenarioFromQuery(cfg, cfg.models, { model: "qwen3-8b", ctx: "40960" });
    expect(s.ctx).toBe(40960);
    expect(scenarioFromQuery(cfg, cfg.models, { ctx: "1" }).ctx).toBe(256);
  });
  it("tries the model limit when looking for the maximum context", () => {
    // Qwen3-8B: 40960 native tokens, between the 32k and 64k presets.
    const input = resolveScenario(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, { model: "qwen3-8b", gpu: "h100", n: "1", ctx: "40960" }), "en")!;
    expect(maxContextFor(cfg, input, 1)).toBe(input.model.maxContextExtended ?? input.model.maxContext);
  });
});

describe("prefill", async () => {
  const cfg = await load();
  it("a single prompt's prefill runs on one PP stage, not on every GPU", () => {
    const one = resolveScenario(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, { model: "qwen3-8b", gpu: "h100", n: "1" }), "en")!;
    const pp2 = resolveScenario(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, { model: "qwen3-8b", gpu: "h100", n: "2", tp: "1" }), "en")!;
    const p1 = estimatePerf(one, estimate(one), cfg.performance).prefillSeconds;
    const p2 = estimatePerf(pp2, estimate(pp2), cfg.performance).prefillSeconds;
    expect(p2).toBeCloseTo(p1, 6);
  });
});

describe("workload", async () => {
  const cfg = await load();
  const w: Workload = { inputTokens: 2000, outputTokens: 500, requestsPerSec: 2, ttftSeconds: 2, itlMs: 50 };
  const inputFor = (q: QueryParams) => resolveScenario(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, { n: "1", ...q }), "en")!;

  it("concurrency follows Little's law at the solved ITL", () => {
    const a = analyzeWorkload(cfg, inputFor({ model: "qwen3-8b", gpu: "h100", ctx: "32768" }), w);
    const s = a.single;
    expect(s.concurrency).toBeCloseTo(w.requestsPerSec * (s.ttftSeconds + (w.outputTokens * s.itlMs) / 1000), 1);
    expect(s.prefillUtil).toBeGreaterThan(0);
    expect(s.ttftSeconds).toBeCloseTo(a.model.singlePrefillSeconds / (1 - s.prefillUtil), 6);
  });

  it("splits the rate over the fewest replicas that meet every target", () => {
    // 16k: one 32k request no longer fits on one H100 next to the BF16 weights and runtime overhead.
    const a = analyzeWorkload(cfg, inputFor({ model: "qwen3-32b", gpu: "h100", ctx: "16384" }), { ...w, requestsPerSec: 4 });
    const agg = a.aggregated!;
    expect(agg.load.ok).toBe(true);
    expect(agg.gpus).toBe(agg.replicas);
    if (agg.replicas > 1) expect(instanceLoad(a.model, { ...w, requestsPerSec: 4 }, 4 / (agg.replicas - 1)).ok).toBe(false);
  });

  it("reports targets that no replica count can meet", () => {
    const input = inputFor({ model: "qwen3-32b", gpu: "l40s", n: "2", tp: "2" });
    expect(analyzeWorkload(cfg, input, { ...w, itlMs: 1 }).infeasible).toBe("itl-unreachable");
    expect(analyzeWorkload(cfg, input, { ...w, ttftSeconds: 0.001 }).infeasible).toBe("ttft-unreachable");
    expect(analyzeWorkload(cfg, inputFor({ model: "qwen3-8b", gpu: "h100", ctx: "1024" }), w).infeasible).toBe("request-exceeds-context");
  });

  it("sizes prefill and decode pools separately for P/D disaggregation", () => {
    const a = analyzeWorkload(cfg, inputFor({ model: "qwen3-32b", gpu: "h100", ctx: "16384" }), { ...w, requestsPerSec: 8 });
    const d = a.disaggregated!;
    expect(d.prefillReplicas).toBeGreaterThanOrEqual(1);
    expect(d.decodeReplicas).toBeGreaterThanOrEqual(1);
    expect(d.gpus).toBe(d.prefillReplicas + d.decodeReplicas);
    expect(d.kvTrafficBytesPerSec).toBeCloseTo(8 * a.model.promptKvBytes, 0);
    expect(d.decode.prefillUtil).toBe(0);
  });

  it("prices the deployment per hour and per million tokens", () => {
    const a = analyzeWorkload(cfg, inputFor({ model: "qwen3-8b", gpu: "h100" }), w, 2);
    const c = a.cost.aggregated!;
    expect(c.usdPerHour).toBe(2 * a.aggregated!.gpus);
    expect(c.usdPerMillionOutput).toBeCloseTo((c.usdPerHour / (w.requestsPerSec * w.outputTokens * 3600)) * 1e6, 6);
  });

  it("sizes every card for the workload, fewest GPUs or cheapest first", () => {
    const input = inputFor({ model: "qwen3-8b", gpu: "h100" });
    const rows = sizeForWorkload(cfg, { ...input, weightFormatId: input.weightFormat.id, kvFormatId: input.kvFormat.id }, w, "h100");
    expect(rows).toHaveLength(cfg.gpus.length);
    const reachable = rows.filter((r) => r.gpus !== null);
    expect(reachable.length).toBeGreaterThan(0);
    for (const r of reachable) {
      expect(r.gpus).toBe(r.replicas * r.gpuCount!);
      expect(r.itlMs!).toBeLessThanOrEqual(w.itlMs);
    }
    expect(rows.findIndex((r) => r.gpus === null)).not.toBe(0);
  });
});

describe("scenario URL and report", async () => {
  const cfg = await load();
  it("round-trips the workload, runtime settings and price", () => {
    const q = { model: "qwen3-8b", gpu: "h100", ctx: "40960", mbt: "4096", ovh: "2.5", in: "1500", out: "300", rps: "3.5", ttft: "1", itl: "40", usd: "2.4" };
    const s = scenarioFromQuery(cfg, cfg.models, q);
    const back = scenarioFromQuery(cfg, cfg.models, Object.fromEntries(new URLSearchParams(scenarioToQuery(s))));
    expect(back).toEqual(s);
    expect(s.workload).toEqual({ inputTokens: 1500, outputTokens: 300, requestsPerSec: 3.5, ttftSeconds: 1, itlMs: 40 });
  });
  it("gives every issue a stable code", () => {
    const input = resolveScenario(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, { model: "kimi-k2-instruct", gpu: "l40s", n: "8", tp: "8" }), "en")!;
    const issues = collectIssues(input, estimate(input), cfg.weightFormats);
    expect(issues.length).toBeGreaterThan(0);
    for (const i of issues) expect(i.code).toMatch(/^[a-z0-9-]+$/);
  });
  it("builds a JSON report and rejects unknown identifiers", () => {
    const s = scenarioFromQuery(cfg, cfg.models, { model: "qwen3-8b", gpu: "h100" });
    const r = buildReport(cfg, cfg.models, s, "en")!;
    expect(r.results.sessions).toBeGreaterThan(0);
    expect(r.results.overheadPerGpu.totalGiB).toBeGreaterThan(0);
    expect(r.workload.aggregated?.gpus).toBeGreaterThan(0);
    expect(r.unmodelled.length).toBeGreaterThan(0);
    expect(JSON.parse(JSON.stringify(r))).toBeTruthy();
    expect(unknownIdentifiers(cfg, cfg.models, { model: "nope", gpu: "h100", eng: "x" })).toEqual(["model=nope", "eng=x"]);
  });
});

describe("GPU picker groups", async () => {
  const cfg = await load();
  it("groups variants under one family, in catalogue order", () => {
    const families = gpuFamilies(cfg.gpus);
    const names = families.map((f) => f.name);
    expect(new Set(names).size).toBe(names.length);
    expect(families.find((f) => f.name === "B200")!.variants.map((g) => g.id)).toEqual(["b200", "gb200"]);
    expect(families.find((f) => f.name === "GH200")!.variants.map((g) => g.id)).toEqual(["gh200", "gh200-144"]);
    expect(gpuGroupOf(cfg.gpus.find((g) => g.id === "l40s")!)).toBe("pcie");
    // Every card belongs to exactly one family.
    expect(families.reduce((a, f) => a + f.variants.length, 0)).toBe(cfg.gpus.length);
  });
});

describe("Hugging Face download", () => {
  const response = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });
  it("returns the converter envelope with the commit and parameter count", async () => {
    const calls: string[] = [];
    const fetchImpl = (async (url: string) => {
      calls.push(url);
      if (url.includes("/api/models/")) return response(200, { sha: "abc123", safetensors: { parameters: { BF16: 8e9 }, total: 8e9 } });
      if (url.endsWith("/config.json")) return response(200, { hidden_size: 4096 });
      return response(404, {});
    }) as typeof fetch;
    const env = await fetchHfModel("Qwen/Qwen3-8B", { fetchImpl });
    expect(env).toEqual({ repo: "Qwen/Qwen3-8B", revision: "abc123", config: { hidden_size: 4096 }, parameterMetadata: { parameters: { BF16: 8e9 }, total: 8e9 } });
    expect(calls.every((u) => u.startsWith("https://huggingface.co/"))).toBe(true);
    expect(calls[1]).toContain("/resolve/abc123/config.json");
  });
  it("falls back to params.json and maps gated repositories to 403", async () => {
    const fetchImpl = (async (url: string) =>
      url.includes("/api/") ? response(200, { sha: "s" }) : url.endsWith("params.json") ? response(200, { dim: 1 }) : response(404, {})) as typeof fetch;
    expect((await fetchHfModel("mistralai/X", { fetchImpl })).config).toEqual({ dim: 1 });
    const gated = (async () => response(401, {})) as unknown as typeof fetch;
    await expect(fetchHfModel("meta-llama/Llama-3.1-8B", { fetchImpl: gated })).rejects.toMatchObject({ status: 403 });
  });
  it("refuses anything but org/name before any request", async () => {
    const fetchImpl = (async () => {
      throw new Error("no request expected");
    }) as typeof fetch;
    for (const repo of ["../etc/passwd", "a/b/c", "https://evil.example/x", "a/..", ""])
      await expect(fetchHfModel(repo, { fetchImpl })).rejects.toBeInstanceOf(HfFetchError);
  });
});

describe("engine catalogue", async () => {
  const cfg = await load();
  const inputFor = (q: QueryParams) => resolveScenario(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, { n: "1", ...q }), "en")!;
  const gpu = (id: string) => cfg.gpus.find((g) => g.id === id)!;
  const engine = (id: string) => cfg.engines.find((e) => e.id === id)!;

  it("vLLM picks its scheduler defaults by GPU memory, A100 excluded", () => {
    const vllm = engine("vllm");
    expect(runtimeDefaults(vllm, gpu("b200"))).toMatchObject({ maxNumBatchedTokens: 16384, maxNumSeqs: 1024 });
    expect(runtimeDefaults(vllm, gpu("h100"))).toMatchObject({ maxNumBatchedTokens: 8192, maxNumSeqs: 1024 });
    expect(runtimeDefaults(vllm, gpu("a100-80"))).toMatchObject({ maxNumBatchedTokens: 2048, maxNumSeqs: 256 });
    expect(runtimeDefaults(vllm, gpu("l4"))).toMatchObject({ maxNumBatchedTokens: 2048, maxNumSeqs: 256 });
  });

  it("SGLang keeps activations outside its static fraction and warns when the reserve is too small", () => {
    const ok = estimate(inputFor({ model: "qwen3-32b", gpu: "h100", eng: "sglang" }));
    expect(ok.memory.overhead.inBudget).toBe(false);
    expect(ok.memory.usablePerGpuBytes).toBeCloseTo(ok.memory.gpuBytes - ok.memory.reservePerGpuBytes, 0);
    const small = inputFor({ model: "qwen3-32b", gpu: "h100", eng: "sglang", r: "1" });
    expect(collectIssues(small, estimate(small), cfg.weightFormats).map((i) => i.code)).toContain("reserve-below-overhead");
  });

  it("writes an SGLang launch command with its own options", () => {
    const r = buildReport(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, { model: "qwen3-30b-a3b-2507", gpu: "h100", eng: "sglang", n: "4", mode: "dp-ep", kv: "fp8" }), "en")!;
    expect(r.command).toContain("python3 -m sglang.launch_server --model-path");
    expect(r.command).toContain("--tp-size 4 --attn-dp-size 4 --ep-size 4");
    expect(r.command).toContain("--kv-cache-dtype fp8_e4m3");
    expect(r.command).toContain("--context-length");
    expect(r.command).toContain("--mem-fraction-static 0.90");
  });

  it("applies vendor-specific engine restrictions (no AWQ/GPTQ on ROCm in vLLM)", () => {
    const mi = gpu("mi300x");
    expect(weightSupport(mi, engine("vllm"), "int4").level).toBe("unsupported");
    expect(weightSupport(mi, engine("sglang"), "int4").level).toBe("limited");
    expect(weightSupport(gpu("h100"), engine("vllm"), "int4").level).toBe("native");
    expect(kvSupport(mi, engine("vllm"), "nvfp4").level).toBe("unsupported");
  });

  it("names the AMD fabric in interconnect messages", () => {
    const input = inputFor({ model: "qwen3-32b", gpu: "mi300x", n: "16", tp: "16" });
    const titles = collectIssues(input, estimate(input), cfg.weightFormats).map((i) => i.title).join(" ");
    expect(titles).toContain("Infinity Fabric");
    expect(titles).not.toContain("NVLink");
  });
});

describe("workload validity", async () => {
  const cfg = await load();
  const inputFor = (q: QueryParams) => resolveScenario(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, { n: "1", ...q }), "en")!;

  it("rejects a load whose concurrency has not settled instead of returning an early iterate", () => {
    // Qwen3-0.6B on one B200: the fixed point lies beyond the KV capacity; early iterates sit far below it.
    const w: Workload = { inputTokens: 2000, outputTokens: 2000, requestsPerSec: 8.1, ttftSeconds: 100, itlMs: 1000 };
    const a = analyzeWorkload(cfg, inputFor({ model: "qwen3-0.6b", gpu: "b200" }), w);
    expect(a.single.ok).toBe(false);
    expect(a.single.over.memory).toBe(true);
    expect(a.aggregated!.replicas).toBeGreaterThan(1);
  });

  it("does not plan deployments for a configuration with blocking errors", () => {
    const w: Workload = { inputTokens: 2000, outputTokens: 500, requestsPerSec: 1, ttftSeconds: 2, itlMs: 50 };
    // 32 attention heads cannot be split over TP 3.
    const a = analyzeWorkload(cfg, inputFor({ model: "qwen3-8b", gpu: "h100", n: "3", tp: "3" }), w);
    expect(a.infeasible).toBe("invalid-config");
    expect(a.aggregated).toBeNull();
    expect(a.disaggregated).toBeNull();
    const r = buildReport(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, { model: "qwen3-8b", gpu: "h100", n: "3", tp: "3" }), "en")!;
    expect(r.workload.singleInstance.meetsTargets).toBe(false);
  });

  it("blocks SGLang configurations whose reserve cannot hold the activations", () => {
    const input = inputFor({ model: "qwen3-8b", gpu: "h100", eng: "sglang", r: "0" });
    const issue = collectIssues(input, estimate(input), cfg.weightFormats).find((i) => i.code === "reserve-below-overhead");
    expect(issue?.severity).toBe("error");
    expect(buildReport(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, { model: "qwen3-8b", gpu: "h100", eng: "sglang", n: "1", r: "0" }), "en")!.results.valid).toBe(false);
  });
});
