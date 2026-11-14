import path from "node:path";
import { describe, expect, it } from "vitest";
import { bestIndices, compareScenarios, instanceCost, valuesDiffer, workloadsDiffer } from "./compare";
import { localize } from "./i18n";
import { parseConfigFile } from "./load-config";
import { resolveScenario, scenarioFromQuery, type QueryParams } from "./scenario";
import { analyzeWorkload, instanceLoad } from "./workload";

describe("comparison helpers", () => {
  it("ranks in both directions, keeps ties and excludes invalid or non-finite values", () => {
    const values = [8, 2, 8, null, Infinity, NaN, 100];
    const eligible = [true, true, true, true, true, true, false];
    expect(bestIndices(values, "higher", eligible)).toEqual([0, 2]);
    expect(bestIndices(values, "lower", eligible)).toEqual([1]);
    expect(bestIndices([null, Infinity], "lower")).toEqual([]);
    expect(bestIndices([0, 1], "lower")).toEqual([0]);
  });

  it("compares raw values, including unavailable columns and values rounded alike", () => {
    expect(valuesDiffer([])).toBe(false);
    expect(valuesDiffer(["H100", "H100"])).toBe(false);
    expect(valuesDiffer([null, null])).toBe(false);
    expect(valuesDiffer([null, 0])).toBe(true);
    expect(valuesDiffer([1.001, 1.002])).toBe(true);
  });

  it("prices all GPUs and a million output tokens at full load", () => {
    expect(instanceCost(2, 3, 1000)).toEqual({ usdPerHour: 6, usdPerMillionOutput: 6e6 / 3600000 });
    expect(instanceCost(2, undefined, 1000)).toEqual({ usdPerHour: null, usdPerMillionOutput: null });
    expect(instanceCost(2, 3, 0)).toEqual({ usdPerHour: 6, usdPerMillionOutput: null });
    expect(instanceCost(2, 3, undefined).usdPerMillionOutput).toBeNull();
  });
});

describe("scenario comparison", async () => {
  const parsed = await parseConfigFile(path.join(__dirname, "../../config/estimator.yaml"));
  if (!parsed.ok) throw new Error(parsed.error);
  const cfg = localize(parsed.config, "en");
  const scenario = (q: QueryParams = {}) => scenarioFromQuery(cfg, cfg.models, { model: "qwen3-32b", gpu: "h100", n: "2", tp: "2", ctx: "32768", ...q });
  const compare = (...scenarios: ReturnType<typeof scenario>[]) => compareScenarios(cfg, cfg.models, scenarios, "en");
  const value = (c: ReturnType<typeof compare>, id: string, i = 0) => c.rows.find((r) => r.id === id)!.values[i];

  it("marks changed configuration rows and retains equal rows for filtering", () => {
    const c = compare(scenario(), scenario({ gpu: "h200" }), scenario({ gpu: "mi300x" }));
    expect(c.rows.find((r) => r.id === "cards")).toMatchObject({ configuration: true, different: true });
    expect(c.rows.find((r) => r.id === "model")!.different).toBe(false);
    expect(c.rows.find((r) => r.id === "context")!.different).toBe(false);
    expect(c.rows.filter((r) => r.different).map((r) => r.id)).toContain("kv-available");
    expect(c.rows.find((r) => r.id === "kv-available")!.best).toEqual([2]);
    expect(c.workloadsDiffer).toBe(false);
  });

  it("detects every saved workload field independently", () => {
    const a = scenario();
    for (const key of ["inputTokens", "outputTokens", "requestsPerSec", "ttftSeconds", "itlMs"] as const) {
      const b = { ...a, workload: { ...a.workload, [key]: a.workload[key] + 1 } };
      expect(workloadsDiffer([a.workload, b.workload])).toBe(true);
      expect(compare(a, b).workloadsDiffer).toBe(true);
    }
  });

  it("uses the scenario price first, otherwise the catalogue, and full-load throughput", () => {
    const c = compare(scenario({ usd: "3" }), scenario());
    expect(value(c, "hourly-cost")).toBe(6);
    expect(value(c, "hourly-cost", 1)).toBe(2 * c.columns[1].input!.gpu.priceUsdPerHour!);
    expect(value(c, "token-cost")).toBeCloseTo(6e6 / (c.columns[0].perf!.full!.tokPerSecTotal * 3600));
    const noPrices = { ...cfg, gpus: cfg.gpus.map((g) => ({ ...g, priceUsdPerHour: undefined })) };
    const missing = compareScenarios(noPrices, cfg.models, [scenario()], "en");
    expect(value(missing, "hourly-cost")).toBeNull();
    expect(value(missing, "load-cost")).toBeNull();
    expect(missing.rows.find((r) => r.id === "token-cost")!.best).toEqual([]);
  });

  it("reuses workload analysis for met targets, replicas, latencies and total cost", () => {
    const s = scenario({ rps: "0.1", usd: "3", ttft: "10", itl: "200" });
    const c = compare(s);
    const a = analyzeWorkload(cfg, resolveScenario(cfg, cfg.models, s, "en")!, s.workload, 3);
    expect(value(c, "slo")).toBe(true);
    expect(value(c, "ttft")).toBe(a.single.ttftSeconds);
    expect(value(c, "itl")).toBe(a.single.itlMs);
    expect(value(c, "replicas")).toBe(a.aggregated!.replicas);
    expect(value(c, "load-gpus")).toBe(a.aggregated!.gpus);
    expect(value(c, "load-cost")).toBe(a.cost.aggregated!.usdPerHour);
  });

  it("shows an overloaded instance as failing while using enough replicas for the saved rate", () => {
    const s = scenario({ rps: "40", ttft: "10", itl: "200", usd: "3" });
    const c = compare(s);
    const a = c.columns[0].workload!;
    const replicas = value(c, "replicas") as number;
    expect(value(c, "slo")).toBe(false);
    expect(replicas).toBeGreaterThan(1);
    expect(instanceLoad(a.model, s.workload, s.workload.requestsPerSec / replicas).ok).toBe(true);
    expect(instanceLoad(a.model, s.workload, s.workload.requestsPerSec / (replicas - 1)).ok).toBe(false);
    expect(value(c, "load-cost")).toBe(replicas * 6);
  });

  it("does not claim targets or replicas for impossible targets, invalid context or invalid configurations", () => {
    const cases = [scenario({ ttft: "0.000001" }), scenario({ itl: "0.000001" }), scenario({ ctx: "1024" }), scenario({ n: "3", tp: "3" })];
    for (const s of cases) {
      const c = compare(s);
      expect(value(c, "slo")).toBe(false);
      expect(value(c, "replicas")).toBeNull();
      expect(value(c, "load-cost")).toBeNull();
    }
    const c = compare(cases[3], scenario());
    expect(c.columns[0].valid).toBe(false);
    for (const r of c.rows) expect(r.best).not.toContain(0);
    const missing = compare({ ...scenario(), modelId: "removed-model" }, scenario());
    expect(missing.columns[0].input).toBeNull();
    for (const r of missing.rows) {
      expect(r.values[0]).toBeNull();
      expect(r.best).not.toContain(0);
    }
  });
});
