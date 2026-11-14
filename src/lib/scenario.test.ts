import path from "node:path";
import { describe, expect, it } from "vitest";
import { GET } from "../app/api/estimate/route";
import { MAX_CONTEXT } from "./calc";
import { localize } from "./i18n";
import { parseConfigFile } from "./load-config";
import { buildReport, unknownIdentifiers } from "./report";
import { adjustmentMessage, scenarioFromQuery, scenarioToQuery, type QueryParams, type ScenarioAdjustment } from "./scenario";

describe("link normalization", async () => {
  const result = await parseConfigFile(path.join(__dirname, "../../config/estimator.yaml"));
  if (!result.ok) throw new Error(result.error);
  const cfg = localize(result.config, "en");
  const parse = (q: QueryParams = {}) => scenarioFromQuery(cfg, cfg.models, q);

  it("reports unknown identifiers and context clamping with received and retained values", () => {
    const { scenario, adjustments } = parse({ model: "doesnotexist", gpu: "zzz", ctx: "99999999", eng: "nope", w: "nope", kv: "nope" });
    expect(adjustments).toEqual([
      { parameter: "model", received: "doesnotexist", retained: scenario.modelId, reason: "unknown-identifier" },
      { parameter: "gpu", received: "zzz", retained: scenario.gpuId, reason: "unknown-identifier" },
      { parameter: "eng", received: "nope", retained: scenario.engineId, reason: "unknown-identifier" },
      { parameter: "ctx", received: "99999999", retained: String(MAX_CONTEXT), reason: "context-out-of-bounds" },
      { parameter: "w", received: "nope", retained: scenario.wf, reason: "unknown-identifier" },
      { parameter: "kv", received: "nope", retained: scenario.kf, reason: "unknown-identifier" },
    ]);
    expect(scenario.ctx).toBe(MAX_CONTEXT);
    expect(scenario).not.toHaveProperty("adjustments");
  });

  it.each<[QueryParams, string, ScenarioAdjustment["reason"]]>([
    [{ gpu: "h100", n: "12" }, "n", "gpu-count-not-offered"],
    [{ n: "3", tp: "2" }, "tp", "invalid-tp"],
    [{ tp: "garbage" }, "tp", "invalid-tp"],
    [{ ctx: "1" }, "ctx", "context-out-of-bounds"],
    [{ ctx: "NaN" }, "ctx", "context-out-of-bounds"],
    [{ gpu: "v100", w: "bf16" }, "w", "unsupported-format"],
    [{ gpu: "v100", kv: "bf16" }, "kv", "unsupported-format"],
    [{ model: "qwen3-32b", mode: "dp-ep" }, "mode", "dp-unavailable"],
    [{ model: "qwen3-32b", p: "dp-attention" }, "p", "dp-unavailable"],
    [{ mode: "bad" }, "mode", "invalid-value"],
    [{ p: "bad" }, "p", "invalid-value"],
    [{ start: "bad" }, "start", "invalid-value"],
    [{ tab: "bad" }, "tab", "invalid-value"],
    [{ obj: "bad" }, "obj", "invalid-value"],
  ])("reports %j", (q, parameter, reason) => {
    const adjustment = parse(q).adjustments.find((a) => a.parameter === parameter);
    expect(adjustment).toMatchObject({ parameter, received: q[parameter], reason });
    expect(adjustment?.retained).not.toBe(q[parameter]);
  });

  it.each(["r", "mem", "off", "mbt", "ovh", "in", "out", "rps", "ttft", "itl", "usd", "ns"])("reports an invalid %s", (parameter) => {
    expect(parse({ [parameter]: "-1" }).adjustments).toContainEqual(expect.objectContaining({ parameter, received: "-1", reason: "invalid-value" }));
  });

  it("records removal and rounding without flagging equivalent numeric spellings", () => {
    expect(parse({ mem: "undefined" }).adjustments).toContainEqual({ parameter: "mem", received: "undefined", retained: null, reason: "invalid-value" });
    expect(parse({ in: "1.2", ns: "1.2" }).adjustments).toEqual([
      { parameter: "in", received: "1.2", retained: "1", reason: "invalid-value" },
      { parameter: "ns", received: "1.2", retained: "2", reason: "invalid-value" },
    ]);
    expect(parse({ n: "02", tp: "2.0", r: "10.0", off: "0", ovh: "0" }).adjustments).toEqual([]);
  });

  it("leaves defaults and valid URLs without corrections", () => {
    expect(parse().adjustments).toEqual([]);
    const valid = parse({ model: "qwen3-32b", gpu: "h100", n: "2", tp: "2", w: "bf16", kv: "bf16", ctx: "32768", tab: "compare" });
    expect(valid.adjustments).toEqual([]);
    expect(parse(Object.fromEntries(new URLSearchParams(scenarioToQuery(valid.scenario))))).toEqual(valid);
  });

  it("reports DP mode disabled by the engine", () => {
    const noDp = { ...cfg, engines: cfg.engines.map((e) => ({ ...e, features: { ...e.features, dpAttention: false as const } })) };
    const { adjustments } = scenarioFromQuery(noDp, cfg.models, { model: "qwen3-30b-a3b-2507", mode: "dp-ep" });
    expect(adjustments).toContainEqual({ parameter: "mode", received: "dp-ep", retained: "tp-pp", reason: "dp-unavailable" });
  });

  it("resolves an imported identifier supplied by the caller", () => {
    const imported = { ...cfg.models[0], id: "my-import", name: "My import" };
    const parsed = scenarioFromQuery(cfg, [...cfg.models, imported], { model: imported.id });
    expect(parsed.scenario.modelId).toBe(imported.id);
    expect(parsed.adjustments).toEqual([]);
  });

  it.each(["detail", "perf", "sizing", "compare", "import", "workload"])("round-trips tab %s and session sizing inputs", (tab) => {
    const parsed = parse({ tab, obj: "sessions", ns: "80", start: "load" });
    expect(parsed.scenario.tab).toBe(tab === "workload" ? "sizing" : tab);
    expect(parsed.scenario.sizingObjective).toBe("sessions");
    expect(parsed.scenario.targetSessions).toBe(80);
    expect(parsed.adjustments).toEqual([]);
    expect(parse(Object.fromEntries(new URLSearchParams(scenarioToQuery(parsed.scenario))))).toEqual(parsed);
  });

  it("omits unspecified default view values and preserves an explicitly selected default tab", () => {
    for (const q of [{}, { start: "load" }]) {
      const params = new URLSearchParams(scenarioToQuery(parse(q).scenario));
      for (const key of ["tab", "obj", "ns"]) expect(params.has(key)).toBe(false);
    }
    for (const q of [{ tab: "detail" }, { start: "load", tab: "detail" }, { start: "load", tab: "sizing" }]) {
      const parsed = parse(q);
      expect(new URLSearchParams(scenarioToQuery(parsed.scenario)).get("tab")).toBe(q.tab);
      expect(parse(Object.fromEntries(new URLSearchParams(scenarioToQuery(parsed.scenario))))).toEqual(parsed);
    }
  });

  it("localizes link corrections", () => {
    const a = parse({ model: "doesnotexist" }).adjustments[0];
    expect(adjustmentMessage(a, "fr", cfg, cfg.models)).toContain("Lien : modèle « doesnotexist » — identifiant inconnu → Qwen3-32B");
    expect(adjustmentMessage(a, "en", cfg, cfg.models)).toContain("Link: model “doesnotexist” — unknown identifier → Qwen3-32B");
  });

  it("adds corrections to reports and continues rejecting unknown identifiers", async () => {
    const { scenario, adjustments } = parse({ model: "qwen3-32b", gpu: "h100", n: "3", tp: "2" });
    expect(buildReport(cfg, cfg.models, scenario, "en", adjustments)?.adjustments).toEqual(adjustments);
    expect(buildReport(cfg, cfg.models, scenario, "en")?.adjustments).toEqual([]);
    const response = await GET(new Request("http://localhost/api/estimate?model=qwen3-32b&gpu=h100&n=3&tp=2"));
    expect(response.status).toBe(200);
    expect((await response.json()).adjustments).toEqual(adjustments);
    expect((await GET(new Request("http://localhost/api/estimate?model=nope"))).status).toBe(400);
    expect(unknownIdentifiers(cfg, cfg.models, { w: "nope", kv: "nope" })).toEqual(["w=nope", "kv=nope"]);
  });
});
