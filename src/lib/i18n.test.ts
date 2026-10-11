import { describe, expect, it } from "vitest";
import path from "node:path";
import { parseConfigFile } from "./load-config";
import { collectIssues, estimate, type EstimateInput } from "./calc";
import { formatters } from "./format";
import { langFromAcceptLanguage, localize } from "./i18n";
import { resolveScenario, scenarioFromQuery } from "./scenario";

describe("i18n", async () => {
  const r = await parseConfigFile(path.join(__dirname, "../../config/estimator.yaml"));
  if (!r.ok) throw new Error(r.error);
  const source = r.config;

  it("localize resolves every { fr, en } text and keeps plain strings", () => {
    const v = { a: { fr: "Poids", en: "Weights" }, b: "BF16", c: [{ fr: "un", en: "one" }], d: 3 };
    expect(localize(v, "en")).toEqual({ a: "Weights", b: "BF16", c: ["one"], d: 3 });
    expect(localize(v, "fr")).toEqual({ a: "Poids", b: "BF16", c: ["un"], d: 3 });
  });

  it("the catalogue has English text for the configuration", () => {
    const en = JSON.stringify(localize(source, "en"));
    expect(en).not.toMatch(/"fr":/);
    const v100 = localize(source, "en").gpus.find((g) => g.id === "v100")!;
    expect(v100.weightFormats.bf16.note).toMatch(/BF16/);
    expect(v100.weightFormats.bf16.note).not.toBe(localize(source, "fr").gpus.find((g) => g.id === "v100")!.weightFormats.bf16.note);
  });

  it("messages follow the language of the input", () => {
    const issues = (lang: "fr" | "en") => {
      const cfg = localize(source, lang);
      const s = scenarioFromQuery(cfg, cfg.models, { model: "qwen3-32b", gpu: "h100" });
      const input = resolveScenario(cfg, cfg.models, { ...s, gpuCount: 3, mode: "tp-pp", tp: 3 }, lang) as EstimateInput;
      return collectIssues(input, estimate(input), cfg.weightFormats);
    };
    expect(issues("fr")[0].title).toMatch(/TP=3 impossible : 64 têtes d'attention/);
    expect(issues("en")[0].title).toMatch(/TP=3 impossible: 64 attention heads/);
  });

  it("formats numbers and units per language", () => {
    expect(formatters("fr").bytes(12.34 * 1024 ** 3)).toBe("12,3 Gio");
    expect(formatters("en").bytes(12.34 * 1024 ** 3)).toBe("12.3 GiB");
  });

  it("detects the browser language", () => {
    expect(langFromAcceptLanguage("en-US,en;q=0.9,fr;q=0.8")).toBe("en");
    expect(langFromAcceptLanguage("de-DE,fr;q=0.5")).toBe("fr");
    expect(langFromAcceptLanguage(undefined)).toBe("fr");
  });
});
