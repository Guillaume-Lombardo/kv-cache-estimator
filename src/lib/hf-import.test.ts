import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { convertHfConfig, modelToYaml } from "./hf-import";
import { parseConfigFile } from "./load-config";
import { modelSchema } from "./config-schema";
import { parse } from "yaml";

const upstream = path.join(__dirname, "../../config/upstream");
const files = readdirSync(upstream).filter((f) => f.endsWith(".json"));

describe("convertHfConfig", async () => {
  const r = await parseConfigFile(path.join(__dirname, "../../config/estimator.yaml"));
  if (!r.ok) throw new Error(r.error);
  const catalog = r.config.models;

  it.each(files)("%s : couches identiques au catalogue vérifié", (file) => {
    const json = JSON.parse(readFileSync(path.join(upstream, file), "utf8"));
    const { model } = convertHfConfig(json);
    expect(() => modelSchema.parse(model)).not.toThrow();
    const ref = catalog.find((m) => (json.revision && m.revision === json.revision) || (json.repo && m.source === json.repo));
    if (!ref) return; // Mistral params.json: no revision
    expect(modelSchema.parse(model).layers).toEqual(ref.layers);
    expect(model.hiddenSize).toBe(ref.hiddenSize);
    expect(model.attentionHeads).toBe(ref.attentionHeads);
    expect(model.nativeWeightFormat).toBe(ref.nativeWeightFormat);
    expect(model.paramsB).toBeGreaterThan(ref.paramsB * 0.95);
    expect(model.paramsB).toBeLessThan(ref.paramsB * 1.05);
  });

  it("estime les paramètres d'un config.json brut (Qwen3-8B ≈ 8,19 B)", () => {
    const json = JSON.parse(readFileSync(path.join(upstream, "Qwen__Qwen3-8B.json"), "utf8"));
    const { model, paramsSource } = convertHfConfig(json.config, { source: "Qwen/Qwen3-8B" });
    expect(paramsSource).toBe("estimation");
    expect(model.paramsB).toBeCloseTo(8.19, 1);
  });

  it("produit un YAML relisible sous models:", () => {
    const json = JSON.parse(readFileSync(path.join(upstream, "google__gemma-4-31B.json"), "utf8"));
    const yaml = modelToYaml(convertHfConfig(json).model);
    expect(yaml).toMatch(/^ {2}- id: gemma-4-31b/);
    expect(yaml).toContain("{ kind: sliding, count: 50");
    const parsed = parse(`models:\n${yaml}`) as { models: unknown[] };
    expect(() => modelSchema.parse(parsed.models[0])).not.toThrow();
  });
});
