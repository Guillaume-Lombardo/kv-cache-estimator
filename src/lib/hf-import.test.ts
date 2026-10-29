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
    expect(paramsSource).toBe("estimate");
    expect(model.paramsB).toBeCloseTo(8.19, 1);
  });

  it("imports shared MLA latents and FP8 weights from Mistral params.json", () => {
    const { model } = convertHfConfig({
      dim: 6144, n_layers: 4, n_heads: 64, vocab_size: 204800,
      kv_lora_rank: 512, qk_rope_head_dim: 64,
      dsa: { head_dim: 128, sharing: "full,shared,full,shared" },
      kv_sharing: { source_layers_str: "0,1,0,1" },
      quantization: { qformat_weight: "fp8_e4m3" },
    });
    expect(model.nativeWeightFormat).toBe("fp8");
    expect(model.layers).toEqual([
      { kind: "mla", count: 1, kvLoraRank: 512, ropeHeadDim: 64, extraBytesPerToken: 132 },
      { kind: "mla", count: 1, kvLoraRank: 512, ropeHeadDim: 64 },
      { kind: "mla", count: 1, kvLoraRank: 512, ropeHeadDim: 64, extraBytesPerToken: 132, kvShared: true },
      { kind: "mla", count: 1, kvLoraRank: 512, ropeHeadDim: 64, kvShared: true },
    ]);
  });

  it("imports cross-layer KV sharing without a sparse indexer", () => {
    const { model } = convertHfConfig({
      hidden_size: 4096, num_hidden_layers: 4, num_attention_heads: 32, vocab_size: 32000,
      kv_lora_rank: 512, qk_rope_head_dim: 64, kv_source_layers: [0, 1, 0, 1],
    });
    expect(model.layers).toEqual([
      { kind: "mla", count: 2, kvLoraRank: 512, ropeHeadDim: 64 },
      { kind: "mla", count: 2, kvLoraRank: 512, ropeHeadDim: 64, kvShared: true },
    ]);
  });

  it("recognizes ModelOpt mixed-precision NVFP4 checkpoints", () => {
    const json = JSON.parse(readFileSync(path.join(upstream, "mistralai__Mistral-Large-4-1T-A52B.json"), "utf8"));
    json.config.quantization_config = {
      quant_method: "modelopt", quant_algo: "MIXED_PRECISION",
      quantized_layers: { "model.layers.0.mlp.experts": { quant_algo: "NVFP4" } },
    };
    expect(convertHfConfig(json).model.nativeWeightFormat).toBe("nvfp4");
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
