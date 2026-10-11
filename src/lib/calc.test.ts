import { describe, expect, it } from "vitest";
import path from "node:path";
import { parseConfigFile } from "./load-config";
import { collectIssues, estimate, GIB, type EstimateInput } from "./calc";
import type { AppConfig } from "./config-schema";

const load = async (): Promise<AppConfig> => {
  const r = await parseConfigFile(path.join(__dirname, "../../config/estimator.yaml"));
  if (!r.ok) throw new Error(r.error);
  return r.config;
};

type Overrides = { model?: string; gpu?: string; wf?: string; kf?: string } & Partial<
  Pick<EstimateInput, "gpuCount" | "contextTokens" | "reservePct" | "parallelism">
>;

function input(cfg: AppConfig, o: Overrides): EstimateInput {
  const find = <T extends { id: string }>(xs: T[], id: string) => {
    const x = xs.find((v) => v.id === id);
    if (!x) throw new Error(id);
    return x;
  };
  return {
    model: find(cfg.models, o.model ?? "qwen3-32b"),
    gpu: find(cfg.gpus, o.gpu ?? "h100"),
    weightFormat: find(cfg.weightFormats, o.wf ?? "bf16"),
    kvFormat: find(cfg.kvCacheFormats, o.kf ?? "bf16"),
    gpuCount: o.gpuCount ?? 1,
    contextTokens: o.contextTokens ?? 32768,
    reservePct: o.reservePct ?? 10,
    parallelism: o.parallelism ?? "tp",
  };
}

describe("estimate", async () => {
  const cfg = await load();

  it("GQA : Qwen3-32B, 32k tokens en BF16 = 64 couches × 8 têtes × 256 × 2 o × 32768 = 8 Gio", () => {
    const e = estimate(input(cfg, {}));
    expect(e.kv.perRequestBytes).toBe(64 * 8 * 256 * 2 * 32768);
    expect(e.kv.perRequestBytes / GIB).toBe(8);
    // 32,8 B params en BF16
    expect(e.weights.totalBytes).toBeCloseTo(32.8e9 * 2, -6);
  });

  it("FP8 KV divise le cache par deux", () => {
    const a = estimate(input(cfg, {}));
    const b = estimate(input(cfg, { kf: "fp8" }));
    expect(b.kv.perRequestBytes).toBe(a.kv.perRequestBytes / 2);
  });

  it("garde les embeddings en 16 bits quand on quantifie", () => {
    const e = estimate(input(cfg, { wf: "int4" }));
    const emb = 151936 * 5120 * 2;
    expect(e.weights.unquantizedParams).toBe(emb);
    expect(e.weights.totalBytes).toBeCloseTo((32.8e9 - emb) * (4.25 / 8) + emb * 2, 0);
  });

  it("MLA : Kimi K2, 576 éléments par couche et par token, répliqué en TP", () => {
    const one = estimate(input(cfg, { model: "kimi-k2-instruct", gpu: "b200", wf: "fp8", gpuCount: 8 }));
    expect(one.kv.perRequestBytes).toBe(61 * 576 * 2 * 32768);
    expect(one.kv.perRequestPerGpuBytes).toBe(one.kv.perRequestBytes);
    expect(one.kv.perRequestClusterBytes).toBe(one.kv.perRequestBytes * 8);

    const dp = estimate(input(cfg, { model: "kimi-k2-instruct", gpu: "b200", wf: "fp8", gpuCount: 8, parallelism: "dp-attention" }));
    expect(dp.kv.perRequestClusterBytes).toBe(dp.kv.perRequestBytes);
    // Attention répliquée → plus de poids par GPU qu'en TP, mais ~8× plus de sessions.
    expect(dp.weights.perGpuBytes).toBeGreaterThan(one.weights.perGpuBytes);
    expect(dp.sessions).toBeGreaterThan(one.sessions * 6);
  });

  it("fenêtre glissante : Gemma 3 27B plafonne le KV local à 1024 tokens", () => {
    const e = estimate(input(cfg, { model: "gemma-3-27b", contextTokens: 131072 }));
    const perTok = 16 * 256 * 2;
    expect(e.kv.perRequestBytes).toBe(52 * perTok * 1024 + 10 * perTok * 131072);
  });

  it("couches linéaires : état constant indépendant du contexte", () => {
    const a = estimate(input(cfg, { model: "qwen3-next-80b", contextTokens: 8192 }));
    const b = estimate(input(cfg, { model: "qwen3-next-80b", contextTokens: 16384 }));
    expect(a.kv.constantBytes).toBe(36 * 548864 * 2);
    expect(b.kv.perRequestBytes - b.kv.constantBytes).toBe(2 * (a.kv.perRequestBytes - a.kv.constantBytes));
  });

  it("TP au-delà du nombre de têtes KV : réplication", () => {
    // Qwen3-235B : 4 têtes KV, TP=8 → chaque tête sur 2 GPUs.
    const e = estimate(input(cfg, { model: "qwen3-235b-2507", gpu: "h200", gpuCount: 8 }));
    expect(e.kv.groups[0].replication).toBe(2);
    expect(e.kv.perRequestClusterBytes).toBe(e.kv.perRequestBytes * 2);
    const issues = collectIssues(input(cfg, { model: "qwen3-235b-2507", gpu: "h200", gpuCount: 8 }), e, cfg.weightFormats);
    expect(issues.some((i) => i.title.startsWith("KV répliqué ×2"))).toBe(true);
  });

  it("sessions = mémoire libre / KV par requête", () => {
    const i = input(cfg, { gpuCount: 2 });
    const e = estimate(i);
    expect(e.sessions).toBe(Math.floor(e.memory.freeForKvPerGpuBytes / e.kv.perRequestPerGpuBytes));
    expect(e.kv.perRequestPerGpuBytes).toBe(e.kv.perRequestBytes / 2);
  });

  it("signale les limitations matérielles", () => {
    const v100 = input(cfg, { gpu: "v100", wf: "bf16", kf: "fp8" });
    const issues = collectIssues(v100, estimate(v100), cfg.weightFormats);
    expect(issues.filter((i) => i.severity === "error").map((i) => i.title)).toEqual(
      expect.arrayContaining([expect.stringContaining("Poids BF16"), expect.stringContaining("KV cache FP8")]),
    );
    const l40s = input(cfg, { gpu: "l40s", gpuCount: 4, model: "qwen3-8b" });
    expect(collectIssues(l40s, estimate(l40s), cfg.weightFormats).some((i) => i.title.includes("NVLink"))).toBe(true);
  });

  it("TP impossible si les têtes ne sont pas divisibles", () => {
    const i = input(cfg, { gpuCount: 3 });
    expect(collectIssues(i, estimate(i), cfg.weightFormats)[0].title).toMatch(/TP=3 impossible/);
  });
});
