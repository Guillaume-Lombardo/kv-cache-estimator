import { describe, expect, it } from "vitest";
import path from "node:path";
import { parseConfigFile } from "./load-config";
import { collectIssues, CONTEXT_STEPS, estimate, GIB, kvSupport, memoryFormatFor, weightSupport, type EstimateInput } from "./calc";
import { estimatePerf } from "./perf";
import { defaultTp, gpuCountOptions, maxContextFor, pickKvFormat, pickWeightFormat, sizeForTarget } from "./planning";
import { launchCommand } from "./scenario";
import type { AppConfig } from "./config-schema";
import { localize } from "./i18n";

const load = async (): Promise<AppConfig> => {
  const r = await parseConfigFile(path.join(__dirname, "../../config/estimator.yaml"));
  if (!r.ok) throw new Error(r.error);
  return localize(r.config, "fr");
};

type Overrides = { model?: string; gpu?: string; wf?: string; kf?: string; engine?: string } & Partial<
  Pick<EstimateInput, "gpuCount" | "contextTokens" | "reservePct" | "mode" | "tp" | "offloadGiBPerGpu" | "memoryGiBOverride">
>;

function find<T extends { id: string }>(xs: T[], id: string): T {
  const x = xs.find((v) => v.id === id);
  if (!x) throw new Error(id);
  return x;
}

function input(cfg: AppConfig, o: Overrides): EstimateInput {
  const n = o.gpuCount ?? 1;
  return {
    model: find(cfg.models, o.model ?? "qwen3-32b"),
    gpu: find(cfg.gpus, o.gpu ?? "h100"),
    engine: find(cfg.engines, o.engine ?? "vllm"),
    weightFormat: find(cfg.weightFormats, o.wf ?? "bf16"),
    kvFormat: find(cfg.kvCacheFormats, o.kf ?? "bf16"),
    gpuCount: n,
    mode: o.mode ?? "tp-pp",
    tp: o.tp ?? n,
    contextTokens: o.contextTokens ?? 32768,
    reservePct: o.reservePct ?? 10,
    offloadGiBPerGpu: o.offloadGiBPerGpu,
    memoryGiBOverride: o.memoryGiBOverride,
  };
}

describe("estimate", async () => {
  const cfg = await load();

  it("GQA : Qwen3-32B, 32k tokens en BF16 = 64 couches × 8 têtes × 256 × 2 o × 32768 = 8 Gio", () => {
    const e = estimate(input(cfg, {}));
    expect(e.kv.perRequestBytes).toBe(64 * 8 * 256 * 2 * 32768);
    expect(e.kv.perRequestBytes / GIB).toBe(8);
    expect(e.weights.totalBytes).toBeCloseTo(32.76e9 * 2, -6);
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
    expect(e.weights.totalBytes).toBeCloseTo((32.76e9 - emb) * (4.25 / 8) + emb * 2, 0);
  });

  it("utilise les bits réels du checkpoint natif (Kimi K2.5 : INT4 groupes de 32 ≈ 595 Go)", () => {
    const e = estimate(input(cfg, { model: "kimi-k2.5", gpu: "b200", wf: "int4", gpuCount: 8 }));
    expect(e.weights.bitsPerParam).toBe(4.5);
    expect(e.weights.totalBytes / 1e9).toBeGreaterThan(580);
    expect(e.weights.totalBytes / 1e9).toBeLessThan(610);
  });

  it("MLA : Kimi K2, 576 éléments par couche et par token, répliqué en TP, pas en DP-attention", () => {
    const tp = estimate(input(cfg, { model: "kimi-k2-instruct", gpu: "b200", wf: "fp8", gpuCount: 8 }));
    expect(tp.kv.perRequestBytes).toBe(61 * 576 * 2 * 32768);
    expect(tp.kv.perRequestPerGpuBytes).toBe(tp.kv.perRequestBytes);
    expect(tp.kv.perRequestClusterBytes).toBe(tp.kv.perRequestBytes * 8);

    const dp = estimate(input(cfg, { model: "kimi-k2-instruct", gpu: "b200", wf: "fp8", gpuCount: 8, mode: "dp-ep" }));
    expect(dp.kv.perRequestClusterBytes).toBe(dp.kv.perRequestBytes);
    expect(dp.weights.perGpuBytes).toBeGreaterThan(tp.weights.perGpuBytes);
    expect(dp.sessions).toBeGreaterThan(tp.sessions * 4);
  });

  it.each([{ kf: "fp8", bytes: 1 }, { kf: "bf16", bytes: 2 }])("shared MLA latents keep independent indexer storage with $kf KV", ({ kf, bytes }) => {
    const e = estimate(input(cfg, { model: "mistral-large-4", gpu: "b200", wf: "fp8", kf, gpuCount: 8 }));
    expect(e.kv.perTokenBytes).toBe(63 * 576 * bytes + 44 * 132);
    expect(e.kv.perRequestBytes).toBe(e.kv.perTokenBytes * 32768);
    expect(e.kv.perRequestPerGpuBytes).toBe(e.kv.perRequestBytes);
    expect(e.kv.perRequestClusterBytes).toBe(e.kv.perRequestBytes * 8);
    expect(e.kv.groups[2].bytes).toBe(9 * 132 * 32768);
    expect(e.kv.groups[2].detail).toContain("132");
  });

  it("TP × PP : couches entières, rang le plus chargé (61 couches sur 2 étages → 31)", () => {
    const e = estimate(input(cfg, { model: "kimi-k2-instruct", gpu: "b200", wf: "fp8", gpuCount: 16, tp: 8 }));
    expect(e.pp).toBe(2);
    expect(e.layersPerStage).toBe(31);
    expect(e.kv.perRequestPerGpuBytes).toBeCloseTo((e.kv.perRequestBytes * 31) / 61, 0);
  });

  it("fenêtre glissante : plafonnée dans vLLM, comptée en entier si la libération des blocs n'est pas vérifiée", () => {
    const perTok = 16 * 256 * 2;
    const v = estimate(input(cfg, { model: "gemma-3-27b", contextTokens: 131072 }));
    expect(v.kv.perRequestBytes).toBe(52 * perTok * 1024 + 10 * perTok * 131072);
    const z = estimate(input(cfg, { model: "gemma-3-27b", contextTokens: 131072, engine: "zml" }));
    expect(z.kv.perRequestBytes).toBe(62 * perTok * 131072);
  });

  it("couches linéaires : état SSM (FP32) + conv, constant", () => {
    const a = estimate(input(cfg, { model: "qwen3.5-35b-a3b", contextTokens: 8192 }));
    const b = estimate(input(cfg, { model: "qwen3.5-35b-a3b", contextTokens: 16384 }));
    expect(a.kv.constantBytes).toBe(30 * (524288 * 4 + 24576 * 2));
    expect(b.kv.perRequestBytes - b.kv.constantBytes).toBe(2 * (a.kv.perRequestBytes - a.kv.constantBytes));
  });

  it("attention compressée : DeepSeek-V4.1-Flash en NVFP4 = 890 o par token publiés, + fenêtre de 128 tokens", () => {
    const ctx = 1048576;
    const e = estimate(input(cfg, { model: "deepseek-v4.1-flash", gpu: "b200", kf: "nvfp4", contextTokens: ctx, gpuCount: 8 }));
    const entry = 512 * 0.5625 + 68;
    expect(e.kv.perTokenBytes).toBeCloseTo(890, 6);
    expect(e.kv.perRequestBytes).toBe(3 * (ctx / 2) * entry + ctx * entry + 40 * 128 * 512 * 0.5625);
    // A single shared K = V head: each TP rank keeps the whole cache, as with MLA.
    expect(e.kv.perRequestClusterBytes).toBe(8 * e.kv.perRequestBytes);
  });

  it("attention compressée : DeepSeek-V4-Flash, entrées arrondies à l'entrée supérieure et fenêtre bornée par le contexte", () => {
    const e = estimate(input(cfg, { model: "deepseek-v4-flash", kf: "fp8", contextTokens: 100 }));
    expect(e.kv.perRequestBytes).toBe(43 * 100 * 512 + 21 * 25 * (512 + 68) + 20 * 1 * 512);
  });

  it("TP au-delà du nombre de têtes KV : réplication signalée", () => {
    const i = input(cfg, { model: "qwen3-235b-2507", gpu: "h200", gpuCount: 8 });
    const e = estimate(i);
    expect(e.kv.groups[0].replication).toBe(2);
    expect(e.kv.perRequestClusterBytes).toBe(e.kv.perRequestBytes * 2);
    expect(collectIssues(i, e, cfg.weightFormats).some((x) => x.title.startsWith("KV répliqué ×2"))).toBe(true);
  });

  it("sessions = mémoire libre / KV par requête", () => {
    const e = estimate(input(cfg, { gpuCount: 2 }));
    expect(e.sessions).toBe(Math.floor(e.memory.freeForKvPerGpuBytes / e.kv.perRequestPerGpuBytes));
    expect(e.kv.perRequestPerGpuBytes).toBe(e.kv.perRequestBytes / 2);
  });

  it("mémoire visible personnalisée", () => {
    const e = estimate(input(cfg, { memoryGiBOverride: 70 }));
    expect(e.memory.gpuBytes).toBe(70 * GIB);
  });

  it("offload CPU : sessions en réserve et temps de rechargement", () => {
    const e = estimate(input(cfg, { gpu: "gh200", offloadGiBPerGpu: 400 }));
    expect(e.offload.parkedSessions).toBe(Math.floor((400 * GIB) / e.kv.perRequestClusterBytes));
    expect(e.offload.reloadMs).toBeCloseTo(((8 * GIB) / 450e9) * 1000, 3);
    // Unverified offload in ZML/LLMD: counted with a warning.
    const zi = input(cfg, { gpu: "gh200", offloadGiBPerGpu: 400, engine: "zml" });
    expect(collectIssues(zi, estimate(zi), cfg.weightFormats).some((i) => i.title.startsWith("Offload KV non vérifié"))).toBe(true);
  });

  it("signale les limitations matérielles et moteur", () => {
    const v100 = input(cfg, { gpu: "v100", wf: "bf16", kf: "fp8" });
    const titles = collectIssues(v100, estimate(v100), cfg.weightFormats).map((i) => `${i.severity}:${i.title}`);
    expect(titles).toEqual(
      expect.arrayContaining([expect.stringMatching(/^error:Poids BF16/), expect.stringMatching(/^error:KV cache FP8/), expect.stringMatching(/^warning:vLLM.*V100/)]),
    );
    const l40s = input(cfg, { gpu: "l40s", gpuCount: 4, model: "qwen3-8b" });
    expect(collectIssues(l40s, estimate(l40s), cfg.weightFormats).some((i) => i.title.includes("sans NVLink"))).toBe(true);
  });

  it("TP impossible si les têtes ne sont pas divisibles", () => {
    const i = input(cfg, { gpuCount: 3 });
    expect(collectIssues(i, estimate(i), cfg.weightFormats)[0].title).toMatch(/TP=3 impossible/);
  });
});

describe("profils moteur", async () => {
  const cfg = await load();
  const zml = find(cfg.engines, "zml");
  const h100 = find(cfg.gpus, "h100");

  it("ZML / LLMD : INT4 et KV FP8 non supportés, FP8 non vérifié, replis automatiques", () => {
    expect(weightSupport(h100, zml, "int4").level).toBe("unsupported");
    expect(weightSupport(h100, zml, "fp8").level).toBe("unverified");
    expect(kvSupport(h100, zml, "fp8").level).toBe("unsupported");
    expect(pickKvFormat(cfg, h100, zml, "fp8")).toBe("bf16");
    expect(pickWeightFormat(cfg, find(cfg.models, "qwen3-32b"), h100, zml, "int4")).toBe("bf16");
  });

  it("format non vérifié : calcul au format du fichier, alternative BF16 signalée", () => {
    const i = input(cfg, { engine: "zml", wf: "fp8" });
    const issues = collectIssues(i, estimate(i), cfg.weightFormats);
    expect(issues.some((x) => x.title.startsWith("Format en mémoire non vérifié pour FP8"))).toBe(true);
  });

  it("format en mémoire distinct du fichier : la VRAM suit le format chargé", () => {
    const eng = { ...zml, weightFormats: { ...zml.weightFormats, fp8: { level: "native" as const, memoryFormat: "bf16" } } };
    const fp8 = find(cfg.weightFormats, "fp8");
    const mem = memoryFormatFor(cfg.weightFormats, h100, eng, fp8);
    expect(mem.id).toBe("bf16");
    const i = { ...input(cfg, { wf: "fp8" }), engine: eng, memoryWeightFormat: mem };
    const e = estimate(i);
    expect(e.weights.totalBytes).toBeCloseTo(32.76e9 * 2, -6);
    expect(collectIssues(i, e, cfg.weightFormats).some((x) => x.title === "Fichier FP8 chargé en BF16 en mémoire")).toBe(true);
  });

  it("fenêtre glissante : économie seulement si la libération des blocs est confirmée", () => {
    const i = input(cfg, { model: "gemma-3-27b", contextTokens: 131072, engine: "zml" });
    const issues = collectIssues(i, estimate(i), cfg.weightFormats);
    expect(issues.some((x) => x.title.includes("comptées au contexte complet"))).toBe(true);
    const off = { ...input(cfg, { model: "gemma-3-27b" }), engine: { ...zml, features: { ...zml.features, slidingWindowAttention: false as const } } };
    expect(collectIssues(off, estimate(off), cfg.weightFormats)[0].title).toMatch(/Attention glissante non supportée/);
  });
});

describe("review e00a496", async () => {
  const cfg = await load();

  it("P1 contexte : pas de recommandation au-delà du contexte maximal du modèle", () => {
    const i = input(cfg, { model: "qwen3-8b", gpu: "h200", gpuCount: 8, kf: "fp8", contextTokens: 1048576 });
    expect(collectIssues(i, estimate(i), cfg.weightFormats).some((x) => x.severity === "error" && x.title.startsWith("Contexte"))).toBe(true);
    expect(maxContextFor(cfg, i, 1)).toBe(131072);
    const rows = sizeForTarget(cfg, { ...i, weightFormatId: "bf16", kvFormatId: "fp8" }, 1, "h200");
    expect(rows.every((r) => r.gpuCount === null)).toBe(true);
    expect(rows[0].note).toContain("contexte hors limite");
  });

  it("P1 VRAM personnalisée : appliquée à la seule carte saisie", () => {
    const base = input(cfg, { model: "qwen3-8b", gpu: "h200", memoryGiBOverride: 140 });
    const withOv = sizeForTarget(cfg, { ...base, weightFormatId: "bf16", kvFormatId: "bf16" }, 20, "h200");
    const without = sizeForTarget(cfg, { ...base, memoryGiBOverride: undefined, weightFormatId: "bf16", kvFormatId: "bf16" }, 20, "h200");
    for (const id of ["v100", "l40s"]) {
      const a = withOv.find((r) => r.gpu.id === id)!;
      const b = without.find((r) => r.gpu.id === id)!;
      expect([a.gpuCount, a.sessions]).toEqual([b.gpuCount, b.sessions]);
      expect(a.gpuCount).not.toBe(1);
    }
  });

  it("P1 PP : refus au-delà du nombre de couches, rang limitant sinon", () => {
    const i = input(cfg, { model: "qwen3-8b", gpu: "gb200", gpuCount: 72, tp: 1 });
    expect(collectIssues(i, estimate(i), cfg.weightFormats).some((x) => x.severity === "error" && x.title.startsWith("PP=72 supérieur"))).toBe(true);
    const e = estimate(input(cfg, { model: "qwen3-8b", gpuCount: 8, tp: 1 }));
    expect(e.layersPerStage).toBe(5);
    expect(e.kv.perRequestPerGpuBytes).toBe((e.kv.perRequestBytes * 5) / 36);
  });

  it("P1 DP-attention : tensors non experts répliqués à leur précision réelle", () => {
    const e = estimate(input(cfg, { model: "kimi-k2.5", gpu: "b300", wf: "int4", gpuCount: 8, mode: "dp-ep" }));
    // Attention alone in BF16: 6.2 B x 2 bytes = about 11.55 GiB; total non-expert storage exceeds this.
    expect(e.weights.nonExpertBytes / GIB).toBeGreaterThan(11.55);
    expect(e.weights.perGpuBytes).toBeCloseTo(e.weights.nonExpertBytes + (e.weights.totalBytes - e.weights.nonExpertBytes) / 8, 0);
    const unknown = input(cfg, { model: "kimi-linear", gpuCount: 4, mode: "dp-ep" });
    expect(collectIssues(unknown, estimate(unknown), cfg.weightFormats).some((x) => x.severity === "error" && x.title.includes("experts inconnus"))).toBe(true);
  });

  it("DP-attention : experts FP4 répartis, autres poids FP8 répliqués (DeepSeek-V4.1-Flash)", () => {
    const e = estimate(input(cfg, { model: "deepseek-v4.1-flash", gpu: "b300", wf: "mxfp4", gpuCount: 8, mode: "dp-ep" }));
    const keep = 129280 * 5120 * 2 + 0.69e9;
    expect(e.weights.nonExpertBytes).toBeCloseTo((748e9 - keep - 543.6e9) + keep * 2, -6);
    expect(e.weights.totalBytes - e.weights.nonExpertBytes).toBeCloseTo((543.6e9 * 4.25) / 8, -6);
  });

  it("P2 commande : checkpoint à fournir hors format natif", () => {
    const i = input(cfg, { model: "qwen3-8b", wf: "nvfp4", gpu: "b200" });
    const c = launchCommand(i, 1, 1)!;
    expect(c.matchesScenario).toBe(false);
    expect(c.text).toContain("<checkpoint-nvfp4-de-qwen3-8b>");
    expect(c.text).not.toMatch(/vllm serve Qwen\/Qwen3-8B/);
    expect(launchCommand(input(cfg, { model: "qwen3-8b" }), 1, 1)!.text).toContain("vllm serve Qwen/Qwen3-8B ");
    expect(launchCommand(input(cfg, { model: "qwen3-8b", wf: "int4" }), 1, 1)!.text).toContain("vllm serve Qwen/Qwen3-8B-AWQ");
  });

  it("P2 modèle + 1 KV : occupation physique avec réplication MLA", () => {
    const e = estimate(input(cfg, { model: "kimi-k2-instruct", gpu: "b200", wf: "fp8", gpuCount: 8 }));
    expect(e.physicalOneRequestBytes).toBeCloseTo(e.weights.totalBytes + 8 * e.kv.perRequestBytes, 0);
    expect(e.physicalOneRequestBytes).toBeGreaterThan(e.modelPlusOneKvBytes);
  });
});

describe("review 0e5ac67", async () => {
  const cfg = await load();
  it("mémoire libre physique totale distincte du budget prudent en PP", () => {
    const e = estimate(input(cfg, { model: "qwen3-8b", gpu: "h200", gpuCount: 8, tp: 1 }));
    const physical = 8 * e.memory.usablePerGpuBytes - e.weights.totalBytes;
    expect(e.memory.freeForKvTotalBytes).toBeCloseTo(physical, 0);
    expect(e.memory.freeForKvTotalBytes / GIB).toBeCloseTo(995.62, 1);
    expect(e.memory.kvBudgetTotalBytes / GIB).toBeCloseTo(987.23, 1);
    expect(e.sessions).toBe(Math.floor(e.memory.freeForKvPerGpuBytes / e.kv.perRequestPerGpuBytes));
  });
  it("TP pur et DP-attention : les deux valeurs coïncident", () => {
    for (const o of [{ gpuCount: 8 }, { model: "qwen3-30b-a3b-2507", gpuCount: 8, mode: "dp-ep" as const }]) {
      const e = estimate(input(cfg, { gpu: "h200", ...o }));
      expect(e.memory.kvBudgetTotalBytes).toBeCloseTo(e.memory.freeForKvTotalBytes, 0);
    }
  });
});

describe("performance", async () => {
  const cfg = await load();
  it("décodage limité par la mémoire : ~ bande passante / octets lus", () => {
    const i = input(cfg, { wf: "fp8", contextTokens: 8192 });
    const e = estimate(i);
    const p = estimatePerf(i, e, cfg.performance);
    const expected = (3.35e12 * 0.7) / (e.weights.totalBytes + e.kv.perRequestBytes);
    expect(p.single.bound).toBe("memory");
    expect(p.single.tokPerSecPerSession).toBeCloseTo(expected, 5);
    expect(p.full!.tokPerSecTotal).toBeGreaterThan(p.single.tokPerSecTotal);
  });
  it("MoE : un token ne lit que les experts actifs", () => {
    const i = input(cfg, { model: "qwen3-30b-a3b-2507", contextTokens: 8192 });
    const p = estimatePerf(i, estimate(i), cfg.performance);
    const dense = input(cfg, { model: "qwen3-32b", contextTokens: 8192 });
    expect(p.single.tokPerSecPerSession).toBeGreaterThan(3 * estimatePerf(dense, estimate(dense), cfg.performance).single.tokPerSecPerSession);
  });
});

describe("planification", async () => {
  const cfg = await load();
  it("TP par défaut : plus grand diviseur valide dans le domaine NVLink", () => {
    const m = find(cfg.models, "qwen3-32b");
    expect(defaultTp(m, find(cfg.gpus, "h100"), 8)).toBe(8);
    expect(defaultTp(m, find(cfg.gpus, "h100"), 16)).toBe(8);
    expect(defaultTp(m, find(cfg.gpus, "gb200"), 64)).toBe(8);
    expect(defaultTp(m, find(cfg.gpus, "h100"), 6)).toBe(2);
  });
  it("72 GPUs proposés uniquement en NVL72", () => {
    expect(gpuCountOptions(cfg, find(cfg.gpus, "gb200"))).toContain(72);
    expect(gpuCountOptions(cfg, find(cfg.gpus, "h100"))).not.toContain(72);
  });
  it("dimensionnement inverse : atteint l'objectif avec le moins de GPUs", () => {
    const base = input(cfg, { model: "llama-3.3-70b", contextTokens: 131072 });
    const rows = sizeForTarget(cfg, { ...base, weightFormatId: "fp8", kvFormatId: "fp8" }, 16);
    const h200 = rows.find((r) => r.gpu.id === "h200")!;
    expect(h200.gpuCount).not.toBeNull();
    expect(h200.sessions).toBeGreaterThanOrEqual(16);
  });
  it("contexte max pour N sessions", () => {
    const i = input(cfg, { gpuCount: 2 });
    const c = maxContextFor(cfg, i, 4);
    expect(c).not.toBeNull();
    expect(estimate({ ...i, contextTokens: c! }).sessions).toBeGreaterThanOrEqual(4);
  });
});

describe("balayage complet", async () => {
  const cfg = await load();
  it("aucun NaN ni exception sur modèles × GPUs × contextes × tailles × modes", () => {
    let n = 0;
    for (const m of cfg.models)
      for (const g of cfg.gpus)
        for (const ctx of [CONTEXT_STEPS[0], CONTEXT_STEPS[4], CONTEXT_STEPS[7]])
          for (const count of [1, 3, 8, 16])
            for (const mode of ["tp-pp", "dp-ep"] as const) {
              const i: EstimateInput = {
                ...input(cfg, { gpuCount: count, contextTokens: ctx, mode }),
                model: m,
                gpu: g,
                tp: defaultTp(m, g, count),
              };
              const e = estimate(i);
              const p = estimatePerf(i, e, cfg.performance);
              collectIssues(i, e, cfg.weightFormats);
              for (const v of [e.weights.totalBytes, e.kv.perRequestBytes, e.sessions, p.single.tokPerSecPerSession, p.prefillSeconds])
                expect(Number.isFinite(v)).toBe(true);
              n++;
            }
    expect(n).toBe(cfg.models.length * cfg.gpus.length * 3 * 4 * 2);
  });
});
