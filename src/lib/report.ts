import { collectIssues, estimate, GIB } from "./calc";
import type { AppConfig, ModelSpec } from "./config-schema";
import type { Lang } from "./i18n";
import { estimatePerf } from "./perf";
import { launchCommand, resolveScenario, type Scenario } from "./scenario";
import { analyzeWorkload } from "./workload";

/** Items the estimate leaves out, reported with every result. */
export const UNMODELLED: Record<Lang, string[]> = {
  fr: [
    "Réutilisation du prefix cache (sessions partageant un préfixe)",
    "Décodage spéculatif (modèle brouillon, tokens acceptés)",
    "Adaptateurs LoRA et leur cache",
    "Encodeur vision ou audio : activations et tokens d'image",
    "État FP32 des compresseurs (attention compressée DeepSeek V4)",
    "Fragmentation de l'allocateur au-delà de la réserve",
    "Attente en file avant le prefill (TTFT sans file d'attente)",
    "Latence réseau entre pools prefill et decode (seul le débit est compté)",
  ],
  en: [
    "Prefix-cache reuse (sessions sharing a prefix)",
    "Speculative decoding (draft model, accepted tokens)",
    "LoRA adapters and their cache",
    "Vision or audio encoder: activations and image tokens",
    "Compressors' FP32 state (DeepSeek V4 compressed attention)",
    "Allocator fragmentation beyond the reserve",
    "Queueing before prefill (TTFT without a queue)",
    "Network latency between prefill and decode pools (only bandwidth is counted)",
  ],
};

const gib = (b: number) => b / GIB;
const finite = (x: number) => (Number.isFinite(x) ? x : null);

/** Machine-readable result of a scenario: JSON export, /api/estimate and the CLI. Null if an identifier is unknown. */
export function buildReport(cfg: AppConfig, models: ModelSpec[], s: Scenario, lang: Lang) {
  const input = resolveScenario(cfg, models, s, lang);
  if (!input) return null;
  const est = estimate(input);
  const perf = estimatePerf(input, est, cfg.performance);
  const issues = collectIssues(input, est, cfg.weightFormats);
  const command = launchCommand(input, est.tp, est.pp);
  const wl = analyzeWorkload(cfg, input, s.workload, s.priceUsd ?? input.gpu.priceUsdPerHour);
  const ov = est.memory.overhead;
  return {
    scenario: s,
    model: { id: input.model.id, name: input.model.name, source: input.model.source, revision: input.model.revision },
    gpu: { id: input.gpu.id, name: input.gpu.name, memoryGiB: gib(est.memory.gpuBytes) },
    engine: input.engine.name,
    parallelism: { mode: s.mode, tp: est.tp, pp: est.pp, gpuCount: s.gpuCount },
    results: {
      weightsGiB: gib(est.weights.totalBytes),
      weightsPerGpuGiB: gib(est.weights.perGpuBytes),
      kvPerRequestGiB: gib(est.kv.perRequestBytes),
      kvPerRequestClusterGiB: gib(est.kv.perRequestClusterBytes),
      modelPlusOneKvGiB: gib(est.modelPlusOneKvBytes),
      reservePerGpuGiB: gib(est.memory.reservePerGpuBytes),
      overheadPerGpu: {
        totalGiB: gib(ov.totalBytes),
        activationGiB: gib(ov.activationBytes),
        samplerGiB: gib(ov.samplerBytes),
        fixedGiB: gib(ov.fixedBytes),
        batchedTokens: ov.batchedTokens,
        overridden: ov.overridden,
        inEngineBudget: ov.inBudget,
      },
      freeKvPhysicalTotalGiB: gib(est.memory.freeForKvTotalBytes),
      kvBudgetTotalGiB: gib(est.memory.kvBudgetTotalBytes),
      valid: !issues.some((i) => i.severity === "error"),
      sessions: est.sessions,
      offloadParkedSessions: est.offload.parkedSessions,
      decodeTokPerSecSingle: perf.single.tokPerSecPerSession,
      decodeTokPerSecFullLoad: perf.full?.tokPerSecTotal ?? null,
      prefillSeconds: perf.prefillSeconds,
    },
    workload: {
      ...s.workload,
      averageResidentTokens: wl.model.avgTokens,
      capacityPerInstance: wl.model.capacity,
      infeasible: wl.infeasible,
      singleInstance: {
        concurrency: finite(wl.single.concurrency),
        ttftSeconds: finite(wl.single.ttftSeconds),
        itlMs: finite(wl.single.itlMs),
        prefillUtil: wl.single.prefillUtil,
        meetsTargets: wl.single.ok,
      },
      aggregated: wl.aggregated && {
        replicas: wl.aggregated.replicas,
        gpus: wl.aggregated.gpus,
        ttftSeconds: wl.aggregated.load.ttftSeconds,
        itlMs: wl.aggregated.load.itlMs,
        concurrencyPerInstance: wl.aggregated.load.concurrency,
        cost: wl.cost.aggregated,
      },
      disaggregated: wl.disaggregated && {
        prefillReplicas: wl.disaggregated.prefillReplicas,
        decodeReplicas: wl.disaggregated.decodeReplicas,
        gpus: wl.disaggregated.gpus,
        ttftSeconds: wl.disaggregated.ttftSeconds,
        itlMs: wl.disaggregated.decode.itlMs,
        kvTrafficGBs: wl.disaggregated.kvTrafficBytesPerSec / 1e9,
        cost: wl.cost.disaggregated,
      },
      priceUsdPerGpuHour: wl.priceUsdPerHour ?? null,
    },
    issues,
    unmodelled: UNMODELLED[lang],
    command: command?.text ?? null,
  };
}
export type Report = NonNullable<ReturnType<typeof buildReport>>;

/** Identifiers in the query that the catalogue does not know (the UI falls back silently; the API and CLI refuse). */
export function unknownIdentifiers(cfg: AppConfig, models: ModelSpec[], q: Record<string, string | undefined>): string[] {
  const checks: [string, { id: string }[]][] = [
    ["model", models],
    ["gpu", cfg.gpus],
    ["eng", cfg.engines],
    ["w", cfg.weightFormats],
    ["kv", cfg.kvCacheFormats],
  ];
  return checks.filter(([k, xs]) => q[k] !== undefined && !xs.some((x) => x.id === q[k])).map(([k]) => `${k}=${q[k]}`);
}
