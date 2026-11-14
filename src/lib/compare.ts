import { collectIssues, estimate } from "./calc";
import type { AppConfig, ModelSpec } from "./config-schema";
import type { Lang } from "./i18n";
import { estimatePerf } from "./perf";
import { resolveScenario, type Scenario, type Workload } from "./scenario";
import { analyzeWorkload } from "./workload";

export type CompareValue = string | number | boolean | null;
export type Direction = "higher" | "lower";
export type CompareFormat = "text" | "int" | "bytes" | "rate" | "usd" | "seconds" | "ms" | "slo";

/** Compare raw values, before rounding or localization. Missing values are meaningful. */
export function valuesDiffer(values: readonly CompareValue[]): boolean {
  return values.some((v) => !Object.is(v, values[0]));
}

/** All tied, finite winners; invalid configurations and unavailable values never win. */
export function bestIndices(values: readonly CompareValue[], direction: Direction, eligible = values.map(() => true)): number[] {
  const candidates = values.flatMap((v, i) => eligible[i] && typeof v === "number" && Number.isFinite(v) ? [{ v, i }] : []);
  if (!candidates.length) return [];
  const best = (direction === "higher" ? Math.max : Math.min)(...candidates.map(({ v }) => v));
  return candidates.filter(({ v }) => v === best).map(({ i }) => i);
}

export interface CompareRow {
  id: string;
  label: [string, string];
  format: CompareFormat;
  configuration: boolean;
  values: CompareValue[];
  different: boolean;
  best: number[];
}

export function workloadsDiffer(workloads: readonly Workload[]): boolean {
  const keys = ["inputTokens", "outputTokens", "requestsPerSec", "ttftSeconds", "itlMs"] as const;
  return keys.some((key) => valuesDiffer(workloads.map((w) => w[key])));
}

/** One instance's hourly and full-load decoding costs (not the workload's replica plan). */
export function instanceCost(gpus: number, price: number | undefined, outputTokPerSec: number | undefined) {
  const usdPerHour = price !== undefined && Number.isFinite(price) && price >= 0 ? gpus * price : null;
  const usdPerMillionOutput = usdPerHour !== null && outputTokPerSec !== undefined && Number.isFinite(outputTokPerSec) && outputTokPerSec > 0
    ? usdPerHour * 1e6 / (outputTokPerSec * 3600)
    : null;
  return { usdPerHour, usdPerMillionOutput };
}

/** Pure comparison model; workload estimates use the same public API as the workload panel. */
export function compareScenarios(config: AppConfig, models: ModelSpec[], scenarios: Scenario[], lang: Lang) {
  const columns = scenarios.map((scenario) => {
    const input = resolveScenario(config, models, scenario, lang);
    if (!input) return { scenario, input: null, valid: false, errors: 0, warnings: 0, est: null, perf: null, cost: null, workload: null };
    const est = estimate(input);
    const perf = estimatePerf(input, est, config.performance);
    const issues = collectIssues(input, est, config.weightFormats);
    const errors = issues.filter((i) => i.severity === "error").length;
    const price = scenario.priceUsd ?? input.gpu.priceUsdPerHour;
    return {
      scenario, input, est, perf, errors, valid: errors === 0,
      warnings: issues.filter((i) => i.severity === "warning").length,
      cost: instanceCost(input.gpuCount, price, perf.full?.tokPerSecTotal),
      workload: analyzeWorkload(config, input, scenario.workload, price),
    };
  });
  type Column = (typeof columns)[number];
  const rows: CompareRow[] = [];
  const row = (id: string, label: [string, string], format: CompareFormat, get: (c: Column) => CompareValue, direction?: Direction, configuration = false, load = false) => {
    const values = columns.map((c) => c.input ? get(c) : null);
    const eligible = columns.map((c) => c.valid && (!load || c.workload?.infeasible === null));
    rows.push({ id, label, format, configuration, values, different: valuesDiffer(values), best: direction ? bestIndices(values, direction, eligible) : [] });
  };
  row("model", ["Modèle", "Model"], "text", (c) => c.input!.model.name, undefined, true);
  row("cards", ["Cartes", "Cards"], "text", (c) => `${c.input!.gpuCount} × ${c.input!.gpu.name}`, undefined, true);
  row("engine", ["Moteur", "Engine"], "text", (c) => c.input!.engine.name, undefined, true);
  row("parallelism", ["Parallélisme", "Parallelism"], "text", (c) => c.input!.mode === "dp-ep" ? "DP-attention + EP" : `TP ${c.est!.tp} × PP ${c.est!.pp}`, undefined, true);
  row("formats", ["Poids / KV", "Weights / KV"], "text", (c) => `${c.input!.weightFormat.label} / ${c.input!.kvFormat.label}`, undefined, true);
  row("context", ["Contexte (tokens)", "Context (tokens)"], "int", (c) => c.input!.contextTokens, "higher", true);
  row("weights", ["Poids du modèle", "Model weights"], "bytes", (c) => c.est!.weights.totalBytes, "lower");
  row("kv-request", ["KV par requête", "KV per request"], "bytes", (c) => c.est!.kv.perRequestBytes, "lower");
  row("kv-available", ["KV disponible", "Available KV"], "bytes", (c) => c.est!.memory.freeForKvTotalBytes, "higher");
  row("sessions", ["Sessions", "Sessions"], "int", (c) => c.est!.sessions, "higher");
  row("single-throughput", ["Débit 1 session", "1-session throughput"], "rate", (c) => c.perf!.single.tokPerSecPerSession, "higher");
  row("full-throughput", ["Débit agrégé", "Aggregate throughput"], "rate", (c) => c.perf!.full?.tokPerSecTotal ?? null, "higher");
  row("gpus", ["GPUs au total / instance", "Total GPUs / instance"], "int", (c) => c.input!.gpuCount, "lower");
  row("hourly-cost", ["Coût / instance ($/h)", "Instance cost ($/h)"], "usd", (c) => c.cost!.usdPerHour, "lower");
  row("token-cost", ["$/M tokens de sortie · pleine charge", "$/M output tokens · full load"], "usd", (c) => c.cost!.usdPerMillionOutput, "lower");
  row("ttft", ["TTFT · 1 instance", "TTFT · 1 instance"], "seconds", (c) => c.workload!.single.ttftSeconds, "lower", false, true);
  row("itl", ["ITL · 1 instance", "ITL · 1 instance"], "ms", (c) => c.workload!.single.itlMs, "lower", false, true);
  row("slo", ["Objectifs tenus · 1 instance", "Targets met · 1 instance"], "slo", (c) => c.valid && c.workload!.infeasible === null && c.workload!.single.ok);
  row("replicas", ["Répliques nécessaires · charge et SLO", "Replicas needed · load and SLO"], "int", (c) => c.workload!.aggregated?.replicas ?? null, "lower", false, true);
  row("load-gpus", ["GPUs au total · charge et SLO", "Total GPUs · load and SLO"], "int", (c) => c.workload!.aggregated?.gpus ?? null, "lower", false, true);
  row("load-cost", ["Coût total · charge et SLO ($/h)", "Total cost · load and SLO ($/h)"], "usd", (c) => c.workload!.cost.aggregated?.usdPerHour ?? null, "lower", false, true);
  row("status", ["Statut", "Status"], "text", (c) => c.errors ? `⛔ ${c.errors}` : c.warnings ? `⚠ ${c.warnings}` : "✓");
  return { columns, rows, workloadsDiffer: workloadsDiffer(scenarios.map((s) => s.workload)) };
}
