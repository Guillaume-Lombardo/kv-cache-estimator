"use client";

import { collectIssues, contextLabel, estimate } from "@/lib/calc";
import type { AppConfig, ModelSpec } from "@/lib/config-schema";
import { estimatePerf } from "@/lib/perf";
import { resolveScenario, type Scenario } from "@/lib/scenario";
import { usePrefs } from "./Prefs";

interface Column {
  key: string;
  scenario: Scenario;
  current: boolean;
}

export function ComparePanel({
  config,
  models,
  current,
  pinned,
  onPin,
  onRemove,
  onLoad,
}: {
  config: AppConfig;
  models: ModelSpec[];
  current: Scenario;
  pinned: Scenario[];
  onPin: () => void;
  onRemove: (i: number) => void;
  onLoad: (s: Scenario) => void;
}) {
  const { lang, t, fmt } = usePrefs();
  const cols: Column[] = [{ key: "current", scenario: current, current: true }, ...pinned.map((s, i) => ({ key: `p${i}`, scenario: s, current: false }))];
  const data = cols.map((c) => {
    const input = resolveScenario(config, models, c.scenario, lang);
    if (!input) return { c, input: null };
    const est = estimate(input);
    const perf = estimatePerf(input, est, config.performance);
    const issues = collectIssues(input, est, config.weightFormats);
    return { c, input, est, perf, errors: issues.filter((i) => i.severity === "error").length, warnings: issues.filter((i) => i.severity === "warning").length };
  });
  // Only configurations without blocking errors can be ranked as best.
  const best = Math.max(0, ...data.filter((d) => d.est && !d.errors).map((d) => d.est!.sessions));

  const rows: { label: string; cell: (d: (typeof data)[number]) => React.ReactNode }[] = [
    { label: t("Modèle", "Model"), cell: (d) => d.input?.model.name },
    { label: t("Cartes", "Cards"), cell: (d) => d.input && `${d.input.gpuCount} × ${d.input.gpu.name}` },
    { label: t("Moteur", "Engine"), cell: (d) => d.input?.engine.name },
    { label: t("Parallélisme", "Parallelism"), cell: (d) => d.est && (d.input?.mode === "dp-ep" ? "DP-attention + EP" : d.input?.gpuCount === 1 ? "—" : `TP ${d.est.tp} × PP ${d.est.pp}`) },
    { label: t("Poids / KV", "Weights / KV"), cell: (d) => d.input && `${d.input.weightFormat.label} / ${d.input.kvFormat.label}` },
    { label: t("Contexte", "Context"), cell: (d) => d.input && contextLabel(d.input.contextTokens) },
    { label: t("Poids du modèle", "Model weights"), cell: (d) => d.est && fmt.bytes(d.est.weights.totalBytes) },
    { label: t("KV par requête", "KV per request"), cell: (d) => d.est && fmt.bytes(d.est.kv.perRequestBytes) },
    { label: t("KV disponible", "Available KV"), cell: (d) => d.est && fmt.bytes(d.est.memory.freeForKvTotalBytes) },
    {
      label: "Sessions",
      cell: (d) =>
        d.est && (
          <strong className={!d.errors && d.est.sessions === best && best > 0 ? "best" : d.errors ? "muted" : ""} title={d.errors ? t("Configuration invalide (erreurs bloquantes)", "Invalid configuration (blocking errors)") : undefined}>
            {fmt.int(d.est.sessions)}
            {d.errors ? t(" (invalide)", " (invalid)") : ""}
          </strong>
        ),
    },
    { label: t("Débit 1 session", "1-session throughput"), cell: (d) => d.perf && fmt.rate(d.perf.single.tokPerSecPerSession) },
    { label: t("Débit agrégé", "Aggregate throughput"), cell: (d) => (d.perf?.full ? fmt.rate(d.perf.full.tokPerSecTotal) : "—") },
    { label: t("Statut", "Status"), cell: (d) => d.est && (d.errors ? `⛔ ${d.errors}` : d.warnings ? `⚠️ ${d.warnings}` : "✓") },
  ];

  return (
    <div>
      <div className="row-between">
        <h3>{t("Comparaison de scénarios", "Scenario comparison")}</h3>
        <button type="button" className="btn" onClick={onPin} disabled={pinned.length >= 6}>
          📌 {t("Épingler la configuration actuelle", "Pin the current configuration")}
        </button>
      </div>
      {pinned.length === 0 && (
        <p className="hint">
          {t(
            "Épinglez des configurations (jusqu'à 6) pour les comparer à la configuration en cours. Elles sont gardées dans ce navigateur.",
            "Pin configurations (up to 6) to compare them with the current one. They are kept in this browser.",
          )}
        </p>
      )}
      <div className="table-wrap">
        <table className="table compact compare">
          <thead>
            <tr>
              <th />
              {data.map((d, i) => (
                <th key={d.c.key}>
                  {d.c.current ? (
                    t("Actuelle", "Current")
                  ) : (
                    <span className="col-actions">
                      #{i}
                      <button type="button" className="link" onClick={() => onLoad(d.c.scenario)}>
                        {t("charger", "load")}
                      </button>
                      <button type="button" className="link" onClick={() => onRemove(i - 1)} aria-label={t("Retirer", "Remove")}>
                        ✕
                      </button>
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label}>
                <th>{r.label}</th>
                {data.map((d) => (
                  <td key={d.c.key}>{d.input ? r.cell(d) : <span className="muted">{t("indisponible", "unavailable")}</span>}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
