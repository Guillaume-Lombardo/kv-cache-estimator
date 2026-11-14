"use client";

import { useMemo, useState } from "react";
import { compareScenarios, type CompareFormat, type CompareValue } from "@/lib/compare";
import type { AppConfig, ModelSpec } from "@/lib/config-schema";
import type { Scenario, Workload } from "@/lib/scenario";
import { usePrefs } from "./Prefs";

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
  const [hideIdentical, setHideIdentical] = useState(false);
  const comparison = useMemo(() => compareScenarios(config, models, [current, ...pinned], lang), [config, models, current, pinned, lang]);
  const { columns, rows } = comparison;
  const workloadLabel = (w: Workload) => t(
    `${fmt.int(w.inputTokens)} tokens en entrée → ${fmt.int(w.outputTokens)} en sortie · ${fmt.num(w.requestsPerSec)} req/s · cibles TTFT ≤ ${fmt.duration(w.ttftSeconds)}, ITL ≤ ${fmt.num(w.itlMs)} ms`,
    `${fmt.int(w.inputTokens)} input tokens → ${fmt.int(w.outputTokens)} output · ${fmt.num(w.requestsPerSec)} req/s · targets TTFT ≤ ${fmt.duration(w.ttftSeconds)}, ITL ≤ ${fmt.num(w.itlMs)} ms`,
  );
  const cell = (value: CompareValue, format: CompareFormat) => {
    if (value === null) return "—";
    if (typeof value === "boolean") return <span className={value ? "slo-ok" : "slo-bad"}>{value ? "✓" : "✕"} {value ? t("Oui", "Yes") : t("Non", "No")}</span>;
    if (typeof value === "string") return value;
    if (!Number.isFinite(value)) return "∞";
    switch (format) {
      case "int": return fmt.int(value);
      case "bytes": return fmt.bytes(value);
      case "rate": return fmt.rate(value);
      case "usd": return lang === "fr" ? `${fmt.num(value)} $` : `$${fmt.num(value)}`;
      case "seconds": return fmt.duration(value);
      case "ms": return `${fmt.num(value)} ms`;
      default: return fmt.num(value);
    }
  };

  return (
    <div className="compare-panel">
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
      <div className="compare-options">
        <label>
          <input type="checkbox" checked={hideIdentical} onChange={(e) => setHideIdentical(e.target.checked)} />
          {t("Masquer les lignes identiques", "Hide identical rows")}
        </label>
        <span className="hint">{t("Fond teinté : configuration différente. Vert : meilleure valeur valide (ex æquo inclus).", "Tinted background: different configuration. Green: best valid value (including ties).")}</span>
      </div>
      <p className="hint">{t("Charge et SLO : répliques identiques en service agrégé. Les coûts sont indicatifs, hors hôte et réseau.", "Load and SLO: identical replicas with aggregated serving. Costs are indicative, excluding host and network.")}</p>
      {!comparison.workloadsDiffer && <p className="hint">{t("Charge commune : ", "Shared workload: ")}{workloadLabel(current.workload)}</p>}
      <div className="table-wrap" role="region" aria-label={t("Tableau de comparaison", "Comparison table")} tabIndex={0}>
        <table className="table compact compare">
          <thead>
            <tr>
              <th scope="col">{t("Critère", "Metric")}</th>
              {columns.map((d, i) => (
                <th key={i} scope="col">
                  {i === 0 ? (
                    t("Actuelle", "Current")
                  ) : (
                    <span className="col-actions">
                      #{fmt.int(i)}
                      <button type="button" className="link" onClick={() => onLoad(d.scenario)}>
                        {t("charger", "load")}
                      </button>
                      <button type="button" className="link" onClick={() => onRemove(i - 1)} aria-label={t(`Retirer le scénario ${fmt.int(i)}`, `Remove scenario ${fmt.int(i)}`)}>
                        ✕
                      </button>
                    </span>
                  )}
                </th>
              ))}
            </tr>
            {comparison.workloadsDiffer && (
              <tr className="compare-workloads">
                <th scope="row">{t("Charge et cibles", "Workload and targets")}</th>
                {columns.map((d, i) => <td key={i}>{workloadLabel(d.scenario.workload)}</td>)}
              </tr>
            )}
          </thead>
          <tbody>
            {rows.filter((r) => !hideIdentical || r.different).map((r) => (
              <tr key={r.id} data-metric={r.id} className={r.configuration && r.different ? "compare-different" : undefined}>
                <th scope="row">{t(...r.label)}</th>
                {columns.map((d, i) => (
                  <td key={i} className={r.best.includes(i) ? "compare-best" : !d.valid ? "muted" : undefined} title={r.best.includes(i) ? t("Meilleure valeur", "Best value") : !d.valid ? t("Configuration indisponible ou invalide : exclue du classement", "Unavailable or invalid configuration: excluded from ranking") : undefined}>
                    {!d.input ? t("indisponible", "unavailable") : <>{cell(r.values[i], r.format)}{r.id === "sessions" && !d.valid ? t(" (invalide)", " (invalid)") : ""}</>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {hideIdentical && !rows.some((r) => r.different) && <p className="hint">{t("Toutes les lignes sont identiques.", "All rows are identical.")}</p>}
    </div>
  );
}
