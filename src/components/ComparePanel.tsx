"use client";

import { collectIssues, contextLabel, estimate } from "@/lib/calc";
import type { AppConfig, ModelSpec } from "@/lib/config-schema";
import { fmtBytes, fmtInt, fmtRate } from "@/lib/format";
import { estimatePerf } from "@/lib/perf";
import { resolveScenario, type Scenario } from "@/lib/scenario";

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
  const cols: Column[] = [{ key: "current", scenario: current, current: true }, ...pinned.map((s, i) => ({ key: `p${i}`, scenario: s, current: false }))];
  const data = cols.map((c) => {
    const input = resolveScenario(config, models, c.scenario);
    if (!input) return { c, input: null };
    const est = estimate(input);
    const perf = estimatePerf(input, est, config.performance);
    const issues = collectIssues(input, est, config.weightFormats);
    return { c, input, est, perf, errors: issues.filter((i) => i.severity === "error").length, warnings: issues.filter((i) => i.severity === "warning").length };
  });
  const best = Math.max(...data.map((d) => d.est?.sessions ?? 0));

  const rows: { label: string; cell: (d: (typeof data)[number]) => React.ReactNode }[] = [
    { label: "Modèle", cell: (d) => d.input?.model.name },
    { label: "Cartes", cell: (d) => d.input && `${d.input.gpuCount} × ${d.input.gpu.name}` },
    { label: "Moteur", cell: (d) => d.input?.engine.name },
    { label: "Parallélisme", cell: (d) => d.est && (d.input?.mode === "dp-ep" ? "DP-attention + EP" : d.input?.gpuCount === 1 ? "—" : `TP ${d.est.tp} × PP ${d.est.pp}`) },
    { label: "Poids / KV", cell: (d) => d.input && `${d.input.weightFormat.label} / ${d.input.kvFormat.label}` },
    { label: "Contexte", cell: (d) => d.input && contextLabel(d.input.contextTokens) },
    { label: "Poids du modèle", cell: (d) => d.est && fmtBytes(d.est.weights.totalBytes) },
    { label: "KV par requête", cell: (d) => d.est && fmtBytes(d.est.kv.perRequestBytes) },
    { label: "KV disponible", cell: (d) => d.est && fmtBytes(d.est.memory.freeForKvTotalBytes) },
    {
      label: "Sessions",
      cell: (d) => d.est && <strong className={d.est.sessions === best && best > 0 ? "best" : ""}>{fmtInt(d.est.sessions)}</strong>,
    },
    { label: "Débit 1 session", cell: (d) => d.perf && fmtRate(d.perf.single.tokPerSecPerSession) },
    { label: "Débit agrégé", cell: (d) => (d.perf?.full ? fmtRate(d.perf.full.tokPerSecTotal) : "—") },
    { label: "Statut", cell: (d) => d.est && (d.errors ? `⛔ ${d.errors}` : d.warnings ? `⚠️ ${d.warnings}` : "✓") },
  ];

  return (
    <div>
      <div className="row-between">
        <h3>Comparaison de scénarios</h3>
        <button type="button" className="btn" onClick={onPin} disabled={pinned.length >= 6}>
          📌 Épingler la configuration actuelle
        </button>
      </div>
      {pinned.length === 0 && <p className="hint">Épinglez des configurations (jusqu&apos;à 6) pour les comparer à la configuration en cours. Elles sont gardées dans ce navigateur.</p>}
      <div className="table-wrap">
        <table className="table compact compare">
          <thead>
            <tr>
              <th />
              {data.map((d, i) => (
                <th key={d.c.key}>
                  {d.c.current ? (
                    "Actuelle"
                  ) : (
                    <span className="col-actions">
                      #{i}
                      <button type="button" className="link" onClick={() => onLoad(d.c.scenario)}>
                        charger
                      </button>
                      <button type="button" className="link" onClick={() => onRemove(i - 1)} aria-label="Retirer">
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
                  <td key={d.c.key}>{d.input ? r.cell(d) : <span className="muted">indisponible</span>}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
