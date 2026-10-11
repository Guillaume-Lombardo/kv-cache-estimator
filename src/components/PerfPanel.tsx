"use client";

import { useMemo } from "react";
import { contextLabel, type Estimate, type EstimateInput } from "@/lib/calc";
import type { PerformanceSettings } from "@/lib/config-schema";
import { fmtBytes, fmtDuration, fmtInt, fmtNum, fmtRate } from "@/lib/format";
import { estimatePerf } from "@/lib/perf";

export function PerfPanel({ input, est, perf }: { input: EstimateInput; est: Estimate; perf: PerformanceSettings }) {
  const p = useMemo(() => estimatePerf(input, est, perf), [input, est, perf]);
  // Throughput/concurrency curve: powers of 4 up to the maximum session count.
  const batches = useMemo(() => {
    const out: number[] = [];
    for (let b = 1; b < est.sessions; b *= 4) out.push(b);
    if (est.sessions >= 1) out.push(est.sessions);
    return out;
  }, [est.sessions]);
  const rows = useMemo(() => batches.map((b) => p.at(b)), [batches, p]);

  return (
    <div>
      <h3>Débit de décodage estimé</h3>
      <div className="perf-grid">
        <div className="perf-card">
          <span className="perf-label">1 session seule</span>
          <strong>{fmtRate(p.single.tokPerSecPerSession)}</strong>
          <span className="muted">
            {fmtNum(p.single.stepMs)} ms/token · limité par la {p.single.bound}
          </span>
        </div>
        <div className="perf-card">
          <span className="perf-label">Pleine charge ({fmtInt(est.sessions)} sessions)</span>
          <strong>{p.full ? fmtRate(p.full.tokPerSecTotal) : "—"}</strong>
          <span className="muted">{p.full ? `${fmtRate(p.full.tokPerSecPerSession)} par session · limité par le ${p.full.bound === "mémoire" ? "transfert mémoire" : "calcul"}` : "aucune session"}</span>
        </div>
        <div className="perf-card">
          <span className="perf-label">Prefill de {contextLabel(input.contextTokens)} tokens</span>
          <strong>{fmtDuration(p.prefillSeconds)}</strong>
          <span className="muted">1 requête, calcul {p.computeLabel}</span>
        </div>
      </div>
      {rows.length > 1 && (
        <div className="table-wrap">
          <table className="table compact">
            <thead>
              <tr>
                <th className="num">Sessions actives</th>
                <th className="num">tok/s par session</th>
                <th className="num">tok/s agrégés</th>
                <th>Limite</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.batch}>
                  <td className="num">{fmtInt(r.batch)}</td>
                  <td className="num">{fmtRate(r.tokPerSecPerSession)}</td>
                  <td className="num">{fmtRate(r.tokPerSecTotal)}</td>
                  <td className="muted">{r.bound}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="hint">
        Ordre de grandeur, contexte plein pour chaque session : bande passante × {perf.memoryEfficiency}, calcul × {perf.computeEfficiency} (MFU), latence des
        collectives {perf.collectiveLatencyUs.nvlink} µs (NVLink) / {perf.collectiveLatencyUs.pcie} µs (PCIe). Pas de décodage spéculatif, ni de prefill
        concurrent. Attention parcimonieuse (DSA) non prise en compte dans le prefill. Paramètres réglables dans <code>performance</code> du YAML.
      </p>

      {est.offload.totalBytes > 0 && (
        <>
          <h3>Offload du KV vers la mémoire CPU</h3>
          <p>
            {fmtBytes(est.offload.totalBytes)} côté CPU : <strong>{fmtInt(est.offload.parkedSessions)} sessions</strong> en réserve en plus des{" "}
            {fmtInt(est.sessions)} actives. Rechargement d&apos;une session : ≈ {fmtDuration(est.offload.reloadMs / 1000)} via {input.gpu.hostLinkLabel}.
          </p>
        </>
      )}
    </div>
  );
}
