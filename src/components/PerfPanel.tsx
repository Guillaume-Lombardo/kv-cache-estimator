"use client";

import { useMemo } from "react";
import { contextLabel, type Estimate, type EstimateInput } from "@/lib/calc";
import type { PerformanceSettings } from "@/lib/config-schema";
import { estimatePerf, type DecodePoint } from "@/lib/perf";
import { usePrefs } from "./Prefs";

export function PerfPanel({ input, est, perf }: { input: EstimateInput; est: Estimate; perf: PerformanceSettings }) {
  const { t, fmt } = usePrefs();
  const p = useMemo(() => estimatePerf(input, est, perf), [input, est, perf]);
  const boundLabel = (b: DecodePoint["bound"]) => (b === "memory" ? t("mémoire", "memory") : t("calcul", "compute"));
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
      <h3>{t("Débit de décodage estimé", "Estimated decoding throughput")}</h3>
      <div className="perf-grid">
        <div className="perf-card">
          <span className="perf-label">{t("1 session seule", "1 session alone")}</span>
          <strong>{fmt.rate(p.single.tokPerSecPerSession)}</strong>
          <span className="muted">
            {fmt.num(p.single.stepMs)} ms/token ·{" "}
            {p.single.bound === "memory" ? t("limité par la mémoire", "memory-bound") : t("limité par le calcul", "compute-bound")}
          </span>
        </div>
        <div className="perf-card">
          <span className="perf-label">
            {t("Pleine charge", "Full load")} ({fmt.int(est.sessions)} sessions)
          </span>
          <strong>{p.full ? fmt.rate(p.full.tokPerSecTotal) : "—"}</strong>
          <span className="muted">
            {p.full
              ? `${fmt.rate(p.full.tokPerSecPerSession)} ${t("par session", "per session")} · ${
                  p.full.bound === "memory" ? t("limité par le transfert mémoire", "memory-transfer-bound") : t("limité par le calcul", "compute-bound")
                }`
              : t("aucune session", "no session")}
          </span>
        </div>
        <div className="perf-card">
          <span className="perf-label">{t(`Prefill de ${contextLabel(input.contextTokens)} tokens`, `Prefill of ${contextLabel(input.contextTokens)} tokens`)}</span>
          <strong>{fmt.duration(p.prefillSeconds)}</strong>
          <span className="muted">{t(`1 requête, calcul ${p.computeLabel}`, `1 request, ${p.computeLabel} compute`)}</span>
        </div>
      </div>
      {rows.length > 1 && (
        <div className="table-wrap">
          <table className="table compact">
            <thead>
              <tr>
                <th className="num">{t("Sessions actives", "Active sessions")}</th>
                <th className="num">{t("tok/s par session", "tok/s per session")}</th>
                <th className="num">{t("tok/s agrégés", "Aggregate tok/s")}</th>
                <th>{t("Limite", "Bound")}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.batch}>
                  <td className="num">{fmt.int(r.batch)}</td>
                  <td className="num">{fmt.rate(r.tokPerSecPerSession)}</td>
                  <td className="num">{fmt.rate(r.tokPerSecTotal)}</td>
                  <td className="muted">{boundLabel(r.bound)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="hint">
        {t(
          `Ordre de grandeur, contexte plein pour chaque session : bande passante × ${perf.memoryEfficiency}, calcul × ${perf.computeEfficiency} (MFU), latence des collectives ${perf.collectiveLatencyUs.nvlink} µs (NVLink) / ${perf.collectiveLatencyUs.pcie} µs (PCIe). Pas de décodage spéculatif, ni de prefill concurrent. Attention parcimonieuse (DSA) non prise en compte dans le prefill. Paramètres réglables dans `,
          `Order of magnitude, full context for every session: bandwidth × ${perf.memoryEfficiency}, compute × ${perf.computeEfficiency} (MFU), collective latency ${perf.collectiveLatencyUs.nvlink} µs (NVLink) / ${perf.collectiveLatencyUs.pcie} µs (PCIe). No speculative decoding or concurrent prefill. Sparse attention (DSA) not taken into account in prefill. Settings can be adjusted in `,
        )}
        <code>performance</code>
        {t(" du YAML.", " in the YAML.")}
      </p>

      {est.offload.totalBytes > 0 && (
        <>
          <h3>{t("Offload du KV vers la mémoire CPU", "KV offload to CPU memory")}</h3>
          <p>
            {fmt.bytes(est.offload.totalBytes)} {t("côté CPU : ", "on the CPU side: ")}
            <strong>
              {fmt.int(est.offload.parkedSessions)} sessions
            </strong>{" "}
            {t(
              `en réserve en plus des ${fmt.int(est.sessions)} actives. Rechargement d'une session : ≈ ${fmt.duration(est.offload.reloadMs / 1000)} via ${input.gpu.hostLinkLabel}.`,
              `parked in addition to the ${fmt.int(est.sessions)} active ones. Reloading one session: ≈ ${fmt.duration(est.offload.reloadMs / 1000)} over ${input.gpu.hostLinkLabel}.`,
            )}
          </p>
        </>
      )}
    </div>
  );
}
