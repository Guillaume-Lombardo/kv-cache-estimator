"use client";

import { useMemo, useState } from "react";
import { contextLabel, type EstimateInput, type ParallelMode } from "@/lib/calc";
import type { AppConfig } from "@/lib/config-schema";
import { fmtInt, fmtRate } from "@/lib/format";
import { maxContextFor, sizeForTarget } from "@/lib/planning";

export interface SizingChoice {
  gpuId: string;
  gpuCount: number;
  mode: ParallelMode;
  tp: number;
  wf: string;
  kf: string;
}

export function SizingPanel({ config, input, onApply }: { config: AppConfig; input: EstimateInput; onApply: (c: SizingChoice) => void }) {
  const [target, setTarget] = useState(32);
  const rows = useMemo(
    () => sizeForTarget(config, { ...input, weightFormatId: input.weightFormat.id, kvFormatId: input.kvFormat.id }, target, input.gpu.id),
    [config, input, target],
  );
  const maxCtx = useMemo(() => maxContextFor(config, input, target), [config, input, target]);

  return (
    <div>
      <h3>Dimensionnement inverse</h3>
      <div className="inline-form">
        <label htmlFor="target">Objectif</label>
        <input id="target" type="number" min={1} max={100000} value={target} onChange={(e) => setTarget(Math.max(1, Number(e.target.value) || 1))} />
        <span>
          sessions simultanées à <strong>{contextLabel(input.contextTokens)}</strong> avec {input.model.name}
        </span>
      </div>
      <p className="hint">
        Pour chaque carte : plus petit nombre de GPUs qui atteint l&apos;objectif, en testant tous les degrés de TP valides (complétés par du PP) et la DP-attention
        pour les MoE ; à égalité de cartes, la configuration au meilleur débit à l&apos;objectif est retenue. Les formats sont conservés quand la carte les supporte.
        Cliquez une ligne pour l&apos;appliquer.
      </p>
      <div className="table-wrap">
        <table className="table compact clickable">
          <thead>
            <tr>
              <th>Carte</th>
              <th className="num">GPUs</th>
              <th>Parallélisme</th>
              <th className="num">Sessions</th>
              <th className="num">Débit à l&apos;objectif</th>
              <th>Remarque</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.gpu.id}
                className={r.gpuCount === null ? "row-disabled" : r.gpu.id === input.gpu.id ? "row-current" : ""}
                onClick={() =>
                  r.gpuCount !== null && onApply({ gpuId: r.gpu.id, gpuCount: r.gpuCount, mode: r.mode, tp: r.tp, wf: r.weightFormat, kf: r.kvFormat })
                }
              >
                <td>{r.gpu.name}</td>
                <td className="num">{r.gpuCount ?? "—"}</td>
                <td>{r.gpuCount === null ? "—" : r.mode === "dp-ep" ? "DP-attention + EP" : r.gpuCount === 1 ? "—" : `TP ${r.tp} × PP ${r.gpuCount / r.tp}`}</td>
                <td className="num">{r.gpuCount === null ? "—" : fmtInt(r.sessions)}</td>
                <td className="num">{r.tokPerSecTotal ? fmtRate(r.tokPerSecTotal) : "—"}</td>
                <td className="muted">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">
        Sur la configuration actuelle ({input.gpuCount} × {input.gpu.name}) :{" "}
        {maxCtx ? (
          <>
            contexte maximal pour {fmtInt(target)} sessions = <strong>{contextLabel(maxCtx)}</strong>.
          </>
        ) : (
          <>objectif inatteignable sans erreur, même à {contextLabel(8192)}.</>
        )}
      </p>
    </div>
  );
}
