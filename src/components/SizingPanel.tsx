"use client";

import { useMemo, useState } from "react";
import { contextLabel, type EstimateInput, type ParallelMode } from "@/lib/calc";
import type { AppConfig } from "@/lib/config-schema";
import { GPU_GROUPS, gpuGroupOf, type GpuGroupId } from "@/lib/gpu-groups";
import { maxContextFor, sizeForTarget } from "@/lib/planning";
import { usePrefs } from "./Prefs";

export interface SizingChoice {
  gpuId: string;
  gpuCount: number;
  mode: ParallelMode;
  tp: number;
  wf: string;
  kf: string;
}

export function SizingPanel({ config, input, onApply }: { config: AppConfig; input: EstimateInput; onApply: (c: SizingChoice) => void }) {
  const { lang, t, fmt } = usePrefs();
  const [target, setTarget] = useState(32);
  const [group, setGroup] = useState<GpuGroupId | "all">("all");
  const rows = useMemo(
    () => sizeForTarget(config, { ...input, weightFormatId: input.weightFormat.id, kvFormatId: input.kvFormat.id }, target, input.gpu.id),
    [config, input, target],
  );
  const maxCtx = useMemo(() => maxContextFor(config, input, target), [config, input, target]);

  return (
    <div>
      <h3>{t("Dimensionnement inverse", "Inverse sizing")}</h3>
      <div className="inline-form">
        <label htmlFor="target">{t("Objectif", "Target")}</label>
        <input id="target" type="number" min={1} max={100000} value={target} onChange={(e) => setTarget(Math.max(1, Number(e.target.value) || 1))} />
        <span>
          {t("sessions simultanées à", "concurrent sessions at")} <strong>{contextLabel(input.contextTokens)}</strong> {t("avec", "with")} {input.model.name}
        </span>
      </div>
      <p className="hint">
        {t(
          "Pour chaque carte : plus petit nombre de GPUs qui atteint l'objectif, en testant tous les degrés de TP valides (complétés par du PP) et la DP-attention pour les MoE ; à égalité de cartes, la configuration au meilleur débit à l'objectif est retenue. Les formats sont conservés quand la carte les supporte. Cliquez une ligne pour l'appliquer.",
          "For each card: smallest number of GPUs that reaches the target, testing every valid TP degree (completed with PP) and DP-attention for MoE models; for the same number of cards, the configuration with the best throughput at the target is kept. Formats are kept when the card supports them. Click a row to apply it.",
        )}
      </p>
      <div className="filter-row">
        {(["all", ...GPU_GROUPS.map((g) => g.id)] as const).map((id) => (
          <button key={id} type="button" className={`btn${group === id ? " primary" : ""}`} onClick={() => setGroup(id)}>
            {id === "all" ? t("Toutes", "All") : GPU_GROUPS.find((g) => g.id === id)![lang]}
          </button>
        ))}
      </div>
      <div className="table-wrap">
        <table className="table compact clickable">
          <thead>
            <tr>
              <th>{t("Carte", "Card")}</th>
              <th className="num">GPUs</th>
              <th>{t("Parallélisme", "Parallelism")}</th>
              <th className="num">Sessions</th>
              <th className="num">{t("Débit à l'objectif", "Throughput at target")}</th>
              <th>{t("Remarque", "Note")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.filter((r) => group === "all" || gpuGroupOf(r.gpu) === group).map((r) => (
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
                <td className="num">{r.gpuCount === null ? "—" : fmt.int(r.sessions)}</td>
                <td className="num">{r.tokPerSecTotal ? fmt.rate(r.tokPerSecTotal) : "—"}</td>
                <td className="muted">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">
        {t(`Sur la configuration actuelle (${input.gpuCount} × ${input.gpu.name}) : `, `With the current configuration (${input.gpuCount} × ${input.gpu.name}): `)}
        {maxCtx ? (
          <>
            {t(`contexte maximal pour ${fmt.int(target)} sessions = `, `maximum context for ${fmt.int(target)} sessions = `)}
            <strong>{contextLabel(maxCtx)}</strong>.
          </>
        ) : (
          <>{t("objectif inatteignable sans erreur, même au plus petit palier.", "target unreachable without errors, even at the smallest preset.")}</>
        )}
      </p>
    </div>
  );
}
