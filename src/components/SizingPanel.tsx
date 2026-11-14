"use client";

import { useMemo, useState } from "react";
import { contextLabel, type EstimateInput, type ParallelMode } from "@/lib/calc";
import type { AppConfig } from "@/lib/config-schema";
import { GPU_GROUPS, gpuGroupOf, type GpuGroupId } from "@/lib/gpu-groups";
import type { Workload } from "@/lib/scenario";
import { sizeForDeployment, type SizingObjective, type SizingSort } from "@/lib/workload";
import { usePrefs } from "./Prefs";
import { WorkloadPanel } from "./WorkloadPanel";

export interface SizingChoice {
  gpuId: string;
  gpuCount: number;
  mode: ParallelMode;
  tp: number;
  wf: string;
  kf: string;
}

export function SizingPanel({ config, input, workload, priceUsd, onChange, onApply, objective, onObjectiveChange, target, onTargetChange, loadFirst = false }: {
  config: AppConfig;
  input: EstimateInput;
  workload: Workload;
  priceUsd?: number;
  onChange: (w: Workload, priceUsd: number | undefined) => void;
  onApply: (c: SizingChoice) => void;
  objective: SizingObjective;
  onObjectiveChange: (o: SizingObjective) => void;
  target: number;
  onTargetChange: (n: number) => void;
  loadFirst?: boolean;
}) {
  const { lang, t, fmt } = usePrefs();
  const [group, setGroup] = useState<GpuGroupId | "all">("all");
  const [sort, setSort] = useState<SizingSort>("cost");
  const rows = useMemo(() => sizeForDeployment(config,
    { ...input, weightFormatId: input.weightFormat.id, kvFormatId: input.kvFormat.id }, workload,
    { objective, targetSessions: target, sort, baseGpuId: input.gpu.id, priceOverride: priceUsd }),
  [config, input, workload, objective, target, sort, priceUsd]);

  const analysis = objective === "load" && <WorkloadPanel config={config} input={input} workload={workload} priceUsd={priceUsd} onChange={onChange} hideFields={loadFirst} />;
  return (
    <div className="unified-sizing">
      <div className="inline-form">
        <label htmlFor="sizing-objective">{t("Objectif", "Objective")}</label>
        <select id="sizing-objective" value={objective} onChange={(e) => onObjectiveChange(e.target.value as SizingObjective)}>
          <option value="load">{t("Charge et latence", "Load and latency")}</option>
          <option value="sessions">{t("Sessions simultanées", "Concurrent sessions")}</option>
        </select>
      </div>
      {objective === "sessions" && <>
        <label className="inline-form" htmlFor="target">
          {t("Sessions simultanées", "Concurrent sessions")}
          <input id="target" type="number" min={1} max={100000} value={target} onChange={(e) => onTargetChange(Math.max(1, Math.ceil(Number(e.target.value) || 1)))} />
          <span>{t("au contexte courant", "at current context")} <strong>{contextLabel(input.contextTokens)}</strong></span>
        </label>
        <p className="hint">{t("Pire cas : sessions au contexte maximal, réparties entre des répliques identiques. TTFT : prefill d'un contexte entier ; ITL et coût par token : décodage à la concurrence demandée, sans SLO ni trafic imposés.", "Worst case: full-context sessions shared across identical replicas. TTFT: prefill of a full context; ITL and token cost: decoding at the requested concurrency, without SLOs or an imposed arrival rate.")}</p>
      </>}
      {!loadFirst && analysis}
      <h3>{t("Cartes pour cet objectif", "Cards for this objective")}</h3>
      <div className="filter-row">
        {(["all", ...GPU_GROUPS.map((g) => g.id)] as const).map((id) => (
          <button key={id} type="button" className={`btn${group === id ? " primary" : ""}`} onClick={() => setGroup(id)}>
            {id === "all" ? t("Toutes", "All") : GPU_GROUPS.find((g) => g.id === id)![lang]}
          </button>
        ))}
        <label className="sizing-sort">{t("Trier par", "Sort by")} <select aria-label={t("Trier par", "Sort by")} value={sort} onChange={(e) => setSort(e.target.value as SizingSort)}>
          <option value="cost">{t("Coût", "Cost")}</option><option value="gpus">GPUs</option>
        </select></label>
      </div>
      <div className="table-wrap">
        <table className="table compact clickable sizing-table">
          <thead><tr>
            <th>{t("Carte", "Card")}</th><th className="num">{t("GPUs au total", "Total GPUs")}</th><th>{t("Instance", "Instance")}</th>
            <th className="num">{objective === "sessions" ? "Sessions" : t("Requêtes simultanées", "Concurrent requests")}</th>
            <th className="num">TTFT</th><th className="num">ITL</th><th className="num">$/h</th><th className="num">{t("$/M tokens sortie", "$/M output tokens")}</th><th>{t("Remarque", "Note")}</th>
          </tr></thead>
          <tbody>{rows.filter((r) => group === "all" || gpuGroupOf(r.gpu) === group).map((r) => {
            const apply = () => r.gpuCount !== null && onApply({ gpuId: r.gpu.id, gpuCount: r.gpuCount, mode: r.mode, tp: r.tp, wf: r.weightFormat, kf: r.kvFormat });
            return <tr key={r.gpu.id} className={r.gpuCount === null ? "row-disabled" : r.gpu.id === input.gpu.id ? "row-current" : ""} onClick={apply}>
              <td>{r.gpuCount === null ? r.gpu.name : <button className="link sizing-apply">{r.gpu.name}</button>}</td>
              <td className="num">{r.gpus ?? "—"}</td>
              <td>{r.gpuCount === null ? "—" : <>{r.replicas} × {r.gpuCount} GPUs<br /><span className="muted">{r.mode === "dp-ep" ? "DP-attention + EP" : `TP ${r.tp} × PP ${r.gpuCount / r.tp}`}</span></>}</td>
              <td className="num">{r.gpuCount === null ? "—" : `${fmt.int(Math.ceil(r.concurrency))} / ${fmt.int(r.capacity)}`}</td>
              <td className="num">{r.ttftSeconds === null ? "—" : fmt.duration(r.ttftSeconds)}</td>
              <td className="num">{r.itlMs === null ? "—" : `${fmt.num(r.itlMs)} ms`}</td>
              <td className="num">{r.cost ? fmt.num(r.cost.usdPerHour) : "—"}</td>
              <td className="num">{r.cost ? fmt.num(r.cost.usdPerMillionOutput) : "—"}</td>
              <td className="muted">{r.note}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>
      <p className="hint">{t("Capacité : nécessaire / disponible sur toutes les répliques. Prix indicatifs ; prix absents en fin de tri par coût. À GPUs et coût égaux, préférence au TP sans PP. Cliquez une carte pour appliquer la forme d'une instance ; les répliques restent un plan de déploiement.", "Capacity: required / available across replicas. Indicative prices; missing prices follow known prices in cost order. At equal GPU count and cost, TP without PP is preferred. Click a card to apply one instance shape; replicas remain a deployment plan.")}</p>
      {loadFirst && analysis}
    </div>
  );
}
