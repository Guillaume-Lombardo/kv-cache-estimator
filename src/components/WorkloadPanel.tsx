"use client";

import { useMemo, useState } from "react";
import type { EstimateInput } from "@/lib/calc";
import type { AppConfig } from "@/lib/config-schema";
import { GPU_GROUPS, gpuGroupOf, type GpuGroupId } from "@/lib/gpu-groups";
import type { Workload } from "@/lib/scenario";
import { analyzeWorkload, sizeForWorkload, type Cost, type Infeasible } from "@/lib/workload";
import { usePrefs } from "./Prefs";
import type { SizingChoice } from "./SizingPanel";

const INFEASIBLE: Record<Infeasible, { fr: string; en: string }> = {
  "request-exceeds-context": {
    fr: "Une requête moyenne (prompt + sortie) dépasse le contexte maximal configuré.",
    en: "An average request (prompt + output) exceeds the configured maximum context.",
  },
  "no-memory": { fr: "Aucune requête moyenne ne tient dans la mémoire KV.", en: "No average request fits in KV memory." },
  "ttft-unreachable": {
    fr: "Le prefill d'un seul prompt dépasse déjà le TTFT cible sur cette instance : ajoutez du TP ou une carte plus rapide.",
    en: "Prefilling a single prompt already exceeds the TTFT target on this instance: add TP or use a faster card.",
  },
  "itl-unreachable": {
    fr: "Même une seule session décode plus lentement que l'ITL cible : ajoutez du TP, quantifiez ou prenez une carte à plus forte bande passante.",
    en: "Even a single session decodes slower than the ITL target: add TP, quantize or use a card with more bandwidth.",
  },
};

export function WorkloadPanel({
  config,
  input,
  workload,
  priceUsd,
  onChange,
  onApply,
}: {
  config: AppConfig;
  input: EstimateInput;
  workload: Workload;
  priceUsd?: number;
  onChange: (w: Workload, priceUsd: number | undefined) => void;
  onApply: (c: SizingChoice) => void;
}) {
  const { lang, t, fmt } = usePrefs();
  const [group, setGroup] = useState<GpuGroupId | "all">("all");
  const price = priceUsd ?? input.gpu.priceUsdPerHour;
  const a = useMemo(() => analyzeWorkload(config, input, workload, price), [config, input, workload, price]);
  const rows = useMemo(
    () => sizeForWorkload(config, { ...input, weightFormatId: input.weightFormat.id, kvFormatId: input.kvFormat.id }, workload, input.gpu.id, priceUsd),
    [config, input, workload, priceUsd],
  );
  const shown = rows.filter((r) => group === "all" || gpuGroupOf(r.gpu) === group);

  const num = (key: keyof Workload, label: string, unit: string, step = 1) => (
    <label>
      {label}
      <input
        type="number"
        min={0}
        step={step}
        value={workload[key]}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (v > 0) onChange({ ...workload, [key]: key === "inputTokens" || key === "outputTokens" ? Math.round(v) : v }, priceUsd);
        }}
      />
      <span className="muted">{unit}</span>
    </label>
  );
  const usd = (x: number) => (lang === "fr" ? `${fmt.num(x)} $` : `$${fmt.num(x)}`);
  const costText = (c: Cost | null) => (c ? `${usd(c.usdPerHour)}/h · ${usd(c.usdPerMillionOutput)}/M ${t("sortie", "out")}` : "—");
  const mark = (bad: boolean) => <span className={bad ? "slo-bad" : "slo-ok"}>{bad ? "✕" : "✓"}</span>;
  const parallel = (mode: string, n: number, tp: number) => (mode === "dp-ep" ? "DP-attention + EP" : `TP ${tp} × PP ${n / tp}`);
  /** "3 × 2 GPUs · TP 2 × PP 1", "1 GPU". */
  const shape = (replicas: number, n: number, mode: string, tp: number) =>
    `${replicas > 1 ? `${replicas} × ` : ""}${n} GPU${n > 1 ? "s" : ""}${n > 1 ? ` · ${parallel(mode, n, tp)}` : ""}`;
  const s = a.single;
  const agg = a.aggregated;
  const dis = a.disaggregated;
  const n = input.gpuCount;

  return (
    <div>
      <h3>{t("Charge et objectifs de latence", "Load and latency targets")}</h3>
      <div className="workload-form">
        {num("inputTokens", t("Prompt moyen", "Average prompt"), "tokens", 100)}
        {num("outputTokens", t("Sortie moyenne", "Average output"), "tokens", 50)}
        {num("requestsPerSec", t("Débit de requêtes", "Request rate"), t("req/s", "req/s"), 0.5)}
        {num("ttftSeconds", t("TTFT cible", "TTFT target"), t("s (premier token)", "s (first token)"), 0.1)}
        {num("itlMs", t("ITL cible", "ITL target"), t("ms entre tokens", "ms between tokens"), 5)}
        <label>
          {t("Prix par GPU", "Price per GPU")}
          <input
            type="number"
            min={0}
            step={0.1}
            value={priceUsd ?? ""}
            placeholder={input.gpu.priceUsdPerHour ? String(input.gpu.priceUsdPerHour) : "—"}
            onChange={(e) => onChange(workload, Number(e.target.value) > 0 ? Number(e.target.value) : undefined)}
          />
          <span className="muted">{t(`USD/h (${input.gpu.name})`, `USD/h (${input.gpu.name})`)}</span>
        </label>
      </div>

      <h3>
        {t("Instance actuelle", "Current instance")} ({n} × {input.gpu.name}
        {n > 1 ? `, ${parallel(input.mode, n, input.tp)}` : ""})
      </h3>
      {a.infeasible ? (
        <p className="hint lim-un">⛔ {INFEASIBLE[a.infeasible][lang]}</p>
      ) : (
        <div className="perf-grid">
          <div className="perf-card">
            <span className="perf-label">{t("Requêtes simultanées", "Concurrent requests")}</span>
            <strong>
              {Number.isFinite(s.concurrency) ? fmt.int(Math.ceil(s.concurrency)) : "∞"} / {fmt.int(a.model.capacity)}
            </strong>
            <span className="muted">
              {mark(s.over.memory)}{" "}
              {t(`nécessaires / capacité à ${fmt.int(a.model.avgTokens)} tokens moyens`, `needed / capacity at ${fmt.int(a.model.avgTokens)} average tokens`)}
            </span>
          </div>
          <div className="perf-card">
            <span className="perf-label">TTFT</span>
            <strong>{Number.isFinite(s.ttftSeconds) ? fmt.duration(s.ttftSeconds) : "∞"}</strong>
            <span className="muted">
              {mark(s.over.ttft || s.over.prefill)} {t(`prefill à ${fmt.int(s.prefillUtil * 100)} % du calcul`, `prefill at ${fmt.int(s.prefillUtil * 100)}% of compute`)}
            </span>
          </div>
          <div className="perf-card">
            <span className="perf-label">ITL</span>
            <strong>{Number.isFinite(s.itlMs) ? `${fmt.num(s.itlMs)} ms` : "∞"}</strong>
            <span className="muted">
              {mark(s.over.itl)} {Number.isFinite(s.itlMs) ? fmt.rate(1000 / s.itlMs) : "—"} {t("par session", "per session")}
            </span>
          </div>
        </div>
      )}

      {!a.infeasible && (
        <div className="table-wrap">
          <table className="table compact">
            <thead>
              <tr>
                <th>{t("Déploiement", "Deployment")}</th>
                <th className="num">GPUs</th>
                <th className="num">TTFT</th>
                <th className="num">ITL</th>
                <th>{t("Coût", "Cost")}</th>
                <th>{t("Détail", "Detail")}</th>
              </tr>
            </thead>
            <tbody>
              <tr className={agg && (!dis || agg.gpus <= dis.gpus) ? "row-current" : ""}>
                <td>{t("Agrégé (prefill + decode)", "Aggregated (prefill + decode)")}</td>
                <td className="num">{agg ? fmt.int(agg.gpus) : "—"}</td>
                <td className="num">{agg ? fmt.duration(agg.load.ttftSeconds) : "—"}</td>
                <td className="num">{agg ? `${fmt.num(agg.load.itlMs)} ms` : "—"}</td>
                <td>{costText(a.cost.aggregated)}</td>
                <td className="muted">
                  {agg
                    ? t(`${agg.replicas} instance${agg.replicas > 1 ? "s" : ""} de ${n} GPU${n > 1 ? "s" : ""}`, `${agg.replicas} instance${agg.replicas > 1 ? "s" : ""} of ${n} GPU${n > 1 ? "s" : ""}`)
                    : t("objectifs inatteignables", "targets unreachable")}
                </td>
              </tr>
              <tr className={dis && agg && dis.gpus < agg.gpus ? "row-current" : ""}>
                <td>{t("Désagrégé P/D (llm-d)", "Disaggregated P/D (llm-d)")}</td>
                <td className="num">{dis ? fmt.int(dis.gpus) : "—"}</td>
                <td className="num">{dis ? fmt.duration(dis.ttftSeconds) : "—"}</td>
                <td className="num">{dis ? `${fmt.num(dis.decode.itlMs)} ms` : "—"}</td>
                <td>{costText(a.cost.disaggregated)}</td>
                <td className="muted">
                  {dis
                    ? t(
                        `${dis.prefillReplicas} prefill + ${dis.decodeReplicas} decode · KV transféré ${fmt.num(dis.kvTrafficBytesPerSec / 1e9)} Go/s (${fmt.duration(dis.kvTransferSeconds)} par prompt)`,
                        `${dis.prefillReplicas} prefill + ${dis.decodeReplicas} decode · KV transfer ${fmt.num(dis.kvTrafficBytesPerSec / 1e9)} GB/s (${fmt.duration(dis.kvTransferSeconds)} per prompt)`,
                      )
                    : t("objectifs inatteignables", "targets unreachable")}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      <h3>{t("Cartes pour cette charge", "Cards for this load")}</h3>
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
              <th className="num">{t("GPUs au total", "Total GPUs")}</th>
              <th>{t("Instance", "Instance")}</th>
              <th className="num">P/D</th>
              <th className="num">ITL</th>
              <th>{t("Coût", "Cost")}</th>
              <th>{t("Remarque", "Note")}</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr
                key={r.gpu.id}
                className={r.gpuCount === null ? "row-disabled" : r.gpu.id === input.gpu.id ? "row-current" : ""}
                onClick={() => r.gpuCount !== null && onApply({ gpuId: r.gpu.id, gpuCount: r.gpuCount, mode: r.mode, tp: r.tp, wf: r.weightFormat, kf: r.kvFormat })}
              >
                <td>{r.gpu.name}</td>
                <td className="num">{r.gpus ?? "—"}</td>
                <td>{r.gpuCount === null ? "—" : shape(r.replicas, r.gpuCount, r.mode, r.tp)}</td>
                <td className="num">{r.disaggregatedGpus ?? "—"}</td>
                <td className="num">{r.itlMs === null ? "—" : `${fmt.num(r.itlMs)} ms`}</td>
                <td>{costText(r.cost)}</td>
                <td className="muted">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">
        {t(
          "Modèle au premier ordre, sans file d'attente : chaque requête garde en moyenne prompt + sortie/2 tokens de KV ; les prefills prennent une part u du calcul et ralentissent les étapes limitées par le calcul de 1/(1−u) ; TTFT = prefill seul/(1−u) ; concurrence = débit × (TTFT + sortie × ITL). Les instances identiques se partagent le débit derrière un routeur. En P/D, le pool prefill transfère le KV du prompt au pool decode par le réseau (",
          "First-order model without queueing: each request holds prompt + output/2 KV tokens on average; prefills take a share u of compute and slow compute-bound steps by 1/(1−u); TTFT = single prefill/(1−u); concurrency = rate × (TTFT + output × ITL). Identical instances share the rate behind a router. In P/D, the prefill pool sends the prompt KV to the decode pool over the network (",
        )}
        {config.performance.networkGBsPerGpu} {t("Go/s par GPU). Prix indicatifs, réglables dans le YAML.", "GB/s per GPU). Indicative prices, adjustable in the YAML.")}
      </p>
    </div>
  );
}
