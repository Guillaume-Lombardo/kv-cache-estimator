"use client";

import { useEffect, useMemo, useState } from "react";
import type { AppConfig, GpuSpec, ModelSpec, Support } from "@/lib/config-schema";
import {
  collectIssues,
  CONTEXT_STEPS,
  contextLabel,
  estimate,
  isMoE,
  PARALLELISM_LABELS,
  supportLevelLabel,
  supportOf,
  type EstimateInput,
  type Issue,
  type Parallelism,
} from "@/lib/calc";
import { fmtBytes, fmtInt, fmtParams, fmtTokens } from "@/lib/format";
import { ChipGroup, type ChipOption } from "./ChipGroup";
import { archLabel, ModelCombobox, TAG_LABELS } from "./ModelCombobox";

export interface InitialState {
  model?: string;
  ctx?: number;
  gpu?: string;
  n?: number;
  w?: string;
  kv?: string;
  r?: number;
  p?: string;
}

interface State {
  modelId: string;
  ctxIdx: number;
  gpuId: string;
  gpuCount: number;
  wf: string;
  kf: string;
  reservePct: number;
  par: Parallelism;
}

const usable = (s: Support) => s.level !== "unsupported";

/** Format de poids : natif du modèle si possible, sinon le plus proche disponible sur le GPU. */
function pickWeightFormat(cfg: AppConfig, m: ModelSpec, g: GpuSpec, current?: string): string {
  if (current && usable(supportOf(g.weightFormats, current))) return current;
  const native = m.nativeWeightFormat;
  if (usable(supportOf(g.weightFormats, native))) return native;
  // Repli : BF16 → FP16 sur Volta, sinon premier format officiel puis premier format natif du GPU.
  if (native === "bf16" && usable(supportOf(g.weightFormats, "fp16"))) return "fp16";
  const official = m.officialWeightFormats.find((f) => supportOf(g.weightFormats, f).level === "native");
  if (official) return official;
  return cfg.weightFormats.find((f) => supportOf(g.weightFormats, f.id).level === "native")?.id ?? cfg.weightFormats[0].id;
}

function pickKvFormat(cfg: AppConfig, g: GpuSpec, current?: string): string {
  if (current && usable(supportOf(g.kvCacheFormats, current))) return current;
  for (const id of ["bf16", "fp16"]) if (supportOf(g.kvCacheFormats, id).level === "native") return id;
  return cfg.kvCacheFormats.find((f) => usable(supportOf(g.kvCacheFormats, f.id)))?.id ?? cfg.kvCacheFormats[0].id;
}

function nearestCtxIdx(tokens: number | undefined): number {
  if (!tokens) return 2;
  let best = 0;
  CONTEXT_STEPS.forEach((t, i) => {
    if (Math.abs(Math.log2(t / tokens)) < Math.abs(Math.log2(CONTEXT_STEPS[best] / tokens))) best = i;
  });
  return best;
}

function initState(cfg: AppConfig, init: InitialState): State {
  const d = cfg.defaults;
  const model = cfg.models.find((m) => m.id === init.model) ?? cfg.models.find((m) => m.id === d.model) ?? cfg.models[0];
  const gpu = cfg.gpus.find((g) => g.id === init.gpu) ?? cfg.gpus.find((g) => g.id === d.gpu) ?? cfg.gpus[0];
  const n = init.n && init.n >= 1 && init.n <= 8 ? Math.round(init.n) : (d.gpuCount ?? 1);
  const par: Parallelism = init.p === "pp" || init.p === "dp-attention" ? init.p : "tp";
  return {
    modelId: model.id,
    gpuId: gpu.id,
    gpuCount: n,
    ctxIdx: nearestCtxIdx(init.ctx ?? d.contextTokens),
    wf: pickWeightFormat(cfg, model, gpu, init.w),
    kf: pickKvFormat(cfg, gpu, init.kv ?? d.kvCacheFormat),
    reservePct: init.r !== undefined && init.r >= 0 && init.r <= 50 ? init.r : (d.reservePct ?? 10),
    par: par === "dp-attention" && !isMoE(model) ? "tp" : par,
  };
}

const SEVERITY_ICON: Record<Issue["severity"], string> = { error: "⛔", warning: "⚠️", info: "ℹ️" };

export function Estimator({ config, configPath, initial }: { config: AppConfig; configPath: string; initial: InitialState }) {
  const [s, setS] = useState<State>(() => initState(config, initial));
  const [notices, setNotices] = useState<string[]>([]);

  const model = config.models.find((m) => m.id === s.modelId)!;
  const gpu = config.gpus.find((g) => g.id === s.gpuId)!;
  const wf = config.weightFormats.find((f) => f.id === s.wf)!;
  const kf = config.kvCacheFormats.find((f) => f.id === s.kf)!;
  const ctx = CONTEXT_STEPS[s.ctxIdx];

  // --- Changements avec ajustement automatique des formats
  const setModel = (id: string) => {
    const m = config.models.find((x) => x.id === id)!;
    const nextWf = pickWeightFormat(config, m, gpu);
    const msgs: string[] = [];
    let par = s.par;
    if (par === "dp-attention" && !isMoE(m)) {
      par = "tp";
      msgs.push("DP-attention réservé aux MoE → retour en TP.");
    }
    if (nextWf !== m.nativeWeightFormat)
      msgs.push(`Format natif ${m.nativeWeightFormat.toUpperCase()} indisponible sur ${gpu.name} → ${nextWf.toUpperCase()}.`);
    setNotices(msgs);
    setS({ ...s, modelId: id, wf: nextWf, par });
  };
  const setGpu = (id: string) => {
    const g = config.gpus.find((x) => x.id === id)!;
    let nextWf = pickWeightFormat(config, model, g, s.wf);
    let nextKf = pickKvFormat(config, g, s.kf);
    // FP16 imposé par l'absence de BF16 (Volta) : on revient en BF16 dès que possible.
    if (nextKf === "fp16" && !usable(supportOf(gpu.kvCacheFormats, "bf16")) && supportOf(g.kvCacheFormats, "bf16").level === "native")
      nextKf = "bf16";
    const msgs: string[] = [];
    const native = config.weightFormats.find((f) => f.id === model.nativeWeightFormat);
    if (nextWf === s.wf && native && wf.bitsPerParam > native.bitsPerParam && usable(supportOf(g.weightFormats, native.id))) {
      // Le format courant était un repli plus lourd que le natif (ex. FP16 forcé par un V100).
      nextWf = native.id;
      msgs.push(`Format natif ${native.label} disponible sur ${g.name} → ${native.label}.`);
    } else if (nextWf !== s.wf) msgs.push(`Poids ${wf.label} non supportés sur ${g.name} → ${nextWf.toUpperCase()}.`);
    if (nextKf !== s.kf)
      msgs.push(
        usable(supportOf(g.kvCacheFormats, s.kf))
          ? `KV BF16 disponible sur ${g.name} → BF16.`
          : `KV ${kf.label} non supporté sur ${g.name} → ${nextKf.toUpperCase()}.`,
      );
    setNotices(msgs);
    setS({ ...s, gpuId: id, wf: nextWf, kf: nextKf });
  };
  const update = (patch: Partial<State>) => {
    setNotices([]);
    setS({ ...s, ...patch });
  };

  // --- Partage de l'état dans l'URL
  useEffect(() => {
    const q = new URLSearchParams({
      model: s.modelId,
      ctx: String(ctx),
      gpu: s.gpuId,
      n: String(s.gpuCount),
      w: s.wf,
      kv: s.kf,
      r: String(s.reservePct),
      p: s.par,
    });
    window.history.replaceState(null, "", `?${q}`);
  }, [s, ctx]);

  const input: EstimateInput = {
    model,
    gpu,
    gpuCount: s.gpuCount,
    contextTokens: ctx,
    weightFormat: wf,
    kvFormat: kf,
    reservePct: s.reservePct,
    parallelism: s.par,
  };
  const est = estimate(input);
  const issues = collectIssues(input, est, config.weightFormats);

  const byContext = useMemo(
    () => CONTEXT_STEPS.map((t) => ({ t, e: estimate({ ...input, contextTokens: t }) })),
    [model, gpu, wf, kf, s.gpuCount, s.reservePct, s.par],
  );

  // --- Options
  const weightOptions: ChipOption[] = config.weightFormats.map((f) => {
    const sup = supportOf(gpu.weightFormats, f.id);
    const official = model.officialWeightFormats.includes(f.id);
    return {
      id: f.id,
      label: f.label,
      sub: `${f.bitsPerParam} bits${official ? " · officiel" : ""}`,
      level: sup.level,
      star: f.id === model.nativeWeightFormat,
      warn: model.weightFormatNotes[f.id] !== undefined,
      title: [f.description, sup.level !== "native" ? `${supportLevelLabel(sup.level)} sur ${gpu.name}` : null, sup.note, model.weightFormatNotes[f.id]?.note]
        .filter(Boolean)
        .join("\n"),
    };
  });
  const kvOptions: ChipOption[] = config.kvCacheFormats.map((f) => {
    const sup = supportOf(gpu.kvCacheFormats, f.id);
    return {
      id: f.id,
      label: f.label,
      sub: `${f.bytesPerElement * 8} bits`,
      level: sup.level,
      warn: model.kvFormatNotes[f.id] !== undefined,
      title: [f.description, sup.level !== "native" ? `${supportLevelLabel(sup.level)} sur ${gpu.name}` : null, sup.note].filter(Boolean).join("\n"),
    };
  });
  const gpuOptions: ChipOption[] = config.gpus.map((g) => ({
    id: g.id,
    label: g.name,
    sub: (
      <>
        {g.memoryLabel.split(" (")[0]}
        <br />
        <span className={g.interconnect.nvlink ? "" : "no-link"}>{g.interconnect.nvlink ? (g.nvlinkDomain > 8 ? "NVL72" : "NVLink") : "PCIe, sans NVLink"}</span>
      </>
    ),
    title: `${g.architecture} (sm ${g.computeCapability}) · ${g.memoryLabel} · ${g.interconnect.label}`,
  }));
  const countOptions: ChipOption[] = [1, 2, 3, 4, 5, 6, 7, 8].map((n) => {
    const reasons: string[] = [];
    if (n > 1 && s.par === "tp" && model.attentionHeads % n !== 0) reasons.push(`TP=${n} : ${model.attentionHeads} têtes non divisibles`);
    if (n > 1 && !gpu.interconnect.nvlink) reasons.push("pas de NVLink");
    if (n > gpu.maxGpusPerNode) reasons.push(`> ${gpu.maxGpusPerNode} GPU/nœud`);
    return { id: String(n), label: String(n), warn: reasons.length > 0, title: reasons.join(" · ") || undefined };
  });
  const parOptions: ChipOption[] = (["tp", "pp", "dp-attention"] as Parallelism[]).map((p) => ({
    id: p,
    label: PARALLELISM_LABELS[p],
    level: p === "dp-attention" && !isMoE(model) ? "unsupported" : undefined,
    title:
      p === "tp"
        ? "Chaque couche répartie sur tous les GPUs (all-reduce à chaque couche, NVLink recommandé)."
        : p === "pp"
          ? "Couches réparties en étages successifs : pas de duplication du KV, peu de communication."
          : isMoE(model)
            ? "Attention répliquée, requêtes réparties par GPU, experts en expert-parallel (SGLang / vLLM)."
            : "Réservé aux modèles MoE.",
  }));

  const limitationLine = (map: Record<string, Support>, formats: { id: string; label: string }[]) => {
    const un = formats.filter((f) => supportOf(map, f.id).level === "unsupported").map((f) => f.label);
    const lim = formats.filter((f) => supportOf(map, f.id).level === "limited").map((f) => f.label);
    if (!un.length && !lim.length) return <p className="hint ok">Tous les formats sont supportés nativement sur {gpu.name}.</p>;
    return (
      <p className="hint">
        Sur {gpu.name} : {un.length > 0 && <span className="lim-un">✕ {un.join(", ")} non supporté{un.length > 1 ? "s" : ""}</span>}
        {un.length > 0 && lim.length > 0 && " · "}
        {lim.length > 0 && <span className="lim-lim">⚠ {lim.join(", ")} limité{lim.length > 1 ? "s" : ""} (émulation ou débit réduit)</span>}
      </p>
    );
  };

  const maxCtx = model.maxContextExtended ?? model.maxContext;
  const replicated = est.kv.perRequestClusterBytes > est.kv.perRequestBytes * 1.001;
  const sessionsPerGpu = s.par === "dp-attention" ? est.sessions / s.gpuCount : est.sessions;
  const kvUsedPerGpu = sessionsPerGpu * est.kv.perRequestPerGpuBytes;
  const hasError = issues.some((i) => i.severity === "error");

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Estimateur KV cache</h1>
          <p className="subtitle">Mémoire des poids et du KV cache pour les LLM open-weight sur GPU NVIDIA</p>
        </div>
        <p className="config-path" title={configPath}>
          {config.models.length} modèles · {config.gpus.length} GPUs · <code>{configPath.split("/").slice(-2).join("/")}</code>
        </p>
      </header>

      <main className="layout">
        {/* ------------------------------------------------ Paramètres */}
        <section className="panel form" aria-label="Paramètres">
          <div className="field">
            <label className="field-label">Modèle</label>
            <ModelCombobox models={config.models} value={s.modelId} onChange={setModel} />
            <ModelCard model={model} />
          </div>

          <div className="field">
            <label className="field-label" htmlFor="ctx">
              Fenêtre de contexte <strong className="value">{contextLabel(ctx)}</strong>
              <span className="muted"> ({fmtInt(ctx)} tokens)</span>
            </label>
            <input
              id="ctx"
              type="range"
              min={0}
              max={CONTEXT_STEPS.length - 1}
              step={1}
              value={s.ctxIdx}
              onChange={(e) => update({ ctxIdx: Number(e.target.value) })}
              className={ctx > maxCtx ? "range-error" : ctx > model.maxContext ? "range-warn" : ""}
            />
            <div className="ticks">
              {CONTEXT_STEPS.map((t, i) => (
                <button
                  key={t}
                  type="button"
                  className={`tick${i === s.ctxIdx ? " current" : ""}${t > maxCtx ? " over" : t > model.maxContext ? " ext" : ""}`}
                  onClick={() => update({ ctxIdx: i })}
                  title={t > maxCtx ? "Au-delà du maximum du modèle" : t > model.maxContext ? "Nécessite une extension RoPE (YaRN)" : undefined}
                >
                  {contextLabel(t)}
                </button>
              ))}
            </div>
            <p className="hint">
              Natif : {contextLabel(model.maxContext)}
              {model.maxContextExtended && <> · étendu (YaRN) : {contextLabel(model.maxContextExtended)}</>}
              <span className="legend-ext"> ■ extension</span>
              <span className="legend-over"> ■ hors limite</span>
            </p>
          </div>

          <div className="field">
            <label className="field-label">Type de carte</label>
            <ChipGroup label="Type de carte" options={gpuOptions} value={s.gpuId} onChange={setGpu} className="gpu-grid" />
            {gpu.notes.length > 0 && (
              <ul className="gpu-notes">
                {gpu.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            )}
          </div>

          <div className="field">
            <label className="field-label">
              Nombre de cartes <strong className="value">{s.gpuCount}</strong>
              <span className="muted"> · {fmtBytes(gpu.memoryGiB * 1024 ** 3 * s.gpuCount)} au total</span>
            </label>
            <ChipGroup label="Nombre de cartes" options={countOptions} value={String(s.gpuCount)} onChange={(v) => update({ gpuCount: Number(v) })} className="count" />
            {s.gpuCount > 1 && (
              <>
                <label className="field-label sub">Parallélisme</label>
                <ChipGroup label="Parallélisme" options={parOptions} value={s.par} onChange={(v) => update({ par: v as Parallelism })} />
              </>
            )}
            <p className={`hint ${gpu.interconnect.nvlink ? "" : "lim-lim"}`}>Interconnexion : {gpu.interconnect.label}</p>
          </div>

          <div className="field">
            <label className="field-label">Format des poids</label>
            <ChipGroup label="Format des poids" options={weightOptions} value={s.wf} onChange={(v) => update({ wf: v })} />
            {limitationLine(gpu.weightFormats, config.weightFormats)}
            <p className="hint">★ format natif du modèle · « officiel » : checkpoint publié par l&apos;éditeur</p>
          </div>

          <div className="field">
            <label className="field-label">Format du KV cache</label>
            <ChipGroup label="Format du KV cache" options={kvOptions} value={s.kf} onChange={(v) => update({ kf: v })} />
            {limitationLine(gpu.kvCacheFormats, config.kvCacheFormats)}
          </div>

          <div className="field">
            <label className="field-label" htmlFor="reserve">
              Réserve mémoire par GPU <strong className="value">{s.reservePct} %</strong>
              <span className="muted"> ({fmtBytes(est.memory.reservePerGpuBytes)})</span>
            </label>
            <input id="reserve" type="range" min={0} max={30} step={1} value={s.reservePct} onChange={(e) => update({ reservePct: Number(e.target.value) })} />
            <p className="hint">
              Activations, graphes CUDA, contexte CUDA, fragmentation. Équivaut à vLLM <code>--gpu-memory-utilization {(1 - s.reservePct / 100).toFixed(2)}</code>.
            </p>
          </div>
        </section>

        {/* ------------------------------------------------ Résultats */}
        <section className="results" aria-label="Résultats">
          <div className="kpis">
            <Kpi title="Poids du modèle" value={fmtBytes(est.weights.totalBytes)} sub={s.gpuCount > 1 ? `${fmtBytes(est.weights.perGpuBytes)} par GPU` : `${wf.label} · ${fmtParams(model.paramsB)} paramètres`} tone={est.fitsWeights ? undefined : "bad"} />
            <Kpi
              title="KV par requête"
              value={fmtBytes(est.kv.perRequestBytes)}
              sub={replicated ? `${fmtBytes(est.kv.perRequestClusterBytes)} réels (réplication)` : `${contextLabel(ctx)} tokens · ${kf.label}`}
              tone={replicated ? "warn" : undefined}
            />
            <Kpi
              title="Modèle + 1 KV"
              value={fmtBytes(est.weights.totalBytes + est.kv.perRequestBytes)}
              sub={`${est.fitsOneRequest ? "✓ tient" : "✕ ne tient pas"} dans ${fmtBytes(est.memory.usablePerGpuBytes * s.gpuCount)} utilisables`}
              tone={est.fitsOneRequest ? "good" : "bad"}
            />
            <Kpi
              title="KV total disponible"
              value={fmtBytes(est.memory.freeForKvTotalBytes)}
              big={`${fmtInt(est.sessions)} session${est.sessions > 1 ? "s" : ""}`}
              sub={est.sessions > 0 ? `à ${contextLabel(ctx)} · ≈ ${fmtTokens(est.sessions * ctx)} tokens cumulés` : "aucune session complète"}
              tone={est.sessions > 0 ? "good" : "bad"}
            />
          </div>

          <MemoryBar
            gpuBytes={est.memory.gpuBytes}
            weights={est.weights.perGpuBytes}
            kvUsed={kvUsedPerGpu}
            free={est.memory.freeForKvPerGpuBytes - kvUsedPerGpu}
            reserve={est.memory.reservePerGpuBytes}
            sessionsPerGpu={sessionsPerGpu}
            gpuCount={s.gpuCount}
          />

          {(issues.length > 0 || notices.length > 0) && (
            <div className={`panel issues${hasError ? " has-error" : ""}`}>
              <h2>Limitations et avertissements</h2>
              <ul>
                {notices.map((n) => (
                  <li key={n} className="issue notice">
                    <span aria-hidden>🔁</span>
                    <div>
                      <strong>Ajusté automatiquement</strong>
                      <p>{n}</p>
                    </div>
                  </li>
                ))}
                {issues.map((i) => (
                  <li key={i.title} className={`issue ${i.severity}`}>
                    <span aria-hidden>{SEVERITY_ICON[i.severity]}</span>
                    <div>
                      <strong>{i.title}</strong>
                      {i.detail && <p>{i.detail}</p>}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="panel">
            <h2>Détail du KV par requête ({contextLabel(ctx)})</h2>
            <table className="table">
              <thead>
                <tr>
                  <th>Couches</th>
                  <th>Type</th>
                  <th>Détail</th>
                  <th className="num">Par requête</th>
                  {s.gpuCount > 1 && <th className="num">Par GPU</th>}
                </tr>
              </thead>
              <tbody>
                {est.kv.groups.map((g, i) => (
                  <tr key={i}>
                    <td>{g.count}</td>
                    <td>{g.label}</td>
                    <td className="muted">{g.detail}</td>
                    <td className="num">{fmtBytes(g.bytes)}</td>
                    {s.gpuCount > 1 && (
                      <td className="num">
                        {fmtBytes(g.perGpuBytes)}
                        {g.replication > 1 && <span className="tag tag-warn">×{g.replication} répliqué</span>}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr>
                  <td colSpan={2}>Total</td>
                  <td />
                  <td className="num">{fmtBytes(est.kv.perRequestBytes)}</td>
                  {s.gpuCount > 1 && <td className="num">{fmtBytes(est.kv.perRequestPerGpuBytes)}</td>}
                </tr>
              </tfoot>
            </table>
            {est.kv.perTokenBytes > 0 && (
              <p className="hint">
                {fmtInt(est.kv.perTokenBytes / 1024)} Kio par token sur les couches à contexte complet
                {est.kv.constantBytes > 0 && <> · {fmtBytes(est.kv.constantBytes)} d&apos;état constant</>}.
              </p>
            )}
          </div>

          <div className="panel">
            <h2>Sessions selon le contexte</h2>
            <SessionsChart rows={byContext} current={ctx} maxNative={model.maxContext} maxExt={maxCtx} onPick={(i) => update({ ctxIdx: i })} />
          </div>

          <details className="panel assumptions">
            <summary>Hypothèses de calcul</summary>
            <ul>
              <li>Poids = paramètres × bits du format. Les embeddings, lm_head et l&apos;encodeur vision restent en 16 bits si on quantifie.</li>
              <li>KV GQA = couches × têtes KV × (dim K + dim V) × octets × tokens. Les couches à fenêtre glissante sont plafonnées à la fenêtre.</li>
              <li>MLA = couches × (rang latent + dim RoPE) × octets × tokens, sans facteur 2 (K et V partagent le latent).</li>
              <li>Couches linéaires (Mamba, DeltaNet, KDA) : état fixe par requête, indépendant du contexte.</li>
              <li>TP : les têtes KV sont réparties entre GPUs et répliquées si TP &gt; têtes KV. Le latent MLA est copié sur chaque GPU.</li>
              <li>PP : couches réparties de façon homogène, sans duplication. DP-attention : chaque requête vit sur un seul GPU, attention répliquée.</li>
              <li>Sessions = ⌊mémoire libre par GPU ÷ KV par requête et par GPU⌋, toutes au contexte maximal (cas le plus défavorable). Le prefix caching et les blocs partiels sont ignorés.</li>
              <li>1 Gio = 2³⁰ octets. La mémoire GPU est celle vue par le driver.</li>
            </ul>
          </details>
        </section>
      </main>
    </div>
  );
}

function ModelCard({ model: m }: { model: ModelSpec }) {
  const layers = m.layers.reduce((a, g) => a + g.count, 0);
  return (
    <div className="model-card">
      <div className="badges">
        <span className="tag">{archLabel(m)}</span>
        {m.tags.filter((t) => t !== "swa" && t !== "hybrid").map((t) => (
          <span key={t} className="tag">
            {TAG_LABELS[t] ?? t}
          </span>
        ))}
        {m.confidence !== "high" && (
          <span className={`tag ${m.confidence === "low" ? "tag-warn" : ""}`} title="Paramètres d'architecture à vérifier dans config.json">
            fiabilité {m.confidence === "low" ? "faible" : "moyenne"}
          </span>
        )}
      </div>
      <dl>
        <div>
          <dt>Paramètres</dt>
          <dd>
            {fmtParams(m.paramsB)}
            {m.activeParamsB && <span className="muted"> ({fmtParams(m.activeParamsB)} actifs)</span>}
          </dd>
        </div>
        <div>
          <dt>Couches à cache</dt>
          <dd>{layers}</dd>
        </div>
        <div>
          <dt>Format natif</dt>
          <dd>{m.nativeWeightFormat.toUpperCase()}</dd>
        </div>
        <div>
          <dt>Officiels</dt>
          <dd>{m.officialWeightFormats.map((f) => f.toUpperCase()).join(", ") || "—"}</dd>
        </div>
      </dl>
      {m.notes.length > 0 && (
        <ul className="model-notes">
          {m.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
      {m.source && (
        <a className="source" href={`https://huggingface.co/${m.source}`} target="_blank" rel="noreferrer">
          {m.source} ↗
        </a>
      )}
    </div>
  );
}

function Kpi({ title, value, sub, big, tone }: { title: string; value: string; sub?: string; big?: string; tone?: "good" | "bad" | "warn" }) {
  return (
    <div className={`kpi${tone ? ` ${tone}` : ""}`}>
      <div className="kpi-title">{title}</div>
      <div className="kpi-value">{value}</div>
      {big && <div className="kpi-big">{big}</div>}
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

function MemoryBar(p: {
  gpuBytes: number;
  weights: number;
  kvUsed: number;
  free: number;
  reserve: number;
  sessionsPerGpu: number;
  gpuCount: number;
}) {
  const overflow = Math.max(0, p.weights - (p.gpuBytes - p.reserve));
  const scale = p.gpuBytes + overflow;
  const pct = (b: number) => `${(Math.max(0, b) / scale) * 100}%`;
  const segs = [
    { key: "weights", label: "Poids", bytes: Math.min(p.weights, p.gpuBytes - p.reserve) },
    { key: "overflow", label: "Dépassement", bytes: overflow },
    { key: "kv", label: `KV (${fmtInt(p.sessionsPerGpu)} session${p.sessionsPerGpu > 1 ? "s" : ""})`, bytes: p.kvUsed },
    { key: "free", label: "Libre (< 1 session)", bytes: p.free },
    { key: "reserve", label: "Réserve", bytes: p.reserve },
  ].filter((s) => s.bytes > 0);
  return (
    <div className="panel">
      <h2>
        Mémoire d&apos;un GPU <span className="muted">({fmtBytes(p.gpuBytes)}{p.gpuCount > 1 ? ` × ${p.gpuCount}` : ""})</span>
      </h2>
      <div className="membar" role="img" aria-label="Répartition de la mémoire d'un GPU">
        {segs.map((s) => (
          <div key={s.key} className={`seg seg-${s.key}`} style={{ width: pct(s.bytes) }} title={`${s.label} : ${fmtBytes(s.bytes)}`} />
        ))}
      </div>
      <div className="legend">
        {segs.map((s) => (
          <span key={s.key}>
            <i className={`dot seg-${s.key}`} /> {s.label} <strong>{fmtBytes(s.bytes)}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

function SessionsChart({
  rows,
  current,
  maxNative,
  maxExt,
  onPick,
}: {
  rows: { t: number; e: ReturnType<typeof estimate> }[];
  current: number;
  maxNative: number;
  maxExt: number;
  onPick: (i: number) => void;
}) {
  const max = Math.max(1, ...rows.map((r) => r.e.sessions));
  // Échelle logarithmique : les sessions varient d'un facteur ~100 entre 8k et 1M.
  const w = (n: number) => (n <= 0 ? 0 : (Math.log10(n + 1) / Math.log10(max + 1)) * 100);
  return (
    <div className="sessions">
      {rows.map((r, i) => (
        <button
          key={r.t}
          type="button"
          className={`srow${r.t === current ? " current" : ""}${r.t > maxExt ? " over" : r.t > maxNative ? " ext" : ""}`}
          onClick={() => onPick(i)}
        >
          <span className="slabel">{contextLabel(r.t)}</span>
          <span className="sbar">
            <span style={{ width: `${w(r.e.sessions)}%` }} />
          </span>
          <span className="sval">{fmtInt(r.e.sessions)}</span>
          <span className="skv muted">{fmtBytes(r.e.kv.perRequestBytes)}/req.</span>
        </button>
      ))}
      <p className="hint">Échelle logarithmique. Cliquez une ligne pour choisir ce contexte.</p>
    </div>
  );
}
