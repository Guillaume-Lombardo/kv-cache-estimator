"use client";

import { useEffect, useMemo, useState } from "react";
import type { AppConfig, EngineSpec, GpuSpec, ModelSpec, Support } from "@/lib/config-schema";
import {
  collectIssues,
  CONTEXT_STEPS,
  contextLabel,
  estimate,
  GIB,
  isMoE,
  isValidTp,
  kvSupport,
  supportLevelLabel,
  weightSupport,
  type ParallelMode,
} from "@/lib/calc";
import { fmtBytes, fmtDuration, fmtInt, fmtParams, fmtRate, fmtTokens } from "@/lib/format";
import { estimatePerf } from "@/lib/perf";
import { defaultTp, divisors, gpuCountOptions, maxTp, pickKvFormat, pickWeightFormat } from "@/lib/planning";
import { launchCommand, resolveScenario, scenarioFromQuery, scenarioToQuery, type QueryParams, type Scenario } from "@/lib/scenario";
import { ChipGroup, type ChipOption } from "./ChipGroup";
import { ComparePanel } from "./ComparePanel";
import { ImportPanel } from "./ImportPanel";
import { ModelCombobox } from "./ModelCombobox";
import { IssuesPanel, Kpi, KvBreakdown, MemoryBar, ModelCard, SessionsChart } from "./panels";
import { PerfPanel } from "./PerfPanel";
import { SizingPanel, type SizingChoice } from "./SizingPanel";

const STORAGE = { pinned: "kv-estimator:pinned", imported: "kv-estimator:imported", tab: "kv-estimator:tab" };
const TABS = [
  { id: "detail", label: "Détail" },
  { id: "perf", label: "Débit" },
  { id: "sizing", label: "Dimensionnement" },
  { id: "compare", label: "Comparaison" },
  { id: "import", label: "Importer un modèle" },
] as const;
type TabId = (typeof TABS)[number]["id"];

function readStorage<T>(key: string, fallback: T): T {
  try {
    const v = window.localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
}
function writeStorage(key: string, value: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // stockage indisponible (navigation privée…) : sans conséquence
  }
}

const usable = (s: Support) => s.level !== "unsupported";

export function Estimator({ config, configPath, initial }: { config: AppConfig; configPath: string; initial: QueryParams }) {
  const [imported, setImported] = useState<ModelSpec[]>([]);
  const models = useMemo(() => [...config.models, ...imported], [config.models, imported]);
  const [s, setS] = useState<Scenario>(() => scenarioFromQuery(config, config.models, initial));
  const [notices, setNotices] = useState<string[]>([]);
  const [pinned, setPinned] = useState<Scenario[]>([]);
  const [tab, setTab] = useState<TabId>("detail");
  const [linkCopied, setLinkCopied] = useState(false);

  // --- Données propres au navigateur (après hydratation)
  useEffect(() => {
    const imp = readStorage<ModelSpec[]>(STORAGE.imported, []);
    setImported(imp);
    setPinned(readStorage<Scenario[]>(STORAGE.pinned, []));
    const t = readStorage<TabId>(STORAGE.tab, "detail");
    if (TABS.some((x) => x.id === t)) setTab(t);
    // Lien vers un modèle importé dans ce navigateur.
    if (initial.model && imp.some((m) => m.id === initial.model)) setS(scenarioFromQuery(config, [...config.models, ...imp], initial));
  }, []);
  useEffect(() => {
    window.history.replaceState(null, "", `?${scenarioToQuery(s)}`);
    setLinkCopied(false);
  }, [s]);

  const model = models.find((m) => m.id === s.modelId) ?? models[0];
  const gpu = config.gpus.find((g) => g.id === s.gpuId)!;
  const engine = config.engines.find((e) => e.id === s.engineId)!;
  const wf = config.weightFormats.find((f) => f.id === s.wf)!;
  const kf = config.kvCacheFormats.find((f) => f.id === s.kf)!;
  const ctx = s.ctx;

  // --- Changements avec ajustement automatique
  const dpAllowed = (m: ModelSpec, e: EngineSpec) => isMoE(m) && e.features.dpAttention;

  /** Formats après changement de GPU ou de moteur : garde le choix courant si possible, revient au natif si le choix était un repli. */
  const adjustFormats = (m: ModelSpec, g: GpuSpec, e: EngineSpec, msgs: string[]) => {
    let nextWf = pickWeightFormat(config, m, g, e, s.wf);
    const native = config.weightFormats.find((f) => f.id === m.nativeWeightFormat);
    if (nextWf === s.wf && native && wf.bitsPerParam > native.bitsPerParam && usable(weightSupport(g, e, native.id)) && s.wf !== native.id) {
      nextWf = native.id;
      msgs.push(`Format natif ${native.label} disponible → ${native.label}.`);
    } else if (nextWf !== s.wf) msgs.push(`Poids ${wf.label} non supportés (${g.name}, ${e.name}) → ${nextWf.toUpperCase()}.`);
    let nextKf = pickKvFormat(config, g, e, s.kf);
    // FP16 imposé par l'absence de BF16 (Volta) : retour en BF16 dès que possible.
    if (nextKf === "fp16" && !usable(kvSupport(gpu, engine, "bf16")) && kvSupport(g, e, "bf16").level === "native") nextKf = "bf16";
    if (nextKf !== s.kf) msgs.push(usable(kvSupport(g, e, s.kf)) ? "KV BF16 disponible → BF16." : `KV ${kf.label} non supporté → ${nextKf.toUpperCase()}.`);
    return { wf: nextWf, kf: nextKf };
  };

  const setModel = (id: string) => {
    const m = models.find((x) => x.id === id)!;
    const msgs: string[] = [];
    const nextWf = pickWeightFormat(config, m, gpu, engine);
    if (nextWf !== m.nativeWeightFormat) msgs.push(`Format natif ${m.nativeWeightFormat.toUpperCase()} indisponible → ${nextWf.toUpperCase()}.`);
    let mode = s.mode;
    if (mode === "dp-ep" && !dpAllowed(m, engine)) {
      mode = "tp-pp";
      msgs.push("DP-attention réservé aux MoE → TP/PP.");
    }
    setNotices(msgs);
    setS({ ...s, modelId: id, wf: nextWf, mode, tp: defaultTp(m, gpu, s.gpuCount) });
  };
  const setGpu = (id: string) => {
    const g = config.gpus.find((x) => x.id === id)!;
    const msgs: string[] = [];
    const counts = gpuCountOptions(config, g);
    const n = counts.includes(s.gpuCount) ? s.gpuCount : 8;
    if (n !== s.gpuCount) msgs.push(`${s.gpuCount} GPUs non proposé pour ${g.name} → ${n}.`);
    const f = adjustFormats(model, g, engine, msgs);
    setNotices(msgs);
    setS({ ...s, gpuId: id, gpuCount: n, tp: defaultTp(model, g, n), ...f });
  };
  const setEngine = (id: string) => {
    const e = config.engines.find((x) => x.id === id)!;
    const msgs: string[] = [];
    const f = adjustFormats(model, gpu, e, msgs);
    let mode = s.mode;
    if (mode === "dp-ep" && !dpAllowed(model, e)) {
      mode = "tp-pp";
      msgs.push(`DP-attention indisponible dans ${e.name} → TP/PP.`);
    }
    setNotices(msgs);
    setS({ ...s, engineId: id, mode, ...f });
  };
  const update = (patch: Partial<Scenario>) => {
    setNotices([]);
    setS({ ...s, ...patch });
  };
  const loadScenario = (next: Scenario) => {
    const fixed = scenarioFromQuery(config, models, Object.fromEntries(new URLSearchParams(scenarioToQuery(next))));
    setNotices([]);
    setS(fixed);
  };
  const applySizing = (c: SizingChoice) => {
    setNotices([`Configuration appliquée depuis le dimensionnement.`]);
    setS({ ...s, ...c });
  };

  const pin = () => {
    const next = [...pinned, s].slice(-6);
    setPinned(next);
    writeStorage(STORAGE.pinned, next);
    setTab("compare");
    writeStorage(STORAGE.tab, "compare");
  };
  const unpin = (i: number) => {
    const next = pinned.filter((_, k) => k !== i);
    setPinned(next);
    writeStorage(STORAGE.pinned, next);
  };
  const addImported = (m: ModelSpec) => {
    const next = [...imported.filter((x) => x.id !== m.id), { ...m, family: m.family || "Importé" }];
    setImported(next);
    writeStorage(STORAGE.imported, next);
    const nextWf = pickWeightFormat(config, m, gpu, engine);
    setNotices([`${m.name} ajouté à la liste des modèles.`]);
    setS({ ...s, modelId: m.id, wf: nextWf, mode: s.mode === "dp-ep" && !dpAllowed(m, engine) ? "tp-pp" : s.mode, tp: defaultTp(m, gpu, s.gpuCount) });
  };
  const clearImported = () => {
    setImported([]);
    writeStorage(STORAGE.imported, []);
    if (imported.some((m) => m.id === s.modelId)) setModel(config.models[0].id);
  };
  const selectTab = (t: TabId) => {
    setTab(t);
    writeStorage(STORAGE.tab, t);
  };

  // --- Calculs
  const input = useMemo(() => resolveScenario(config, models, { ...s, modelId: model.id })!, [config, models, s, model.id]);
  const est = useMemo(() => estimate(input), [input]);
  const issues = useMemo(() => collectIssues(input, est, config.weightFormats), [input, est, config.weightFormats]);
  const perf = useMemo(() => estimatePerf(input, est, config.performance), [input, est, config.performance]);
  const byContext = useMemo(() => CONTEXT_STEPS.map((t) => ({ t, e: estimate({ ...input, contextTokens: t }) })), [input]);
  const command = launchCommand(input, est.tp, est.pp);

  // --- Options
  const weightOptions: ChipOption[] = config.weightFormats.map((f) => {
    const sup = weightSupport(gpu, engine, f.id);
    const official = model.officialWeightFormats.includes(f.id);
    const bits = f.id === model.nativeWeightFormat && model.nativeBitsPerParam ? model.nativeBitsPerParam : f.bitsPerParam;
    return {
      id: f.id,
      label: f.label,
      sub: `${bits} bits${official ? " · officiel" : ""}`,
      level: sup.level,
      star: f.id === model.nativeWeightFormat,
      warn: model.weightFormatNotes[f.id] !== undefined,
      title: [f.description, sup.level !== "native" ? supportLevelLabel(sup.level) : null, sup.note, model.weightFormatNotes[f.id]?.note].filter(Boolean).join("\n"),
    };
  });
  const kvOptions: ChipOption[] = config.kvCacheFormats.map((f) => {
    const sup = kvSupport(gpu, engine, f.id);
    return {
      id: f.id,
      label: f.label,
      sub: `${f.bytesPerElement * 8} bits`,
      level: sup.level,
      warn: model.kvFormatNotes[f.id] !== undefined,
      title: [f.description, sup.level !== "native" ? supportLevelLabel(sup.level) : null, sup.note].filter(Boolean).join("\n"),
    };
  });
  const gpuOptions: ChipOption[] = config.gpus.map((g) => ({
    id: g.id,
    label: g.name,
    sub: (
      <>
        {g.memoryLabel.split(" (")[0]}
        <br />
        <span className={g.interconnect.nvlink ? "" : "no-link"}>{g.interconnect.nvlink ? (g.nvlinkDomain > 8 ? "NVL72" : "NVLink") : "sans NVLink GPU↔GPU"}</span>
      </>
    ),
    title: `${g.architecture} (sm ${g.computeCapability}) · ${g.memoryLabel} · ${g.bandwidthTBs} To/s · ${g.interconnect.label}`,
  }));
  const countOptions: ChipOption[] = gpuCountOptions(config, gpu).map((n) => {
    const reasons: string[] = [];
    if (n > 1 && !gpu.interconnect.nvlink) reasons.push("pas de NVLink");
    if (n > gpu.maxGpusPerNode) reasons.push(`${Math.ceil(n / gpu.maxGpusPerNode)} nœuds`);
    const sameDomain = gpu.interconnect.nvlink && n <= gpu.nvlinkDomain;
    return {
      id: String(n),
      label: String(n),
      warn: n > 1 && (!gpu.interconnect.nvlink || (n > gpu.maxGpusPerNode && !sameDomain)),
      title: [reasons.join(" · "), sameDomain && n > gpu.maxGpusPerNode ? "même domaine NVLink" : ""].filter(Boolean).join(" — ") || undefined,
    };
  });
  const modeOptions: ChipOption[] = [
    { id: "tp-pp", label: "TP / PP", title: "Tensor parallel dans un étage, pipeline parallel entre étages." },
    {
      id: "dp-ep",
      label: "DP-attention + EP",
      level: dpAllowed(model, engine) ? undefined : "unsupported",
      title: !isMoE(model) ? "Réservé aux modèles MoE." : !engine.features.dpAttention ? `Indisponible dans ${engine.name}.` : "Attention répliquée, requêtes réparties par GPU, experts en expert-parallel.",
    },
  ];
  const tpOptions: ChipOption[] = divisors(s.gpuCount).map((d) => {
    const valid = isValidTp(model, d);
    const crosses = d > maxTp(gpu);
    return {
      id: String(d),
      label: `TP ${d}`,
      sub: `PP ${s.gpuCount / d}`,
      level: valid ? undefined : "unsupported",
      warn: valid && (crosses || (d > 1 && !gpu.interconnect.nvlink)),
      title: !valid
        ? `${model.attentionHeads} têtes d'attention / têtes KV non divisibles par ${d}`
        : crosses
          ? `TP au-delà du domaine ${gpu.interconnect.nvlink ? "NVLink" : "du serveur"}`
          : d > 1 && !gpu.interconnect.nvlink
            ? "TP via PCIe : lent"
            : undefined,
    };
  });
  const engineOptions: ChipOption[] = config.engines.map((e) => ({
    id: e.id,
    label: e.name,
    sub: e.confidence === "high" ? undefined : "profil à compléter",
    warn: e.confidence !== "high",
    title: [e.description, ...e.notes].filter(Boolean).join("\n"),
  }));

  const limitationLine = (supports: { label: string; s: Support }[]) => {
    const un = supports.filter((x) => x.s.level === "unsupported").map((x) => x.label);
    const lim = supports.filter((x) => x.s.level === "limited").map((x) => x.label);
    if (!un.length && !lim.length) return <p className="hint ok">Tous les formats sont supportés nativement ({gpu.name}, {engine.name}).</p>;
    return (
      <p className="hint">
        {gpu.name} · {engine.name} : {un.length > 0 && <span className="lim-un">✕ {un.join(", ")} non supporté{un.length > 1 ? "s" : ""}</span>}
        {un.length > 0 && lim.length > 0 && " · "}
        {lim.length > 0 && <span className="lim-lim">⚠ {lim.join(", ")} limité{lim.length > 1 ? "s" : ""}</span>}
        <span className="muted"> (survol pour le détail)</span>
      </p>
    );
  };

  const maxCtx = model.maxContextExtended ?? model.maxContext;
  const replicated = est.kv.perRequestClusterBytes > est.kv.perRequestBytes * 1.001;
  const memGiB = s.memGiB ?? gpu.memoryGiB;

  const exportJson = () => {
    const data = {
      generatedAt: new Date().toISOString(),
      url: window.location.href,
      scenario: s,
      model: { id: model.id, name: model.name, source: model.source, revision: model.revision },
      gpu: { id: gpu.id, name: gpu.name, memoryGiB: memGiB },
      engine: engine.name,
      parallelism: { mode: s.mode, tp: est.tp, pp: est.pp, gpuCount: s.gpuCount },
      results: {
        weightsGiB: est.weights.totalBytes / GIB,
        weightsPerGpuGiB: est.weights.perGpuBytes / GIB,
        kvPerRequestGiB: est.kv.perRequestBytes / GIB,
        kvPerRequestClusterGiB: est.kv.perRequestClusterBytes / GIB,
        modelPlusOneKvGiB: est.modelPlusOneKvBytes / GIB,
        freeKvTotalGiB: est.memory.freeForKvTotalBytes / GIB,
        sessions: est.sessions,
        offloadParkedSessions: est.offload.parkedSessions,
        decodeTokPerSecSingle: perf.single.tokPerSecPerSession,
        decodeTokPerSecFullLoad: perf.full?.tokPerSecTotal ?? null,
        prefillSeconds: perf.prefillSeconds,
      },
      issues,
      command,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `kv-${model.id}-${s.gpuCount}x${gpu.id}-${contextLabel(ctx)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setLinkCopied(true);
    } catch {
      setLinkCopied(false);
    }
  };

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Estimateur KV cache</h1>
          <p className="subtitle">Mémoire des poids et du KV cache, sessions et débit pour les LLM open-weight sur GPU NVIDIA</p>
        </div>
        <p className="config-path" title={configPath}>
          {config.models.length} modèles{imported.length ? ` + ${imported.length} importé${imported.length > 1 ? "s" : ""}` : ""} · {config.gpus.length} GPUs ·{" "}
          {config.engines.length} moteurs · <code>{configPath.split("/").slice(-2).join("/")}</code>
        </p>
      </header>

      <main className="layout">
        {/* ------------------------------------------------ Paramètres */}
        <section className="panel form" aria-label="Paramètres">
          <div className="field">
            <label className="field-label">Modèle</label>
            <ModelCombobox models={models} value={model.id} onChange={setModel} />
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
              value={CONTEXT_STEPS.indexOf(ctx)}
              onChange={(e) => update({ ctx: CONTEXT_STEPS[Number(e.target.value)] })}
              className={ctx > maxCtx ? "range-error" : ctx > model.maxContext ? "range-warn" : ""}
            />
            <div className="ticks">
              {CONTEXT_STEPS.map((t) => (
                <button
                  key={t}
                  type="button"
                  className={`tick${t === ctx ? " current" : ""}${t > maxCtx ? " over" : t > model.maxContext ? " ext" : ""}`}
                  onClick={() => update({ ctx: t })}
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
              <span className="muted"> · {fmtBytes(memGiB * GIB * s.gpuCount)} au total</span>
            </label>
            <ChipGroup
              label="Nombre de cartes"
              options={countOptions}
              value={String(s.gpuCount)}
              onChange={(v) => update({ gpuCount: Number(v), tp: defaultTp(model, gpu, Number(v)) })}
              className="count"
            />
            {s.gpuCount > 1 && (
              <>
                <label className="field-label sub">Parallélisme</label>
                <ChipGroup label="Parallélisme" options={modeOptions} value={s.mode} onChange={(v) => update({ mode: v as ParallelMode })} />
                {s.mode === "tp-pp" && (
                  <ChipGroup label="Degré de TP" options={tpOptions} value={String(s.tp)} onChange={(v) => update({ tp: Number(v) })} className="count" />
                )}
              </>
            )}
            <p className={`hint ${gpu.interconnect.nvlink ? "" : "lim-lim"}`}>
              Interconnexion : {gpu.interconnect.label}
              {gpu.nvlinkDomain > 8 && ` · ${gpu.maxGpusPerNode} GPUs par tray`}
            </p>
          </div>

          <div className="field">
            <label className="field-label">Moteur d&apos;inférence</label>
            <ChipGroup label="Moteur" options={engineOptions} value={s.engineId} onChange={setEngine} />
            {engine.description && <p className="hint">{engine.description}</p>}
          </div>

          <div className="field">
            <label className="field-label">Format des poids</label>
            <ChipGroup label="Format des poids" options={weightOptions} value={s.wf} onChange={(v) => update({ wf: v })} />
            {limitationLine(config.weightFormats.map((f) => ({ label: f.label, s: weightSupport(gpu, engine, f.id) })))}
            <p className="hint">★ format natif du modèle · « officiel » : checkpoint publié par l&apos;éditeur</p>
          </div>

          <div className="field">
            <label className="field-label">Format du KV cache</label>
            <ChipGroup label="Format du KV cache" options={kvOptions} value={s.kf} onChange={(v) => update({ kf: v })} />
            {limitationLine(config.kvCacheFormats.map((f) => ({ label: f.label, s: kvSupport(gpu, engine, f.id) })))}
          </div>

          <div className="field">
            <label className="field-label" htmlFor="reserve">
              Réserve mémoire par GPU <strong className="value">{s.reservePct} %</strong>
              <span className="muted"> ({fmtBytes(est.memory.reservePerGpuBytes)})</span>
            </label>
            <input id="reserve" type="range" min={0} max={30} step={1} value={s.reservePct} onChange={(e) => update({ reservePct: Number(e.target.value) })} />
            <p className="hint">
              Activations, graphes CUDA, contexte CUDA, fragmentation.
              {engine.flags.memoryUtilization && (
                <>
                  {" "}
                  Équivaut à <code>{engine.flags.memoryUtilization} {(1 - s.reservePct / 100).toFixed(2)}</code>.
                </>
              )}
            </p>
          </div>

          <details className="field advanced" open={s.memGiB !== undefined || s.offloadGiB > 0}>
            <summary className="field-label">Options avancées</summary>
            <label className="inline-form">
              <span>Mémoire visible par GPU</span>
              <input
                type="number"
                step="0.1"
                min={1}
                value={s.memGiB ?? ""}
                placeholder={String(gpu.memoryGiB)}
                onChange={(e) => update({ memGiB: e.target.value ? Number(e.target.value) : undefined })}
              />
              <span className="muted">Gio (nvidia-smi MiB ÷ 1024)</span>
            </label>
            <label className="inline-form">
              <span>Offload KV vers la RAM CPU</span>
              <input type="number" min={0} step={16} value={s.offloadGiB || ""} placeholder="0" onChange={(e) => update({ offloadGiB: Math.max(0, Number(e.target.value) || 0) })} />
              <span className="muted">Gio par GPU</span>
            </label>
            {gpu.hostMemoryGiBPerGpu > 0 && s.offloadGiB !== gpu.hostMemoryGiBPerGpu && (
              <button type="button" className="link" onClick={() => update({ offloadGiB: gpu.hostMemoryGiBPerGpu })}>
                Utiliser la mémoire Grace ({fmtInt(gpu.hostMemoryGiBPerGpu)} Gio par GPU)
              </button>
            )}
            {!engine.features.kvOffload && <p className="hint lim-lim">Offload non disponible dans {engine.name}.</p>}
            <p className="hint">
              Lien CPU↔GPU : {gpu.hostLinkLabel} ({gpu.hostLinkGBs} Go/s). {engine.flags.kvOffload && <>Option : {engine.flags.kvOffload}.</>}
            </p>
          </details>
        </section>

        {/* ------------------------------------------------ Résultats */}
        <section className="results" aria-label="Résultats">
          <div className="toolbar">
            <button type="button" className="btn" onClick={pin} disabled={pinned.length >= 6} title="Ajouter à la comparaison">
              📌 Épingler
            </button>
            <button type="button" className="btn" onClick={copyLink}>
              {linkCopied ? "✓ Lien copié" : "🔗 Copier le lien"}
            </button>
            <button type="button" className="btn" onClick={exportJson}>
              ⬇ Export JSON
            </button>
          </div>

          <div className="kpis">
            <Kpi
              title="Poids du modèle"
              value={fmtBytes(est.weights.totalBytes)}
              sub={s.gpuCount > 1 ? `${fmtBytes(est.weights.perGpuBytes)} par GPU` : `${wf.label} · ${fmtParams(model.paramsB)} paramètres`}
              tone={est.fitsWeights ? undefined : "bad"}
            />
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
              sub={
                est.sessions > 0
                  ? `à ${contextLabel(ctx)} · ≈ ${fmtTokens(est.sessions * ctx)} tokens cumulés${est.offload.parkedSessions ? ` · +${fmtInt(est.offload.parkedSessions)} en RAM CPU` : ""}`
                  : "aucune session complète"
              }
              tone={est.sessions > 0 ? "good" : "bad"}
            />
          </div>

          <button type="button" className="perf-strip" onClick={() => selectTab("perf")} title="Voir le détail du débit">
            <span>
              ⚡ Débit estimé : <strong>{fmtRate(perf.single.tokPerSecPerSession)}</strong> pour 1 session
            </span>
            {perf.full && (
              <span>
                · <strong>{fmtRate(perf.full.tokPerSecTotal)}</strong> agrégés à pleine charge
              </span>
            )}
            <span>
              · prefill {contextLabel(ctx)} ≈ <strong>{fmtDuration(perf.prefillSeconds)}</strong>
            </span>
          </button>

          <MemoryBar est={est} gpuCount={s.gpuCount} />

          <IssuesPanel issues={issues} notices={notices} />

          <div className="panel">
            <div className="tabs" role="tablist">
              {TABS.map((t) => (
                <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} className={`tab${tab === t.id ? " active" : ""}`} onClick={() => selectTab(t.id)}>
                  {t.label}
                  {t.id === "compare" && pinned.length > 0 && <span className="badge">{pinned.length}</span>}
                </button>
              ))}
            </div>
            <div className="tab-body">
              {tab === "detail" && (
                <>
                  <KvBreakdown est={est} ctx={ctx} gpuCount={s.gpuCount} />
                  <SessionsChart rows={byContext} current={ctx} maxNative={model.maxContext} maxExt={maxCtx} onPick={(t) => update({ ctx: t })} />
                  {command && (
                    <>
                      <h3>Commande indicative</h3>
                      <pre className="yaml">{command}</pre>
                    </>
                  )}
                  {engine.notes.length > 0 && (
                    <ul className="gpu-notes">
                      {engine.notes.map((n) => (
                        <li key={n}>{n}</li>
                      ))}
                    </ul>
                  )}
                </>
              )}
              {tab === "perf" && <PerfPanel input={input} est={est} perf={config.performance} />}
              {tab === "sizing" && <SizingPanel config={config} input={input} onApply={applySizing} />}
              {tab === "compare" && (
                <ComparePanel config={config} models={models} current={s} pinned={pinned} onPin={pin} onRemove={unpin} onLoad={loadScenario} />
              )}
              {tab === "import" && (
                <ImportPanel existingIds={new Set(models.map((m) => m.id))} imported={imported} onAdd={addImported} onClear={clearImported} />
              )}
            </div>
          </div>

          <details className="panel assumptions">
            <summary>Hypothèses de calcul</summary>
            <ul>
              <li>Poids = paramètres × bits effectifs du format (échelles incluses). Embeddings, lm_head et encodeur vision restent en 16 bits si on quantifie.</li>
              <li>KV GQA = couches × têtes KV × (dim K + dim V) × octets × tokens. Fenêtres glissantes plafonnées à la fenêtre si le moteur le gère.</li>
              <li>MLA = couches × (rang latent + dim RoPE) × octets × tokens (+ indexeur DSA en FP8), sans facteur 2.</li>
              <li>Couches linéaires (Mamba, DeltaNet, KDA) : état SSM + convolution fixes par requête, en FP32 si le modèle l&apos;impose.</li>
              <li>TP : têtes KV réparties, répliquées si TP &gt; têtes KV ; latent MLA copié sur chaque rang. PP : couches réparties sans duplication.</li>
              <li>DP-attention + EP : attention répliquée sur chaque GPU, experts répartis, chaque requête sur un seul GPU.</li>
              <li>
                Sessions = ⌊mémoire libre par GPU ÷ KV par requête et par GPU⌋, toutes au contexte maximal (pire cas). Prefix caching et blocs partiels ignorés.
              </li>
              <li>Offload CPU : le KV déchargé ne sert qu&apos;aux sessions en attente ou préemptées, il n&apos;augmente pas la concurrence de décodage.</li>
              <li>Débit : modèle au premier ordre (bande passante, MFU, latences), à calibrer avec des mesures.</li>
              <li>1 Gio = 2³⁰ octets. Mémoire GPU vue par le driver, modifiable dans les options avancées.</li>
            </ul>
          </details>
        </section>
      </main>
    </div>
  );
}
