"use client";

import { useEffect, useMemo, useState } from "react";
import type { ConfigSource, EngineSpec, GpuSpec, ModelSource, ModelSpec, Support } from "@/lib/config-schema";
import {
  collectIssues,
  CONTEXT_STEPS,
  contextLabel,
  estimate,
  GIB,
  isMoE,
  isValidTp,
  kvSupport,
  runtimeDefaults,
  supportLevelLabel,
  weightSupport,
  type ParallelMode,
} from "@/lib/calc";
import { localize, type Lang } from "@/lib/i18n";
import { estimatePerf } from "@/lib/perf";
import { defaultTp, divisors, gpuCountOptions, maxTp, pickKvFormat, pickWeightFormat } from "@/lib/planning";
import { buildReport, UNMODELLED } from "@/lib/report";
import { launchCommand, resolveScenario, scenarioFromQuery, scenarioToQuery, type QueryParams, type Scenario } from "@/lib/scenario";
import { ChipGroup, type ChipOption } from "./ChipGroup";
import { ComparePanel } from "./ComparePanel";
import { ContextInput } from "./ContextInput";
import { GpuPicker } from "./GpuPicker";
import { ImportPanel } from "./ImportPanel";
import { IMPORTED_FAMILY, ModelCombobox } from "./ModelCombobox";
import { IssuesPanel, Kpi, KvBreakdown, MemoryBar, ModelCard, SessionsChart } from "./panels";
import { PerfPanel } from "./PerfPanel";
import { PrefsToggles, usePrefs } from "./Prefs";
import { SizingPanel, type SizingChoice } from "./SizingPanel";
import { WorkloadPanel } from "./WorkloadPanel";

const STORAGE = { pinned: "kv-estimator:pinned", imported: "kv-estimator:imported", tab: "kv-estimator:tab" };
const TABS = [
  { id: "detail", fr: "Détail", en: "Detail" },
  { id: "perf", fr: "Débit", en: "Throughput" },
  { id: "workload", fr: "Charge et SLO", en: "Load and SLOs" },
  { id: "sizing", fr: "Dimensionnement", en: "Sizing" },
  { id: "compare", fr: "Comparaison", en: "Comparison" },
  { id: "import", fr: "Importer un modèle", en: "Import a model" },
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
    // Unavailable storage (e.g. private browsing) is nonfatal.
  }
}

const usable = (s: Support) => s.level !== "unsupported";

/** Automatic adjustment message, kept in both languages so it follows a language switch. */
type Notice = Record<Lang, string>;
const notice = (fr: string, en: string): Notice => ({ fr, en });

export function Estimator({ source, configPath, initial }: { source: ConfigSource; configPath: string; initial: QueryParams }) {
  const { lang, t, fmt } = usePrefs();
  const config = useMemo(() => localize(source, lang), [source, lang]);
  const [imported, setImported] = useState<ModelSource[]>([]);
  const importedModels = useMemo(() => localize(imported, lang), [imported, lang]);
  const models = useMemo(() => [...config.models, ...importedModels], [config.models, importedModels]);
  const [s, setS] = useState<Scenario>(() => scenarioFromQuery(config, config.models, initial));
  const [notices, setNotices] = useState<Notice[]>([]);
  const [pinned, setPinned] = useState<Scenario[]>([]);
  const [tab, setTab] = useState<TabId>("detail");
  const [linkCopied, setLinkCopied] = useState(false);

  // --- Browser-specific data (after hydration)
  useEffect(() => {
    const imp = readStorage<ModelSource[]>(STORAGE.imported, []);
    setImported(imp);
    setPinned(readStorage<Scenario[]>(STORAGE.pinned, []));
    const t = readStorage<TabId>(STORAGE.tab, "detail");
    if (TABS.some((x) => x.id === t)) setTab(t);
    // Link to a model imported in this browser.
    if (initial.model && imp.some((m) => m.id === initial.model)) setS(scenarioFromQuery(config, [...config.models, ...localize(imp, lang)], initial));
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

  // --- Changes with automatic adjustment
  const dpAllowed = (m: ModelSpec, e: EngineSpec) => isMoE(m) && e.features.dpAttention !== false;

  /** After changing GPU or engine, keep the selected format when possible; restore the native format after a fallback. */
  const adjustFormats = (m: ModelSpec, g: GpuSpec, e: EngineSpec, msgs: Notice[]) => {
    let nextWf = pickWeightFormat(config, m, g, e, s.wf);
    const native = config.weightFormats.find((f) => f.id === m.nativeWeightFormat);
    if (nextWf === s.wf && native && wf.bitsPerParam > native.bitsPerParam && usable(weightSupport(g, e, native.id)) && s.wf !== native.id) {
      nextWf = native.id;
      msgs.push(notice(`Format natif ${native.label} disponible → ${native.label}.`, `Native ${native.label} format available → ${native.label}.`));
    } else if (nextWf !== s.wf)
      msgs.push(
        notice(
          `Poids ${wf.label} non supportés (${g.name}, ${e.name}) → ${nextWf.toUpperCase()}.`,
          `${wf.label} weights not supported (${g.name}, ${e.name}) → ${nextWf.toUpperCase()}.`,
        ),
      );
    let nextKf = pickKvFormat(config, g, e, s.kf);
    // FP16 fallback when BF16 is unavailable (Volta): restore BF16 when possible.
    if (nextKf === "fp16" && !usable(kvSupport(gpu, engine, "bf16")) && kvSupport(g, e, "bf16").level === "native") nextKf = "bf16";
    if (nextKf !== s.kf)
      msgs.push(
        usable(kvSupport(g, e, s.kf))
          ? notice("KV BF16 disponible → BF16.", "BF16 KV available → BF16.")
          : notice(`KV ${kf.label} non supporté → ${nextKf.toUpperCase()}.`, `${kf.label} KV not supported → ${nextKf.toUpperCase()}.`),
      );
    return { wf: nextWf, kf: nextKf };
  };

  const setModel = (id: string) => {
    const m = models.find((x) => x.id === id)!;
    const msgs: Notice[] = [];
    const nextWf = pickWeightFormat(config, m, gpu, engine);
    if (nextWf !== m.nativeWeightFormat)
      msgs.push(
        notice(
          `Format natif ${m.nativeWeightFormat.toUpperCase()} indisponible → ${nextWf.toUpperCase()}.`,
          `Native ${m.nativeWeightFormat.toUpperCase()} format unavailable → ${nextWf.toUpperCase()}.`,
        ),
      );
    let mode = s.mode;
    if (mode === "dp-ep" && !dpAllowed(m, engine)) {
      mode = "tp-pp";
      msgs.push(notice("DP-attention réservé aux MoE → TP/PP.", "DP-attention is for MoE models only → TP/PP."));
    }
    setNotices(msgs);
    setS({ ...s, modelId: id, wf: nextWf, mode, tp: defaultTp(m, gpu, s.gpuCount) });
  };
  const setGpu = (id: string) => {
    const g = config.gpus.find((x) => x.id === id)!;
    const msgs: Notice[] = [];
    const counts = gpuCountOptions(config, g);
    const n = counts.includes(s.gpuCount) ? s.gpuCount : 8;
    if (n !== s.gpuCount) msgs.push(notice(`${s.gpuCount} GPUs non proposé pour ${g.name} → ${n}.`, `${s.gpuCount} GPUs not offered for ${g.name} → ${n}.`));
    const f = adjustFormats(model, g, engine, msgs);
    // Custom visible memory belongs to one GPU type: reset it when switching.
    if (s.memGiB !== undefined)
      msgs.push(
        notice(`Mémoire visible personnalisée (${s.memGiB} Gio) réinitialisée pour ${g.name}.`, `Custom visible memory (${s.memGiB} GiB) reset for ${g.name}.`),
      );
    setNotices(msgs);
    setS({ ...s, gpuId: id, gpuCount: n, tp: defaultTp(model, g, n), ...f, memGiB: undefined, priceUsd: undefined });
  };
  const setEngine = (id: string) => {
    const e = config.engines.find((x) => x.id === id)!;
    const msgs: Notice[] = [];
    const f = adjustFormats(model, gpu, e, msgs);
    let mode = s.mode;
    if (mode === "dp-ep" && !dpAllowed(model, e)) {
      mode = "tp-pp";
      msgs.push(notice(`DP-attention indisponible dans ${e.name} → TP/PP.`, `DP-attention unavailable in ${e.name} → TP/PP.`));
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
    const sameGpu = c.gpuId === s.gpuId;
    setNotices([
      notice("Configuration appliquée depuis le dimensionnement.", "Configuration applied from sizing."),
      ...(!sameGpu && s.memGiB !== undefined
        ? [notice("Mémoire visible personnalisée réinitialisée (autre carte).", "Custom visible memory reset (different card).")]
        : []),
    ]);
    setS({ ...s, ...c, memGiB: sameGpu ? s.memGiB : undefined, priceUsd: sameGpu ? s.priceUsd : undefined });
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
  const addImported = (source: ModelSource) => {
    const next = [...imported.filter((x) => x.id !== source.id), { ...source, family: source.family || IMPORTED_FAMILY }];
    setImported(next);
    writeStorage(STORAGE.imported, next);
    const m = localize(source, lang);
    const nextWf = pickWeightFormat(config, m, gpu, engine);
    setNotices([notice(`${m.name} ajouté à la liste des modèles.`, `${m.name} added to the model list.`)]);
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

  // --- Calculations
  const input = useMemo(() => resolveScenario(config, models, { ...s, modelId: model.id }, lang)!, [config, models, s, model.id, lang]);
  const est = useMemo(() => estimate(input), [input]);
  const issues = useMemo(() => collectIssues(input, est, config.weightFormats), [input, est, config.weightFormats]);
  /** Blocking error (format, parallelism, context, etc.): keep the figures but do not mark them as successful. */
  const blocking = issues.some((i) => i.severity === "error");
  const perf = useMemo(() => estimatePerf(input, est, config.performance), [input, est, config.performance]);
  const byContext = useMemo(() => {
    const steps = [...new Set([...CONTEXT_STEPS, ctx])].sort((a, b) => a - b);
    return steps.map((t) => ({ t, e: estimate({ ...input, contextTokens: t }) }));
  }, [input, ctx]);
  const command = launchCommand(input, est.tp, est.pp);

  // --- Options
  const weightOptions: ChipOption[] = config.weightFormats.map((f) => {
    const sup = weightSupport(gpu, engine, f.id, lang);
    const official = model.officialWeightFormats.includes(f.id);
    const bits = f.id === model.nativeWeightFormat && model.nativeBitsPerParam ? model.nativeBitsPerParam : f.bitsPerParam;
    const mem = sup.memoryFormat && sup.memoryFormat !== f.id ? config.weightFormats.find((x) => x.id === sup.memoryFormat) : undefined;
    return {
      id: f.id,
      label: f.label,
      sub: mem ? t(`→ ${mem.label} en mémoire`, `→ ${mem.label} in memory`) : `${bits} bits${official ? t(" · officiel", " · official") : ""}`,
      level: sup.level,
      star: f.id === model.nativeWeightFormat,
      warn: model.weightFormatNotes[f.id] !== undefined,
      title: [f.description, sup.level !== "native" ? supportLevelLabel(sup.level, lang) : null, sup.note, model.weightFormatNotes[f.id]?.note].filter(Boolean).join("\n"),
    };
  });
  const kvOptions: ChipOption[] = config.kvCacheFormats.map((f) => {
    const sup = kvSupport(gpu, engine, f.id, lang);
    return {
      id: f.id,
      label: f.label,
      sub: `${f.bytesPerElement * 8} bits`,
      level: sup.level,
      warn: model.kvFormatNotes[f.id] !== undefined,
      title: [f.description, sup.level !== "native" ? supportLevelLabel(sup.level, lang) : null, sup.note].filter(Boolean).join("\n"),
    };
  });
  const countOptions: ChipOption[] = gpuCountOptions(config, gpu).map((n) => {
    const reasons: string[] = [];
    if (n > 1 && !gpu.interconnect.nvlink) reasons.push(t("pas de liaison GPU↔GPU dédiée", "no dedicated GPU↔GPU link"));
    if (n > gpu.maxGpusPerNode) reasons.push(t(`${Math.ceil(n / gpu.maxGpusPerNode)} nœuds`, `${Math.ceil(n / gpu.maxGpusPerNode)} nodes`));
    const sameDomain = gpu.interconnect.nvlink && n <= gpu.nvlinkDomain;
    return {
      id: String(n),
      label: String(n),
      warn: n > 1 && (!gpu.interconnect.nvlink || (n > gpu.maxGpusPerNode && !sameDomain)),
      title: [reasons.join(" · "), sameDomain && n > gpu.maxGpusPerNode ? t(`même domaine ${gpu.interconnect.fabric}`, `same ${gpu.interconnect.fabric} domain`) : ""].filter(Boolean).join(" — ") || undefined,
    };
  });
  const modeOptions: ChipOption[] = [
    { id: "tp-pp", label: "TP / PP", title: t("Tensor parallel dans un étage, pipeline parallel entre étages.", "Tensor parallel within a stage, pipeline parallel across stages.") },
    {
      id: "dp-ep",
      label: "DP-attention + EP",
      level: !dpAllowed(model, engine) ? "unsupported" : engine.features.dpAttention === "unverified" ? "unverified" : undefined,
      title: !isMoE(model)
        ? t("Réservé aux modèles MoE.", "MoE models only.")
        : engine.features.dpAttention === false
          ? t(`Indisponible dans ${engine.name}.`, `Unavailable in ${engine.name}.`)
          : t("Attention répliquée, requêtes réparties par GPU, experts en expert-parallel.", "Replicated attention, requests spread across GPUs, experts in expert parallel.") +
            (engine.features.dpAttention === "unverified" ? t(` Non vérifié dans ${engine.name}.`, ` Not verified in ${engine.name}.`) : ""),
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
        ? t(`${model.attentionHeads} têtes d'attention / têtes KV non divisibles par ${d}`, `${model.attentionHeads} attention heads / KV heads not divisible by ${d}`)
        : crosses
          ? gpu.interconnect.nvlink
            ? t(`TP au-delà du domaine ${gpu.interconnect.fabric}`, `TP beyond the ${gpu.interconnect.fabric} domain`)
            : t("TP au-delà du domaine du serveur", "TP beyond the server")
          : d > 1 && !gpu.interconnect.nvlink
            ? t("TP via PCIe : lent", "TP over PCIe: slow")
            : undefined,
    };
  });
  const engineOptions: ChipOption[] = config.engines.map((e) => ({
    id: e.id,
    label: e.name,
    sub: e.confidence === "high" ? undefined : e.confidence === "medium" ? t("partiellement vérifié", "partially verified") : t("profil à compléter", "incomplete profile"),
    warn: e.confidence !== "high",
    title: [e.description, ...e.notes].filter(Boolean).join("\n"),
  }));

  const limitationLine = (supports: { label: string; s: Support }[]) => {
    const pick = (l: Support["level"]) => supports.filter((x) => x.s.level === l).map((x) => x.label);
    const groups = [
      { items: pick("unsupported"), cls: "lim-un", icon: "✕", one: t("non supporté", "not supported"), many: t("non supportés", "not supported") },
      { items: pick("limited"), cls: "lim-lim", icon: "⚠", one: t("limité", "limited"), many: t("limités", "limited") },
      { items: pick("unverified"), cls: "lim-unv", icon: "?", one: t("non vérifié", "not verified"), many: t("non vérifiés", "not verified") },
    ].filter((g) => g.items.length);
    if (!groups.length)
      return (
        <p className="hint ok">
          {t("Tous les formats sont supportés nativement", "All formats are natively supported")} ({gpu.name}, {engine.name}).
        </p>
      );
    return (
      <p className="hint">
        {gpu.name} · {engine.name}
        {t(" :", ":")}{" "}
        {groups.map((g, i) => (
          <span key={g.cls}>
            {i > 0 && " · "}
            <span className={g.cls}>
              {g.icon} {g.items.join(", ")} {g.items.length > 1 ? g.many : g.one}
            </span>
          </span>
        ))}
        <span className="muted">{t(" (survol pour le détail)", " (hover for details)")}</span>
      </p>
    );
  };

  const maxCtx = model.maxContextExtended ?? model.maxContext;
  const ov = est.memory.overhead;
  const replicated = est.kv.perRequestClusterBytes > est.kv.perRequestBytes * 1.001;
  const memGiB = s.memGiB ?? gpu.memoryGiB;

  const exportJson = () => {
    const data = { generatedAt: new Date().toISOString(), url: window.location.href, ...buildReport(config, models, s, lang) };
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
        <h1>{t("Estimateur KV cache", "KV cache estimator")}</h1>
        <div className="header-side">
          <PrefsToggles />
          <p className="config-path" title={configPath}>
            {config.models.length} {t("modèles", "models")}
            {imported.length ? ` + ${imported.length} ${t(`importé${imported.length > 1 ? "s" : ""}`, "imported")}` : ""} · {config.gpus.length} GPUs ·{" "}
            {config.engines.length} {t("moteurs", "engines")} · <code>{configPath.split("/").slice(-2).join("/")}</code>
          </p>
        </div>
      </header>

      <main className="layout">
        {/* ------------------------------------------------ Parameters */}
        <section className="panel form" aria-label={t("Paramètres", "Settings")}>
          <div className="field">
            <label className="field-label">{t("Modèle", "Model")}</label>
            <ModelCombobox models={models} value={model.id} onChange={setModel} />
            <ModelCard model={model} />
          </div>

          <div className="field">
            <GpuPicker gpus={config.gpus} value={s.gpuId} onChange={setGpu} />
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
              {t("Nombre de cartes", "Number of cards")} <strong className="value">{s.gpuCount}</strong>
              <span className="muted">
                {" "}
                · {fmt.bytes(memGiB * GIB * s.gpuCount)} {t("au total", "in total")}
              </span>
            </label>
            <ChipGroup
              label={t("Nombre de cartes", "Number of cards")}
              options={countOptions}
              value={String(s.gpuCount)}
              onChange={(v) => update({ gpuCount: Number(v), tp: defaultTp(model, gpu, Number(v)) })}
              className="count"
            />
            {s.gpuCount > 1 && (
              <>
                <label className="field-label sub">{t("Parallélisme", "Parallelism")}</label>
                <ChipGroup label={t("Parallélisme", "Parallelism")} options={modeOptions} value={s.mode} onChange={(v) => update({ mode: v as ParallelMode })} />
                {s.mode === "tp-pp" && (
                  <ChipGroup label={t("Degré de TP", "TP degree")} options={tpOptions} value={String(s.tp)} onChange={(v) => update({ tp: Number(v) })} className="count" />
                )}
              </>
            )}
            <p className={`hint ${gpu.interconnect.nvlink ? "" : "lim-lim"}`}>
              {t("Interconnexion : ", "Interconnect: ")}
              {gpu.interconnect.label}
              {gpu.nvlinkDomain > 8 && t(` · ${gpu.maxGpusPerNode} GPUs par tray`, ` · ${gpu.maxGpusPerNode} GPUs per tray`)}
            </p>
          </div>

          <div className="field">
            <label className="field-label">{t("Moteur d'inférence", "Inference engine")}</label>
            <ChipGroup label={t("Moteur", "Engine")} options={engineOptions} value={s.engineId} onChange={setEngine} />
            {engine.description && <p className="hint">{engine.description}</p>}
            <EngineCapabilities engine={engine} />
          </div>

          <div className="field">
            <label className="field-label">{t("Format des poids", "Weight format")}</label>
            <ChipGroup label={t("Format des poids", "Weight format")} options={weightOptions} value={s.wf} onChange={(v) => update({ wf: v })} />
            {limitationLine(config.weightFormats.map((f) => ({ label: f.label, s: weightSupport(gpu, engine, f.id, lang) })))}
            <p className="hint">
              {t("★ format natif du modèle · « officiel » : checkpoint publié par l'éditeur", "★ native model format · “official”: checkpoint published by the vendor")}
            </p>
          </div>

          <div className="field">
            <label className="field-label">{t("Format du KV cache", "KV cache format")}</label>
            <ChipGroup label={t("Format du KV cache", "KV cache format")} options={kvOptions} value={s.kf} onChange={(v) => update({ kf: v })} />
            {limitationLine(config.kvCacheFormats.map((f) => ({ label: f.label, s: kvSupport(gpu, engine, f.id, lang) })))}
          </div>

          <ContextInput model={model} value={ctx} onChange={(v) => update({ ctx: v })} />

          <div className="field">
            <label className="field-label" htmlFor="reserve">
              {t("Réserve hors moteur par GPU", "Reserve outside the engine per GPU")} <strong className="value">{t(`${s.reservePct} %`, `${s.reservePct}%`)}</strong>
              <span className="muted"> ({fmt.bytes(est.memory.reservePerGpuBytes)})</span>
            </label>
            <input id="reserve" type="range" min={0} max={30} step={1} value={s.reservePct} onChange={(e) => update({ reservePct: Number(e.target.value) })} />
            <p className="hint">
              {ov.inBudget
                ? t("Mémoire laissée hors du moteur : contexte CUDA, autres processus, fragmentation.", "Memory left outside the engine: CUDA context, other processes, fragmentation.")
                : t("Mémoire hors de la fraction statique du moteur : activations, graphes CUDA, contexte CUDA.", "Memory outside the engine's static fraction: activations, CUDA graphs, CUDA context.")}
              {engine.flags.memoryUtilization && (
                <>
                  {" "}
                  {t("Équivaut à", "Equivalent to")} <code>{engine.flags.memoryUtilization} {(1 - s.reservePct / 100).toFixed(2)}</code>.
                </>
              )}
            </p>
            <p className={`hint${!ov.inBudget && ov.totalBytes > est.memory.reservePerGpuBytes ? " lim-lim" : ""}`}>
              {t("Activations et runtime", "Activations and runtime")} <strong>{fmt.bytes(ov.totalBytes)}</strong>
              {ov.overridden
                ? t(" (valeur saisie)", " (entered value)")
                : t(
                    ` (pic d'activations ${fmt.bytes(ov.activationBytes)} pour ${fmt.int(ov.batchedTokens)} tokens par étape, logits ${fmt.bytes(ov.samplerBytes)}, fixe ${fmt.bytes(ov.fixedBytes)})`,
                    ` (activation peak ${fmt.bytes(ov.activationBytes)} for ${fmt.int(ov.batchedTokens)} tokens per step, logits ${fmt.bytes(ov.samplerBytes)}, fixed ${fmt.bytes(ov.fixedBytes)})`,
                  )}
              {ov.inBudget
                ? t(`, pris dans la fraction de ${engine.name}.`, `, taken from ${engine.name}'s fraction.`)
                : t(", à couvrir par la réserve.", ", to be covered by the reserve.")}
            </p>
          </div>

          <details className="field advanced" open={s.memGiB !== undefined || s.offloadGiB > 0 || s.batchedTokens !== undefined || s.overheadGiB !== undefined}>
            <summary className="field-label">{t("Options avancées", "Advanced options")}</summary>
            <label className="inline-form">
              <span>{t("Mémoire visible par GPU", "Visible memory per GPU")}</span>
              <input
                type="number"
                step="0.1"
                min={1}
                value={s.memGiB ?? ""}
                placeholder={String(gpu.memoryGiB)}
                onChange={(e) => update({ memGiB: e.target.value ? Number(e.target.value) : undefined })}
              />
              <span className="muted">{t(`Gio (${gpu.vendor === "amd" ? "amd-smi" : "nvidia-smi"} MiB ÷ 1024)`, `GiB (${gpu.vendor === "amd" ? "amd-smi" : "nvidia-smi"} MiB ÷ 1024)`)}</span>
            </label>
            <label className="inline-form">
              <span>{t("Tokens par étape", "Tokens per step")}</span>
              <input
                type="number"
                min={256}
                step={1024}
                value={s.batchedTokens ?? ""}
                placeholder={String(runtimeDefaults(engine, gpu, s.memGiB).maxNumBatchedTokens)}
                onChange={(e) => update({ batchedTokens: Number(e.target.value) > 0 ? Math.round(Number(e.target.value)) : undefined })}
              />
              <span className="muted">{engine.flags.maxNumBatchedTokens ?? t("prefill fractionné", "chunked prefill")}</span>
            </label>
            <label className="inline-form">
              <span>{t("Activations et runtime mesurés", "Measured activations and runtime")}</span>
              <input
                type="number"
                min={0}
                step={0.1}
                value={s.overheadGiB ?? ""}
                placeholder={(est.memory.overhead.overridden ? 0 : est.memory.overhead.totalBytes / GIB).toFixed(1)}
                onChange={(e) => update({ overheadGiB: e.target.value === "" ? undefined : Math.max(0, Number(e.target.value)) })}
              />
              <span className="muted">{t("Gio par GPU (remplace l'estimation)", "GiB per GPU (replaces the estimate)")}</span>
            </label>
            <label className="inline-form">
              <span>{t("Offload KV vers la RAM CPU", "KV offload to CPU RAM")}</span>
              <input type="number" min={0} step={16} value={s.offloadGiB || ""} placeholder="0" onChange={(e) => update({ offloadGiB: Math.max(0, Number(e.target.value) || 0) })} />
              <span className="muted">{t("Gio par GPU", "GiB per GPU")}</span>
            </label>
            {gpu.hostMemoryGiBPerGpu > 0 && s.offloadGiB !== gpu.hostMemoryGiBPerGpu && (
              <button type="button" className="link" onClick={() => update({ offloadGiB: gpu.hostMemoryGiBPerGpu })}>
                {t(`Utiliser la mémoire Grace (${fmt.int(gpu.hostMemoryGiBPerGpu)} Gio par GPU)`, `Use Grace memory (${fmt.int(gpu.hostMemoryGiBPerGpu)} GiB per GPU)`)}
              </button>
            )}
            {engine.features.kvOffload === false && (
              <p className="hint lim-lim">{t(`Offload non disponible dans ${engine.name}.`, `Offload unavailable in ${engine.name}.`)}</p>
            )}
            {engine.features.kvOffload === "unverified" && (
              <p className="hint lim-unv">{t(`Offload non vérifié dans ${engine.name}.`, `Offload not verified in ${engine.name}.`)}</p>
            )}
            <p className="hint">
              {t(`Lien CPU↔GPU : ${gpu.hostLinkLabel} (${gpu.hostLinkGBs} Go/s).`, `CPU↔GPU link: ${gpu.hostLinkLabel} (${gpu.hostLinkGBs} GB/s).`)}{" "}
              {engine.flags.kvOffload && t(`Option : ${engine.flags.kvOffload}.`, `Option: ${engine.flags.kvOffload}.`)}
            </p>
          </details>
        </section>

        {/* ------------------------------------------------ Results */}
        <section className="results" aria-label={t("Résultats", "Results")}>
          <div className="toolbar">
            <button type="button" className="btn" onClick={pin} disabled={pinned.length >= 6} title={t("Ajouter à la comparaison", "Add to the comparison")}>
              📌 {t("Épingler", "Pin")}
            </button>
            <button type="button" className="btn" onClick={copyLink}>
              {linkCopied ? t("✓ Lien copié", "✓ Link copied") : t("🔗 Copier le lien", "🔗 Copy link")}
            </button>
            <button type="button" className="btn" onClick={exportJson}>
              ⬇ Export JSON
            </button>
          </div>

          <div className="kpis">
            <Kpi
              title={t("Poids du modèle", "Model weights")}
              value={fmt.bytes(est.weights.totalBytes)}
              sub={[
                input.memoryWeightFormat && input.memoryWeightFormat.id !== wf.id
                  ? t(`fichier ${wf.label} → ${input.memoryWeightFormat.label} en VRAM`, `${wf.label} file → ${input.memoryWeightFormat.label} in VRAM`)
                  : null,
                s.gpuCount > 1
                  ? t(`${fmt.bytes(est.weights.perGpuBytes)} par GPU`, `${fmt.bytes(est.weights.perGpuBytes)} per GPU`)
                  : t(`${wf.label} · ${fmt.params(model.paramsB)} paramètres`, `${wf.label} · ${fmt.params(model.paramsB)} parameters`),
              ]
                .filter(Boolean)
                .join(" · ")}
              tone={est.fitsWeights ? undefined : "bad"}
            />
            <Kpi
              title={t("KV par requête", "KV per request")}
              value={fmt.bytes(est.kv.perRequestBytes)}
              sub={
                replicated
                  ? t(`${fmt.bytes(est.kv.perRequestClusterBytes)} réels (réplication)`, `${fmt.bytes(est.kv.perRequestClusterBytes)} actual (replication)`)
                  : `${contextLabel(ctx)} tokens · ${kf.label}`
              }
              tone={replicated ? "warn" : undefined}
            />
            <Kpi
              title={t("Modèle + 1 KV", "Model + 1 KV")}
              value={fmt.bytes(est.physicalOneRequestBytes)}
              sub={[
                est.physicalOneRequestBytes > est.modelPlusOneKvBytes * 1.001
                  ? t(`occupation réelle · somme logique ${fmt.bytes(est.modelPlusOneKvBytes)}`, `actual footprint · logical sum ${fmt.bytes(est.modelPlusOneKvBytes)}`)
                  : null,
                blocking
                  ? t(
                      `capacité mémoire théorique (${est.fitsOneRequest ? "tiendrait" : "ne tient pas"}) — configuration invalide`,
                      `theoretical memory capacity (${est.fitsOneRequest ? "would fit" : "does not fit"}) — invalid configuration`,
                    )
                  : `${est.fitsOneRequest ? t("✓ tient", "✓ fits") : t("✕ ne tient pas", "✕ does not fit")} (${
                      s.gpuCount > 1
                        ? `${t("GPU le plus chargé : ", "most loaded GPU: ")}${fmt.bytes(est.weights.perGpuBytes + est.kv.perRequestPerGpuBytes)} / ${fmt.bytes(est.memory.usablePerGpuBytes)}`
                        : `${fmt.bytes(est.memory.usablePerGpuBytes)} ${t("utilisables", "usable")}`
                    })`,
              ]
                .filter(Boolean)
                .join(" · ")}
              tone={blocking || !est.fitsOneRequest ? "bad" : "good"}
            />
            <Kpi
              title={t("KV total disponible", "Total available KV")}
              value={fmt.bytes(est.memory.freeForKvTotalBytes)}
              big={`${fmt.int(est.sessions)} session${est.sessions > 1 ? "s" : ""}${blocking ? t(" (théorique)", " (theoretical)") : ""}`}
              sub={[
                blocking ? t("capacité mémoire théorique — configuration invalide", "theoretical memory capacity — invalid configuration") : null,
                est.memory.kvBudgetTotalBytes < est.memory.freeForKvTotalBytes * 0.999
                  ? t(
                      `mémoire libre physique · budget prudent ${fmt.bytes(est.memory.kvBudgetTotalBytes)} (GPU le plus chargé × ${s.gpuCount})`,
                      `physical free memory · conservative budget ${fmt.bytes(est.memory.kvBudgetTotalBytes)} (most loaded GPU × ${s.gpuCount})`,
                    )
                  : null,
                est.sessions > 0
                  ? t(
                      `à ${contextLabel(ctx)} · ≈ ${fmt.tokens(est.sessions * ctx)} tokens cumulés`,
                      `at ${contextLabel(ctx)} · ≈ ${fmt.tokens(est.sessions * ctx)} tokens in total`,
                    ) + (est.offload.parkedSessions ? t(` · +${fmt.int(est.offload.parkedSessions)} en RAM CPU`, ` · +${fmt.int(est.offload.parkedSessions)} in CPU RAM`) : "")
                  : t("aucune session complète", "no complete session"),
              ]
                .filter(Boolean)
                .join(" · ")}
              tone={blocking || est.sessions === 0 ? "bad" : "good"}
            />
          </div>

          <button type="button" className="perf-strip" onClick={() => selectTab("perf")} title={t("Voir le détail du débit", "See the throughput details")}>
            <span>
              ⚡ {t("Débit estimé", "Estimated throughput")}
              {blocking ? t(" (théorique, configuration invalide)", " (theoretical, invalid configuration)") : ""}
              {t(" : ", ": ")}
              <strong>{fmt.rate(perf.single.tokPerSecPerSession)}</strong> {t("pour 1 session", "for 1 session")}
            </span>
            {perf.full && (
              <span>
                · <strong>{fmt.rate(perf.full.tokPerSecTotal)}</strong> {t("agrégés à pleine charge", "aggregate at full load")}
              </span>
            )}
            <span>
              · prefill {contextLabel(ctx)} ≈ <strong>{fmt.duration(perf.prefillSeconds)}</strong>
            </span>
          </button>

          <MemoryBar est={est} gpuCount={s.gpuCount} />

          <IssuesPanel issues={issues} notices={notices.map((n) => n[lang])} />

          <div className="panel">
            <div className="tabs" role="tablist">
              {TABS.map((x) => (
                <button key={x.id} type="button" role="tab" aria-selected={tab === x.id} className={`tab${tab === x.id ? " active" : ""}`} onClick={() => selectTab(x.id)}>
                  {x[lang]}
                  {x.id === "compare" && pinned.length > 0 && <span className="badge">{pinned.length}</span>}
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
                      <h3>
                        {t("Commande indicative", "Indicative command")}
                        {command.matchesScenario ? "" : t(" (checkpoint à fournir)", " (checkpoint to supply)")}
                      </h3>
                      <pre className="yaml">{command.text}</pre>
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
              {tab === "workload" && (
                <WorkloadPanel
                  config={config}
                  input={input}
                  workload={s.workload}
                  priceUsd={s.priceUsd}
                  onChange={(workload, priceUsd) => update({ workload, priceUsd })}
                  onApply={applySizing}
                />
              )}
              {tab === "sizing" && <SizingPanel config={config} input={input} onApply={applySizing} />}
              {tab === "compare" && (
                <ComparePanel config={config} models={models} current={s} pinned={pinned} onPin={pin} onRemove={unpin} onLoad={loadScenario} />
              )}
              {tab === "import" && (
                <ImportPanel existingIds={new Set(models.map((m) => m.id))} imported={importedModels} onAdd={addImported} onClear={clearImported} />
              )}
            </div>
          </div>

          <details className="panel assumptions">
            <summary>{t("Hypothèses de calcul", "Calculation assumptions")}</summary>
            <ul>
              {ASSUMPTIONS.map((a) => (
                <li key={a.en}>{a[lang]}</li>
              ))}
            </ul>
            <h3>{t("Non modélisé", "Not modelled")}</h3>
            <ul className="unmodelled">
              {UNMODELLED[lang].map((u) => (
                <li key={u}>{u}</li>
              ))}
            </ul>
          </details>
        </section>
      </main>
    </div>
  );
}

const ASSUMPTIONS: Notice[] = [
  notice(
    "Poids = paramètres × bits effectifs du format (échelles incluses). Embeddings, lm_head et encodeur vision restent en 16 bits si on quantifie.",
    "Weights = parameters × effective bits of the format (scales included). Embeddings, lm_head and the vision encoder stay at 16 bits when quantizing.",
  ),
  notice(
    "KV GQA = couches × têtes KV × (dim K + dim V) × octets × tokens. Fenêtres glissantes plafonnées à la fenêtre si le moteur le gère.",
    "GQA KV = layers × KV heads × (K dim + V dim) × bytes × tokens. Sliding windows capped at the window if the engine supports it.",
  ),
  notice(
    "MLA = couches × (rang latent + dim RoPE) × octets × tokens (+ indexeur DSA en FP8), sans facteur 2.",
    "MLA = layers × (latent rank + RoPE dim) × bytes × tokens (+ DSA indexer in FP8), without the factor of 2.",
  ),
  notice(
    "Attention compressée (DeepSeek V4) = couches × dim × octets × (tokens / ratio + fenêtre), K = V, + indexeur par entrée compressée.",
    "Compressed attention (DeepSeek V4) = layers × dim × bytes × (tokens / ratio + window), K = V, + indexer per compressed entry.",
  ),
  notice(
    "Couches linéaires (Mamba, DeltaNet, KDA) : état SSM + convolution fixes par requête, en FP32 si le modèle l'impose.",
    "Linear layers (Mamba, DeltaNet, KDA): fixed SSM + convolution state per request, in FP32 if the model requires it.",
  ),
  notice(
    "TP : têtes KV réparties, répliquées si TP > têtes KV ; latent MLA copié sur chaque rang. PP : couches réparties sans duplication.",
    "TP: KV heads sharded, replicated if TP > KV heads; MLA latent copied on every rank. PP: layers distributed without duplication.",
  ),
  notice(
    "DP-attention + EP : attention répliquée sur chaque GPU, experts répartis, chaque requête sur un seul GPU.",
    "DP-attention + EP: attention replicated on every GPU, experts sharded, each request on a single GPU.",
  ),
  notice(
    "Mémoire du moteur = mémoire visible − réserve. vLLM y prend aussi le pic d'activations (tokens par étape × largeur MLP estimée), les logits de l'échantillonneur et un fixe runtime ; SGLang les prend dans la réserve.",
    "Engine memory = visible memory − reserve. vLLM also takes the activation peak (tokens per step × estimated MLP width), sampler logits and a fixed runtime amount from it; SGLang takes them from the reserve.",
  ),
  notice(
    "Sessions = ⌊mémoire libre par GPU ÷ KV par requête et par GPU⌋, toutes au contexte maximal (pire cas). L'onglet Charge et SLO utilise la longueur moyenne des requêtes.",
    "Sessions = ⌊free memory per GPU ÷ KV per request per GPU⌋, all at maximum context (worst case). The Load and SLOs tab uses the average request length.",
  ),
  notice(
    "Offload CPU : le KV déchargé ne sert qu'aux sessions en attente ou préemptées, il n'augmente pas la concurrence de décodage.",
    "CPU offload: offloaded KV only serves waiting or preempted sessions; it does not increase decoding concurrency.",
  ),
  notice(
    "Débit : modèle au premier ordre (bande passante, MFU, latences), à calibrer avec des mesures.",
    "Throughput: first-order model (bandwidth, MFU, latencies), to be calibrated with measurements.",
  ),
  notice(
    "1 Gio = 2³⁰ octets. Mémoire GPU vue par le driver, modifiable dans les options avancées.",
    "1 GiB = 2³⁰ bytes. GPU memory as seen by the driver, adjustable in the advanced options.",
  ),
];

const CAPABILITY_LABELS: Record<keyof EngineSpec["features"], Notice> = {
  kvPaging: notice("KV paginé (PagedAttention)", "Paged KV (PagedAttention)"),
  kvPoolPreallocated: notice("Pool KV préalloué au démarrage", "KV pool preallocated at startup"),
  prefixCaching: notice("Prefix caching", "Prefix caching"),
  continuousBatching: notice("Continuous batching", "Continuous batching"),
  slidingWindowAttention: notice("Attention glissante (noyaux)", "Sliding-window attention (kernels)"),
  slidingWindowFreesBlocks: notice("Libération des blocs hors fenêtre", "Freeing of blocks outside the window"),
  tensorParallel: notice("Tensor parallel", "Tensor parallel"),
  pipelineParallel: notice("Pipeline parallel", "Pipeline parallel"),
  dpAttention: notice("DP-attention + EP", "DP-attention + EP"),
  kvOffload: notice("Offload du KV en RAM CPU", "KV offload to CPU RAM"),
};

function EngineCapabilities({ engine }: { engine: EngineSpec }) {
  const { lang, t } = usePrefs();
  const entries = Object.entries(engine.features) as [keyof EngineSpec["features"], EngineSpec["features"][keyof EngineSpec["features"]]][];
  return (
    <details className="engine-caps">
      <summary className="hint">
        {t(`Capacités de ${engine.name}`, `${engine.name} capabilities`)}
        {engine.version ? ` (${engine.version})` : ""}
        {t(" : ", ": ")}
        {entries.filter(([, v]) => v === true).length} {t("oui", "yes")} · {entries.filter(([, v]) => v === false).length} {t("non", "no")} ·{" "}
        {entries.filter(([, v]) => v === "unverified").length} {t("non vérifiées", "not verified")}
      </summary>
      <table className="caps">
        <tbody>
          {entries.map(([k, v]) => (
            <tr key={k}>
              <td>
                <span className={v === true ? "cap-yes" : v === false ? "cap-no" : "cap-unv"}>{v === true ? "✓" : v === false ? "✕" : "?"}</span> {CAPABILITY_LABELS[k][lang]}
              </td>
              <td className="muted">{engine.featureNotes[k] ?? (v === "unverified" ? t("Non vérifié : aucune source.", "Not verified: no source.") : "")}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {engine.sources.length > 0 && (
        <p className="hint sources">
          Sources{t(" :", ":")}{" "}
          {engine.sources.map((src, i) => (
            <span key={src.url}>
              {i > 0 && " · "}
              <a href={src.url} target="_blank" rel="noreferrer">
                {src.label}
              </a>
            </span>
          ))}
        </p>
      )}
    </details>
  );
}
