"use client";

import { contextLabel, type Estimate, type Issue } from "@/lib/calc";
import type { ModelSpec } from "@/lib/config-schema";
import { archLabel, IMPORTED_FAMILY, tagLabel } from "./ModelCombobox";
import { usePrefs } from "./Prefs";

const IMPORT_NOTES = ["Importé depuis config.json.", "Imported from config.json."];

const SEVERITY_ICON: Record<Issue["severity"], string> = { error: "⛔", warning: "⚠️", info: "ℹ️" };

export function Kpi({ title, value, sub, big, tone }: { title: string; value: string; sub?: string; big?: string; tone?: "good" | "bad" | "warn" }) {
  return (
    <div className={`kpi${tone ? ` ${tone}` : ""}`}>
      <div className="kpi-title">{title}</div>
      <div className="kpi-value">{value}</div>
      {big && <div className="kpi-big">{big}</div>}
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

export function ModelCard({ model: m }: { model: ModelSpec }) {
  const { lang, t, fmt } = usePrefs();
  const layers = m.layers.reduce((a, g) => a + g.count, 0);
  return (
    <div className="model-card">
      <div className="badges">
        <span className="tag">{archLabel(m, lang)}</span>
        {m.tags
          .filter((tag) => tag !== "swa" && tag !== "hybrid")
          .map((tag) => (
            <span key={tag} className="tag">
              {tagLabel(tag, lang)}
            </span>
          ))}
        {m.family === IMPORTED_FAMILY || m.notes.some((n) => IMPORT_NOTES.includes(n)) ? <span className="tag tag-accent">{t("importé", "imported")}</span> : null}
        {m.confidence !== "high" && (
          <span
            className={`tag ${m.confidence === "low" ? "tag-warn" : ""}`}
            title={t("Paramètres d'architecture à vérifier dans config.json", "Architecture parameters to be checked in config.json")}
          >
            {m.confidence === "low" ? t("fiabilité faible", "low confidence") : t("fiabilité moyenne", "medium confidence")}
          </span>
        )}
        {m.revision && (
          <span className="tag tag-ok" title={t(`config.json vérifié, révision ${m.revision}`, `config.json verified, revision ${m.revision}`)}>
            {t("config vérifiée", "verified config")}
          </span>
        )}
      </div>
      <dl>
        <div>
          <dt>{t("Paramètres", "Parameters")}</dt>
          <dd>
            {fmt.params(m.paramsB)}
            {m.activeParamsB && (
              <span className="muted">
                {" "}
                ({fmt.params(m.activeParamsB)} {t("actifs", "active")})
              </span>
            )}
          </dd>
        </div>
        <div>
          <dt>{t("Couches à cache", "Cache layers")}</dt>
          <dd>{layers}</dd>
        </div>
        <div>
          <dt>{t("Format natif", "Native format")}</dt>
          <dd>
            {m.nativeWeightFormat.toUpperCase()}
            {m.nativeBitsPerParam && <span className="muted"> ({m.nativeBitsPerParam} bits)</span>}
          </dd>
        </div>
        <div>
          <dt>{t("Officiels", "Official")}</dt>
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
        <a className="source" href={`https://huggingface.co/${m.source}${m.revision ? `/blob/${m.revision}/config.json` : ""}`} target="_blank" rel="noreferrer">
          {m.source} ↗
        </a>
      )}
    </div>
  );
}

export function MemoryBar({ est, gpuCount }: { est: Estimate; gpuCount: number }) {
  const { t, fmt } = usePrefs();
  const gpuBytes = est.memory.gpuBytes;
  const reserve = est.memory.reservePerGpuBytes;
  const kvUsed = est.sessionsPerGpu * est.kv.perRequestPerGpuBytes;
  const free = est.memory.freeForKvPerGpuBytes - kvUsed;
  const overflow = Math.max(0, est.weights.perGpuBytes - (gpuBytes - reserve));
  const scale = gpuBytes + overflow;
  const pct = (b: number) => `${(Math.max(0, b) / scale) * 100}%`;
  const segs = [
    { key: "weights", label: t("Poids", "Weights"), bytes: Math.min(est.weights.perGpuBytes, gpuBytes - reserve) },
    { key: "overflow", label: t("Dépassement", "Overflow"), bytes: overflow },
    { key: "kv", label: `KV (${fmt.int(est.sessionsPerGpu)} session${est.sessionsPerGpu > 1 ? "s" : ""})`, bytes: kvUsed },
    { key: "free", label: t("Libre (< 1 session)", "Free (< 1 session)"), bytes: free },
    { key: "reserve", label: t("Réserve", "Reserve"), bytes: reserve },
  ].filter((s) => s.bytes > 0);
  return (
    <div className="panel">
      <h2>
        {t("Mémoire d'un GPU", "Memory of one GPU")}{" "}
        <span className="muted">
          ({fmt.bytes(gpuBytes)}
          {gpuCount > 1 ? ` × ${gpuCount}` : ""}
          {gpuCount > 1 ? ` · TP ${est.tp} × PP ${est.pp}` : ""})
        </span>
      </h2>
      <div className="membar" role="img" aria-label={t("Répartition de la mémoire d'un GPU", "Memory breakdown of one GPU")}>
        {segs.map((s) => (
          <div key={s.key} className={`seg seg-${s.key}`} style={{ width: pct(s.bytes) }} title={`${s.label}${t(" : ", ": ")}${fmt.bytes(s.bytes)}`} />
        ))}
      </div>
      <div className="legend">
        {segs.map((s) => (
          <span key={s.key}>
            <i className={`dot seg-${s.key}`} /> {s.label} <strong>{fmt.bytes(s.bytes)}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

export function IssuesPanel({ issues, notices }: { issues: Issue[]; notices: string[] }) {
  const { t } = usePrefs();
  if (!issues.length && !notices.length) return null;
  const hasError = issues.some((i) => i.severity === "error");
  const counts = (["error", "warning", "info"] as const).map((s) => issues.filter((i) => i.severity === s).length);
  return (
    <div className={`panel issues${hasError ? " has-error" : ""}`}>
      <h2>
        {t("Limitations et avertissements", "Limitations and warnings")}{" "}
        <span className="muted">
          {counts[0] > 0 && `⛔ ${counts[0]} `}
          {counts[1] > 0 && `⚠️ ${counts[1]} `}
          {counts[2] > 0 && `ℹ️ ${counts[2]}`}
        </span>
      </h2>
      <ul>
        {notices.map((n) => (
          <li key={n} className="issue notice">
            <span aria-hidden>🔁</span>
            <div>
              <strong>{t("Ajusté automatiquement", "Adjusted automatically")}</strong>
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
  );
}

export function KvBreakdown({ est, ctx, gpuCount }: { est: Estimate; ctx: number; gpuCount: number }) {
  const { t, fmt } = usePrefs();
  return (
    <div>
      <h3>
        {t("KV par requête", "KV per request")} ({contextLabel(ctx)})
      </h3>
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>{t("Couches", "Layers")}</th>
              <th>Type</th>
              <th>{t("Détail", "Detail")}</th>
              <th className="num">{t("Par requête", "Per request")}</th>
              {gpuCount > 1 && <th className="num">{t("Par GPU", "Per GPU")}</th>}
            </tr>
          </thead>
          <tbody>
            {est.kv.groups.map((g, i) => (
              <tr key={i}>
                <td>{g.count}</td>
                <td>{g.label}</td>
                <td className="muted">{g.detail}</td>
                <td className="num">{fmt.bytes(g.bytes)}</td>
                {gpuCount > 1 && (
                  <td className="num">
                    {fmt.bytes(g.perGpuBytes)}
                    {g.replication > 1 && <span className="tag tag-warn">×{g.replication} {t("répliqué", "replicated")}</span>}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={2}>Total</td>
              <td />
              <td className="num">{fmt.bytes(est.kv.perRequestBytes)}</td>
              {gpuCount > 1 && <td className="num">{fmt.bytes(est.kv.perRequestPerGpuBytes)}</td>}
            </tr>
          </tfoot>
        </table>
      </div>
      {est.kv.perTokenBytes > 0 && (
        <p className="hint">
          {t(
            `${fmt.int(est.kv.perTokenBytes / 1024)} Kio par token sur les couches à contexte complet`,
            `${fmt.int(est.kv.perTokenBytes / 1024)} KiB per token on full-context layers`,
          )}
          {est.kv.constantBytes > 0 && t(` · ${fmt.bytes(est.kv.constantBytes)} d'état constant`, ` · ${fmt.bytes(est.kv.constantBytes)} of constant state`)}.
        </p>
      )}
    </div>
  );
}

export function SessionsChart({
  rows,
  current,
  maxNative,
  maxExt,
  onPick,
}: {
  rows: { t: number; e: Estimate }[];
  current: number;
  maxNative: number;
  maxExt: number;
  onPick: (t: number) => void;
}) {
  const { t, fmt } = usePrefs();
  const max = Math.max(1, ...rows.map((r) => r.e.sessions));
  // Logarithmic scale: session counts vary by roughly 100x between 8k and 1M.
  const w = (n: number) => (n <= 0 ? 0 : (Math.log10(n + 1) / Math.log10(max + 1)) * 100);
  return (
    <div>
      <h3>{t("Sessions selon le contexte", "Sessions by context")}</h3>
      <div className="sessions">
        {rows.map((r) => (
          <button
            key={r.t}
            type="button"
            className={`srow${r.t === current ? " current" : ""}${r.t > maxExt ? " over" : r.t > maxNative ? " ext" : ""}`}
            onClick={() => onPick(r.t)}
          >
            <span className="slabel">{contextLabel(r.t)}</span>
            <span className="sbar">
              <span style={{ width: `${w(r.e.sessions)}%` }} />
            </span>
            <span className="sval">{fmt.int(r.e.sessions)}</span>
            <span className="skv muted">{fmt.bytes(r.e.kv.perRequestBytes)}/req.</span>
          </button>
        ))}
      </div>
      <p className="hint">{t("Échelle logarithmique. Cliquez une ligne pour choisir ce contexte.", "Logarithmic scale. Click a row to select that context.")}</p>
    </div>
  );
}
