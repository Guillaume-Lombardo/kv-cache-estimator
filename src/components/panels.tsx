"use client";

import { contextLabel, type Estimate, type Issue } from "@/lib/calc";
import type { ModelSpec } from "@/lib/config-schema";
import { fmtBytes, fmtInt, fmtParams } from "@/lib/format";
import { archLabel, TAG_LABELS } from "./ModelCombobox";

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
  const layers = m.layers.reduce((a, g) => a + g.count, 0);
  return (
    <div className="model-card">
      <div className="badges">
        <span className="tag">{archLabel(m)}</span>
        {m.tags
          .filter((t) => t !== "swa" && t !== "hybrid")
          .map((t) => (
            <span key={t} className="tag">
              {TAG_LABELS[t] ?? t}
            </span>
          ))}
        {m.family === "Importé" || m.notes.includes("Importé depuis config.json.") ? <span className="tag tag-accent">importé</span> : null}
        {m.confidence !== "high" && (
          <span className={`tag ${m.confidence === "low" ? "tag-warn" : ""}`} title="Paramètres d'architecture à vérifier dans config.json">
            fiabilité {m.confidence === "low" ? "faible" : "moyenne"}
          </span>
        )}
        {m.revision && (
          <span className="tag tag-ok" title={`config.json vérifié, révision ${m.revision}`}>
            config vérifiée
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
          <dd>
            {m.nativeWeightFormat.toUpperCase()}
            {m.nativeBitsPerParam && <span className="muted"> ({m.nativeBitsPerParam} bits)</span>}
          </dd>
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
        <a className="source" href={`https://huggingface.co/${m.source}${m.revision ? `/blob/${m.revision}/config.json` : ""}`} target="_blank" rel="noreferrer">
          {m.source} ↗
        </a>
      )}
    </div>
  );
}

export function MemoryBar({ est, gpuCount }: { est: Estimate; gpuCount: number }) {
  const gpuBytes = est.memory.gpuBytes;
  const reserve = est.memory.reservePerGpuBytes;
  const kvUsed = est.sessionsPerGpu * est.kv.perRequestPerGpuBytes;
  const free = est.memory.freeForKvPerGpuBytes - kvUsed;
  const overflow = Math.max(0, est.weights.perGpuBytes - (gpuBytes - reserve));
  const scale = gpuBytes + overflow;
  const pct = (b: number) => `${(Math.max(0, b) / scale) * 100}%`;
  const segs = [
    { key: "weights", label: "Poids", bytes: Math.min(est.weights.perGpuBytes, gpuBytes - reserve) },
    { key: "overflow", label: "Dépassement", bytes: overflow },
    { key: "kv", label: `KV (${fmtInt(est.sessionsPerGpu)} session${est.sessionsPerGpu > 1 ? "s" : ""})`, bytes: kvUsed },
    { key: "free", label: "Libre (< 1 session)", bytes: free },
    { key: "reserve", label: "Réserve", bytes: reserve },
  ].filter((s) => s.bytes > 0);
  return (
    <div className="panel">
      <h2>
        Mémoire d&apos;un GPU{" "}
        <span className="muted">
          ({fmtBytes(gpuBytes)}
          {gpuCount > 1 ? ` × ${gpuCount}` : ""}
          {gpuCount > 1 ? ` · TP ${est.tp} × PP ${est.pp}` : ""})
        </span>
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

export function IssuesPanel({ issues, notices }: { issues: Issue[]; notices: string[] }) {
  if (!issues.length && !notices.length) return null;
  const hasError = issues.some((i) => i.severity === "error");
  const counts = (["error", "warning", "info"] as const).map((s) => issues.filter((i) => i.severity === s).length);
  return (
    <div className={`panel issues${hasError ? " has-error" : ""}`}>
      <h2>
        Limitations et avertissements{" "}
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
  );
}

export function KvBreakdown({ est, ctx, gpuCount }: { est: Estimate; ctx: number; gpuCount: number }) {
  return (
    <div>
      <h3>KV par requête ({contextLabel(ctx)})</h3>
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>Couches</th>
              <th>Type</th>
              <th>Détail</th>
              <th className="num">Par requête</th>
              {gpuCount > 1 && <th className="num">Par GPU</th>}
            </tr>
          </thead>
          <tbody>
            {est.kv.groups.map((g, i) => (
              <tr key={i}>
                <td>{g.count}</td>
                <td>{g.label}</td>
                <td className="muted">{g.detail}</td>
                <td className="num">{fmtBytes(g.bytes)}</td>
                {gpuCount > 1 && (
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
              {gpuCount > 1 && <td className="num">{fmtBytes(est.kv.perRequestPerGpuBytes)}</td>}
            </tr>
          </tfoot>
        </table>
      </div>
      {est.kv.perTokenBytes > 0 && (
        <p className="hint">
          {fmtInt(est.kv.perTokenBytes / 1024)} Kio par token sur les couches à contexte complet
          {est.kv.constantBytes > 0 && <> · {fmtBytes(est.kv.constantBytes)} d&apos;état constant</>}.
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
  const max = Math.max(1, ...rows.map((r) => r.e.sessions));
  // Échelle logarithmique : les sessions varient d'un facteur ~100 entre 8k et 1M.
  const w = (n: number) => (n <= 0 ? 0 : (Math.log10(n + 1) / Math.log10(max + 1)) * 100);
  return (
    <div>
      <h3>Sessions selon le contexte</h3>
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
            <span className="sval">{fmtInt(r.e.sessions)}</span>
            <span className="skv muted">{fmtBytes(r.e.kv.perRequestBytes)}/req.</span>
          </button>
        ))}
      </div>
      <p className="hint">Échelle logarithmique. Cliquez une ligne pour choisir ce contexte.</p>
    </div>
  );
}
