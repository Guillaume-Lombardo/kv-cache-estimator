"use client";

import { useState } from "react";
import { modelSchema, type ModelSpec } from "@/lib/config-schema";
import { convertHfConfig, modelToYaml, type ImportResult } from "@/lib/hf-import";
import { fmtParams } from "@/lib/format";
import { z } from "zod";

export function ImportPanel({
  existingIds,
  imported,
  onAdd,
  onClear,
}: {
  existingIds: Set<string>;
  imported: ModelSpec[];
  onAdd: (m: ModelSpec) => void;
  onClear: () => void;
}) {
  const [text, setText] = useState("");
  const [source, setSource] = useState("");
  const [params, setParams] = useState("");
  const [active, setActive] = useState("");
  const [result, setResult] = useState<ImportResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const convert = (raw = text) => {
    setCopied(false);
    try {
      const json = JSON.parse(raw);
      const r = convertHfConfig(json, {
        source: source.trim() || undefined,
        paramsB: params ? Number(params) : undefined,
        activeParamsB: active ? Number(active) : undefined,
      });
      if (existingIds.has(r.model.id)) r.model.id = `${r.model.id}-import`;
      setResult(r);
      setError(null);
    } catch (e) {
      setResult(null);
      setError((e as Error).message);
    }
  };

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    const t = await f.text();
    setText(t);
    convert(t);
  };

  const yaml = result ? modelToYaml(result.model) : "";
  const add = () => {
    if (!result) return;
    const parsed = modelSchema.safeParse(result.model);
    if (!parsed.success) {
      setError(z.prettifyError(parsed.error));
      return;
    }
    onAdd(parsed.data);
  };

  return (
    <div>
      <h3>Importer un config.json</h3>
      <p className="hint">
        Collez le <code>config.json</code> d&apos;un dépôt Hugging Face (ou le <code>params.json</code> de Mistral). L&apos;outil détecte l&apos;architecture (GQA, fenêtre
        glissante, MLA, DSA, Mamba, DeltaNet, KDA), estime les paramètres si besoin et produit l&apos;entrée YAML à ajouter au fichier de configuration. Équivalent en ligne
        de commande : <code>npm run import-model -- config.json --append</code>.
      </p>
      <div className="import-grid">
        <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder='{ "architectures": ["…"], "hidden_size": … }' spellCheck={false} rows={10} />
        <div className="import-side">
          <label>
            Fichier
            <input type="file" accept=".json,application/json" onChange={(e) => onFile(e.target.files?.[0])} />
          </label>
          <label>
            Dépôt (org/nom)
            <input type="text" value={source} onChange={(e) => setSource(e.target.value)} placeholder="Qwen/Qwen3-8B" />
          </label>
          <label>
            Paramètres totaux (B)
            <input type="number" value={params} onChange={(e) => setParams(e.target.value)} placeholder="auto" />
          </label>
          <label>
            Paramètres actifs (B)
            <input type="number" value={active} onChange={(e) => setActive(e.target.value)} placeholder="auto" />
          </label>
          <button type="button" className="btn primary" onClick={() => convert()} disabled={!text.trim()}>
            Convertir
          </button>
        </div>
      </div>
      {error && <pre className="error-box">{error}</pre>}
      {result && (
        <div className="import-result">
          <p>
            <strong>{result.model.name}</strong> — {fmtParams(result.model.paramsB)} paramètres ({result.paramsSource})
            {result.model.activeParamsB ? `, ${fmtParams(result.model.activeParamsB)} actifs` : ""} ·{" "}
            {result.model.layers.map((l) => `${l.count} ${l.kind}`).join(" + ")}
          </p>
          {result.warnings.length > 0 && (
            <ul className="warn-list">
              {result.warnings.map((w) => (
                <li key={w}>⚠️ {w}</li>
              ))}
            </ul>
          )}
          <pre className="yaml">{yaml}</pre>
          <div className="row-gap">
            <button
              type="button"
              className="btn"
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(yaml);
                  setCopied(true);
                } catch {
                  setCopied(false);
                }
              }}
            >
              {copied ? "✓ Copié" : "Copier le YAML"}
            </button>
            <button type="button" className="btn primary" onClick={add}>
              Utiliser dans l&apos;estimateur
            </button>
          </div>
          <p className="hint">
            « Utiliser » ajoute le modèle à la liste pour ce navigateur uniquement. Pour le partager, collez le YAML sous <code>models:</code> dans le fichier de configuration.
          </p>
        </div>
      )}
      {imported.length > 0 && (
        <p className="hint">
          Modèles importés dans ce navigateur : {imported.map((m) => m.name).join(", ")}.{" "}
          <button type="button" className="link" onClick={onClear}>
            Tout retirer
          </button>
        </p>
      )}
    </div>
  );
}
