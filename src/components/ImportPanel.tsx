"use client";

import { useState } from "react";
import { modelSchema, type ModelSource, type ModelSpec } from "@/lib/config-schema";
import { modelToYaml, type ImportResult } from "@/lib/hf-import";
import { z } from "zod";
import { buildImportPreview, type ImportPreview } from "@/lib/import-preview";
import { usePrefs } from "./Prefs";

const PARAMS_SOURCE: Record<ImportResult["paramsSource"], { fr: string; en: string }> = {
  option: { fr: "option", en: "option" },
  safetensors: { fr: "métadonnées safetensors", en: "safetensors metadata" },
  "repo-name": { fr: "nom du dépôt", en: "repository name" },
  estimate: { fr: "estimation", en: "estimate" },
};

export function ImportPanel({
  existingIds,
  imported,
  onAdd,
  onClear,
}: {
  existingIds: Set<string>;
  imported: ModelSpec[];
  onAdd: (m: ModelSource) => void;
  onClear: () => void;
}) {
  const { lang, t, fmt } = usePrefs();
  const [text, setText] = useState("");
  const [source, setSource] = useState("");
  const [params, setParams] = useState("");
  const [active, setActive] = useState("");
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const result = preview?.[lang].result ?? null;
  const error = preview?.[lang].error ?? validationError;
  const [copied, setCopied] = useState(false);
  const [fetching, setFetching] = useState(false);

  const convert = (raw = text) => {
    setCopied(false);
    setValidationError(null);
    // Store both translations of this submission; switching language must not submit pending edits.
    setPreview(buildImportPreview(raw, {
      source: source.trim() || undefined,
      paramsB: params ? Number(params) : undefined,
      activeParamsB: active ? Number(active) : undefined,
    }, existingIds));
  };

  /** Download config.json and the parameter count through the server (/api/hf-model). */
  const fetchFromHub = async () => {
    const repo = source.trim();
    if (!repo) return;
    setFetching(true);
    setValidationError(null);
    try {
      const res = await fetch(`${process.env.KV_ESTIMATOR_PUBLIC_BASE_PATH ?? ""}/api/hf-model?repo=${encodeURIComponent(repo)}`);
      const body = await res.json();
      if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`);
      const raw = JSON.stringify(body, null, 2);
      setText(raw);
      convert(raw);
    } catch (e) {
      setPreview(null);
      setValidationError(t(`Téléchargement impossible : ${(e as Error).message}`, `Download failed: ${(e as Error).message}`));
    } finally {
      setFetching(false);
    }
  };

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    const content = await f.text();
    setText(content);
    convert(content);
  };

  const yaml = result ? modelToYaml(result.model) : "";
  const add = () => {
    if (!result) return;
    const parsed = modelSchema.safeParse(result.model);
    if (!parsed.success) {
      setValidationError(z.prettifyError(parsed.error));
      return;
    }
    onAdd(parsed.data);
  };

  return (
    <div>
      <h3>{t("Importer un modèle", "Import a model")}</h3>
      <div className="inline-form hub-fetch">
        <label htmlFor="hf-repo">{t("Dépôt Hugging Face", "Hugging Face repository")}</label>
        <input
          id="hf-repo"
          type="text"
          value={source}
          onChange={(e) => setSource(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && fetchFromHub()}
          placeholder="Qwen/Qwen3-8B"
          spellCheck={false}
        />
        <button type="button" className="btn primary" onClick={fetchFromHub} disabled={!source.trim() || fetching}>
          {fetching ? t("Téléchargement…", "Downloading…") : t("Récupérer", "Fetch")}
        </button>
      </div>
      <p className="hint">
        {t(
          "Le serveur télécharge le config.json et le nombre de paramètres (safetensors) depuis huggingface.co. Dépôts restreints : définissez HF_TOKEN côté serveur. Sans accès réseau, collez le fichier ci-dessous.",
          "The server downloads config.json and the parameter count (safetensors) from huggingface.co. Gated repositories: set HF_TOKEN on the server. Without network access, paste the file below.",
        )}
      </p>
      <p className="hint">
        {t("Collez le ", "Paste the ")}
        <code>config.json</code>
        {t(" d'un dépôt Hugging Face (ou le ", " of a Hugging Face repository (or Mistral's ")}
        <code>params.json</code>
        {t(
          " de Mistral). L'outil détecte l'architecture (GQA, fenêtre glissante, MLA, DSA, Mamba, DeltaNet, KDA), estime les paramètres si besoin et produit l'entrée YAML à ajouter au fichier de configuration. Équivalent en ligne de commande : ",
          "). The tool detects the architecture (GQA, sliding window, MLA, DSA, Mamba, DeltaNet, KDA), estimates the parameters if needed and produces the YAML entry to add to the configuration file. Command-line equivalent: ",
        )}
        <code>pnpm run import-model config.json --append</code>
        {t(" ou ", " or ")}
        <code>pnpm run import-model --hf org/nom</code>.
      </p>
      <div className="import-grid">
        <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder='{ "architectures": ["…"], "hidden_size": … }' spellCheck={false} rows={10} />
        <div className="import-side">
          <label>
            {t("Fichier", "File")}
            <input type="file" accept=".json,application/json" onChange={(e) => onFile(e.target.files?.[0])} />
          </label>
          <label>
            {t("Paramètres totaux (B)", "Total parameters (B)")}
            <input type="number" value={params} onChange={(e) => setParams(e.target.value)} placeholder="auto" />
          </label>
          <label>
            {t("Paramètres actifs (B)", "Active parameters (B)")}
            <input type="number" value={active} onChange={(e) => setActive(e.target.value)} placeholder="auto" />
          </label>
          <button type="button" className="btn primary" onClick={() => convert()} disabled={!text.trim()}>
            {t("Convertir", "Convert")}
          </button>
        </div>
      </div>
      {error && <pre className="error-box">{error}</pre>}
      {result && (
        <div className="import-result">
          <p>
            <strong>{result.model.name}</strong> — {fmt.params(result.model.paramsB)} {t("paramètres", "parameters")} ({PARAMS_SOURCE[result.paramsSource][lang]})
            {result.model.activeParamsB ? `, ${fmt.params(result.model.activeParamsB)} ${t("actifs", "active")}` : ""} ·{" "}
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
              {copied ? t("✓ Copié", "✓ Copied") : t("Copier le YAML", "Copy the YAML")}
            </button>
            <button type="button" className="btn primary" onClick={add}>
              {t("Utiliser dans l'estimateur", "Use in the estimator")}
            </button>
          </div>
          <p className="hint">
            {t(
              "« Utiliser » ajoute le modèle à la liste pour ce navigateur uniquement. Pour le partager, collez le YAML sous ",
              "“Use” adds the model to the list for this browser only. To share it, paste the YAML under ",
            )}
            <code>models:</code>
            {t(" dans le fichier de configuration.", " in the configuration file.")}
          </p>
        </div>
      )}
      {imported.length > 0 && (
        <p className="hint">
          {t("Modèles importés dans ce navigateur : ", "Models imported in this browser: ")}
          {imported.map((m) => m.name).join(", ")}.{" "}
          <button type="button" className="link" onClick={onClear}>
            {t("Tout retirer", "Remove all")}
          </button>
        </p>
      )}
    </div>
  );
}
