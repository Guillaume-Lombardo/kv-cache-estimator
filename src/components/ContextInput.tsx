"use client";

import { useEffect, useState } from "react";
import { CONTEXT_STEPS, contextLabel, MAX_CONTEXT, MIN_CONTEXT, parseContext } from "@/lib/calc";
import type { ModelSpec } from "@/lib/config-schema";
import { usePrefs } from "./Prefs";

const nearestStep = (tokens: number) =>
  CONTEXT_STEPS.reduce((best, s, i) => (Math.abs(Math.log2(s / tokens)) < Math.abs(Math.log2(CONTEXT_STEPS[best] / tokens)) ? i : best), 0);

/** Context length: any value, typed or picked from the presets and model limits. */
export function ContextInput({ model, value, onChange }: { model: ModelSpec; value: number; onChange: (tokens: number) => void }) {
  const { t, fmt } = usePrefs();
  const [text, setText] = useState(contextLabel(value));
  useEffect(() => setText(contextLabel(value)), [value]);
  const parsed = parseContext(text);
  const valid = text === contextLabel(value) || (parsed !== null && parsed >= MIN_CONTEXT && parsed <= MAX_CONTEXT);
  const commit = () => {
    // The displayed label ("40k" for 40960) does not round-trip through the parser: leave it alone.
    if (text === contextLabel(value)) return;
    if (valid && parsed !== null && parsed !== value) onChange(parsed);
    else setText(contextLabel(value));
  };
  const maxCtx = model.maxContextExtended ?? model.maxContext;
  const limits = [
    { tokens: model.maxContext, label: t("max natif", "native max") },
    ...(model.maxContextExtended ? [{ tokens: model.maxContextExtended, label: t("max étendu", "extended max") }] : []),
  ];

  return (
    <div className="field">
      <label className="field-label" htmlFor="ctx">
        {t("Fenêtre de contexte", "Context window")} <strong className="value">{contextLabel(value)}</strong>
        <span className="muted"> ({fmt.int(value)} tokens)</span>
      </label>
      <input
        type="range"
        min={0}
        max={CONTEXT_STEPS.length - 1}
        step={1}
        value={nearestStep(value)}
        aria-label={t("Fenêtre de contexte (paliers)", "Context window (presets)")}
        onChange={(e) => onChange(CONTEXT_STEPS[Number(e.target.value)])}
        className={value > maxCtx ? "range-error" : value > model.maxContext ? "range-warn" : ""}
      />
      <div className="ticks">
        {CONTEXT_STEPS.map((step) => (
          <button
            key={step}
            type="button"
            className={`tick${step === value ? " current" : ""}${step > maxCtx ? " over" : step > model.maxContext ? " ext" : ""}`}
            onClick={() => onChange(step)}
            title={
              step > maxCtx
                ? t("Au-delà du maximum du modèle", "Beyond the model maximum")
                : step > model.maxContext
                  ? t("Nécessite une extension RoPE (YaRN)", "Requires a RoPE extension (YaRN)")
                  : undefined
            }
          >
            {contextLabel(step)}
          </button>
        ))}
      </div>
      <div className="ctx-row">
        <input
          id="ctx"
          type="text"
          inputMode="numeric"
          className={valid ? "" : "invalid"}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => e.key === "Enter" && commit()}
          aria-invalid={!valid}
          title={t("Valeur libre : 200000, 200k, 1,5M…", "Any value: 200000, 200k, 1.5M…")}
        />
        {limits.map((l) => (
          <button key={l.label} type="button" className={`btn${value === l.tokens ? " primary" : ""}`} onClick={() => onChange(l.tokens)}>
            {l.label} {contextLabel(l.tokens)}
          </button>
        ))}
      </div>
      <p className="hint">
        {t("Valeur libre (", "Any value (")}
        <code>--max-model-len</code>
        {t(") ou paliers.", ") or presets.")}
        <span className="legend-ext"> ■ extension</span>
        <span className="legend-over"> ■ {t("hors limite", "out of range")}</span>
      </p>
    </div>
  );
}
