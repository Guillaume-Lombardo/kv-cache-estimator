"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { ModelSpec } from "@/lib/config-schema";
import { contextLabel } from "@/lib/calc";
import type { Lang } from "@/lib/i18n";
import { usePrefs } from "./Prefs";

const TAG_LABELS: Record<Lang, Record<string, string>> = {
  fr: { moe: "MoE", vision: "Vision", code: "Code", reasoning: "Raisonnement", hybrid: "Hybride", swa: "SWA" },
  en: { moe: "MoE", vision: "Vision", code: "Code", reasoning: "Reasoning", hybrid: "Hybrid", swa: "SWA" },
};
export const tagLabel = (tag: string, lang: Lang) => TAG_LABELS[lang][tag] ?? tag;

/** Family value given to models imported in the browser. */
export const IMPORTED_FAMILY = "Importé";
export const familyLabel = (family: string, lang: Lang) => (family === IMPORTED_FAMILY && lang === "en" ? "Imported" : family);

/** Architecture summary displayed as a badge. */
export function archLabel(m: ModelSpec, lang: Lang): string {
  const kinds = new Set(m.layers.map((g) => g.kind));
  const linear = lang === "en" ? "Linear" : "Linéaire";
  if (kinds.has("linear")) return kinds.has("mla") ? `${linear} + MLA` : `${linear} + GQA`;
  if (kinds.has("mla")) return "MLA";
  if (kinds.has("sliding")) return "SWA + GQA";
  return "GQA";
}

/** Searchable text, in both languages so either one matches. */
function haystack(m: ModelSpec): string {
  const langs: Lang[] = ["fr", "en"];
  return [m.name, m.id, m.family, ...langs.map((l) => archLabel(m, l)), ...m.tags, ...langs.flatMap((l) => m.tags.map((t) => tagLabel(t, l))), `${m.paramsB}b`]
    .join(" ")
    .toLowerCase();
}

export function ModelCombobox({
  models,
  value,
  onChange,
}: {
  models: ModelSpec[];
  value: string;
  onChange: (id: string) => void;
}) {
  const { lang, t, fmt } = usePrefs();
  const id = useId();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listRef = useRef<HTMLUListElement>(null);
  const selected = models.find((m) => m.id === value);

  const filtered = useMemo(() => {
    const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
    return models.filter((m) => {
      const h = haystack(m);
      return tokens.every((t) => h.includes(t));
    });
  }, [models, query]);

  useEffect(() => {
    listRef.current?.querySelector(`[data-index="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active, open]);

  const choose = (m: ModelSpec) => {
    onChange(m.id);
    setOpen(false);
    setQuery("");
  };

  const openList = () => {
    setOpen(true);
    setQuery("");
    setActive(Math.max(0, models.findIndex((m) => m.id === value)));
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!open && (e.key === "ArrowDown" || e.key === "Enter")) {
      openList();
      e.preventDefault();
      return;
    }
    if (e.key === "ArrowDown") setActive((a) => Math.min(filtered.length - 1, a + 1));
    else if (e.key === "ArrowUp") setActive((a) => Math.max(0, a - 1));
    else if (e.key === "Enter" && filtered[active]) choose(filtered[active]);
    else if (e.key === "Escape") setOpen(false);
    else return;
    e.preventDefault();
  };

  let lastFamily = "";
  return (
    <div className="combo">
      <input
        className="combo-input"
        role="combobox"
        aria-expanded={open}
        aria-controls={`${id}-list`}
        aria-activedescendant={open && filtered[active] ? `${id}-opt-${active}` : undefined}
        aria-autocomplete="list"
        aria-label={t("Modèle", "Model")}
        placeholder={selected ? selected.name : t("Rechercher un modèle…", "Search for a model…")}
        value={open ? query : (selected?.name ?? "")}
        onFocus={openList}
        onClick={() => !open && openList()}
        onBlur={() => setOpen(false)}
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0);
          setOpen(true);
        }}
        onKeyDown={onKeyDown}
      />
      <span className="combo-caret" aria-hidden>
        ▾
      </span>
      {open && (
        <ul className="combo-list" role="listbox" id={`${id}-list`} ref={listRef}>
          {filtered.length === 0 && <li className="combo-empty">{t(`Aucun modèle pour « ${query} »`, `No model matches “${query}”`)}</li>}
          {filtered.map((m, i) => {
            const header = m.family !== lastFamily ? m.family : null;
            lastFamily = m.family;
            return (
              <li key={m.id} role="presentation">
                {header && <div className="combo-group">{familyLabel(header, lang)}</div>}
                <div
                  role="option"
                  id={`${id}-opt-${i}`}
                  data-index={i}
                  aria-selected={m.id === value}
                  className={`combo-option${i === active ? " active" : ""}`}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    choose(m);
                  }}
                  onMouseEnter={() => setActive(i)}
                >
                  <span className="combo-name">{m.name}</span>
                  <span className="combo-meta">
                    {fmt.params(m.paramsB)}
                    {m.activeParamsB ? ` · ${fmt.params(m.activeParamsB)} ${t("actifs", "active")}` : ""} · {archLabel(m, lang)} ·{" "}
                    {contextLabel(m.maxContext)}
                    {m.confidence === "low" && <span className="tag tag-warn" title={t("Paramètres à vérifier", "Parameters to be checked")}>
                        {t("à vérifier", "to check")}
                      </span>}
                  </span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
