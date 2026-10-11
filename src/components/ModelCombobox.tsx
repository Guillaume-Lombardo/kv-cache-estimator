"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { ModelSpec } from "@/lib/config-schema";
import { contextLabel } from "@/lib/calc";
import { fmtParams } from "@/lib/format";

export const TAG_LABELS: Record<string, string> = {
  moe: "MoE",
  vision: "Vision",
  code: "Code",
  reasoning: "Raisonnement",
  hybrid: "Hybride",
  swa: "SWA",
};

/** Architecture summary displayed as a badge. */
export function archLabel(m: ModelSpec): string {
  const kinds = new Set(m.layers.map((g) => g.kind));
  if (kinds.has("linear")) return kinds.has("mla") ? "Linéaire + MLA" : "Linéaire + GQA";
  if (kinds.has("mla")) return "MLA";
  if (kinds.has("sliding")) return "SWA + GQA";
  return "GQA";
}

function haystack(m: ModelSpec): string {
  return [m.name, m.id, m.family, archLabel(m), ...m.tags, ...m.tags.map((t) => TAG_LABELS[t] ?? t), `${m.paramsB}b`]
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
        aria-label="Modèle"
        placeholder={selected ? selected.name : "Rechercher un modèle…"}
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
          {filtered.length === 0 && <li className="combo-empty">Aucun modèle pour « {query} »</li>}
          {filtered.map((m, i) => {
            const header = m.family !== lastFamily ? m.family : null;
            lastFamily = m.family;
            return (
              <li key={m.id} role="presentation">
                {header && <div className="combo-group">{header}</div>}
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
                    {fmtParams(m.paramsB)}
                    {m.activeParamsB ? ` · ${fmtParams(m.activeParamsB)} actifs` : ""} · {archLabel(m)} ·{" "}
                    {contextLabel(m.maxContext)}
                    {m.confidence === "low" && <span className="tag tag-warn" title="Paramètres à vérifier">à vérifier</span>}
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
