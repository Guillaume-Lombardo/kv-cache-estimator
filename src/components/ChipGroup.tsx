"use client";

import type { ReactNode } from "react";
import type { SupportLevel } from "@/lib/config-schema";

export interface ChipOption {
  id: string;
  label: ReactNode;
  sub?: ReactNode;
  /** unsupported : grisé et non cliquable ; limited : badge d'avertissement. */
  level?: SupportLevel;
  /** Avertissement non bloquant spécifique (ex. TP invalide). */
  warn?: boolean;
  title?: string;
  star?: boolean;
}

export function ChipGroup({
  options,
  value,
  onChange,
  label,
  className = "",
}: {
  options: ChipOption[];
  value: string;
  onChange: (id: string) => void;
  label: string;
  className?: string;
}) {
  return (
    <div className={`chips ${className}`} role="radiogroup" aria-label={label}>
      {options.map((o) => {
        const disabled = o.level === "unsupported";
        const checked = o.id === value;
        const cls = [
          "chip",
          checked && "checked",
          disabled && "disabled",
          o.level === "limited" && "limited",
          o.warn && "warn",
        ]
          .filter(Boolean)
          .join(" ");
        return (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={checked}
            aria-disabled={disabled}
            className={cls}
            title={o.title}
            onClick={() => !disabled && onChange(o.id)}
          >
            <span className="chip-label">
              {o.star && (
                <span className="chip-star" aria-label="format natif">
                  ★
                </span>
              )}
              {o.label}
              {o.level === "limited" && <span className="chip-badge">⚠</span>}
              {(o.warn && o.level !== "limited") && <span className="chip-badge">⚠</span>}
              {disabled && <span className="chip-badge">✕</span>}
            </span>
            {o.sub && <span className="chip-sub">{o.sub}</span>}
          </button>
        );
      })}
    </div>
  );
}
