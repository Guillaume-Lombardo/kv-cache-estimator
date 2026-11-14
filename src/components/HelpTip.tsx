"use client";

import type { ReactNode } from "react";

/** Native disclosure works with keyboard and touch as well as a pointer. */
export function HelpTip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <details className="help-tip">
      <summary aria-label={label}>ⓘ</summary>
      <span className="help-tip-text">{children}</span>
    </details>
  );
}
