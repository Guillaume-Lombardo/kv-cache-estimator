"use client";

import { useEffect, useState } from "react";
import { usePrefs } from "./Prefs";

/** Observe presentation only; scenario state remains in the estimator. */
export function ResultSummary({ fits, blocking, sessions, tokPerSec, pricePerGpu, gpuCount, targetId, loadFirst }: {
  fits: boolean;
  blocking: boolean;
  sessions: number;
  tokPerSec: number;
  pricePerGpu?: number;
  gpuCount: number;
  targetId: string;
  loadFirst: boolean;
}) {
  const { t, fmt } = usePrefs();
  const [pastIndicators, setPastIndicators] = useState(false);
  useEffect(() => {
    const target = document.getElementById(targetId);
    if (!target || loadFirst) return;
    const observer = new IntersectionObserver(([entry]) => {
      setPastIndicators(!entry.isIntersecting && entry.boundingClientRect.bottom <= 12);
    }, { rootMargin: "-12px 0px 0px 0px" });
    observer.observe(target);
    return () => observer.disconnect();
  }, [targetId, loadFirst]);

  const goToResults = () => {
    const target = document.getElementById(targetId);
    target?.scrollIntoView({ block: "start" });
    target?.focus({ preventScroll: true });
  };
  return (
    <div className="result-summary-anchor" data-visible={!loadFirst && pastIndicators}>
      <button type="button" className="result-summary" onClick={goToResults}
        title={t("Voir les résultats détaillés", "See detailed results")}>
        <span className={blocking || !fits ? "lim-un" : "summary-good"}>
          {blocking ? t("✕ Invalide", "✕ Invalid") : fits ? t("✓ Ça tient", "✓ Fits") : t("✕ Ne tient pas", "✕ Does not fit")}
        </span>
        <span><strong>{fmt.int(sessions)}</strong> {t("sessions", "sessions")}{blocking && "*"}</span>
        <span><strong>{fmt.rate(tokPerSec)}</strong><small>{t("1 session · estimé", "1 session · estimated")}{blocking && "*"}</small></span>
        <span><strong>{pricePerGpu === undefined ? "—" : fmt.num(pricePerGpu * gpuCount)} $/h</strong><small>{t("configuration", "configuration")}</small></span>
        {blocking && <small className="summary-warning">{t("* Théorique · configuration invalide", "* Theoretical · invalid configuration")}</small>}
      </button>
    </div>
  );
}
