import { GIB } from "./calc";
import type { Lang } from "./i18n";

const UNITS = {
  fr: { kib: "Mio", gib: "Gio", tib: "Tio" },
  en: { kib: "MiB", gib: "GiB", tib: "TiB" },
} as const;

/** Number and unit formatting for one UI language ("12,3 Gio" in French, "12.3 GiB" in English). */
export function formatters(lang: Lang) {
  const locale = lang === "en" ? "en-US" : "fr-FR";
  const nf1 = new Intl.NumberFormat(locale, { maximumFractionDigits: 1, minimumFractionDigits: 1 });
  const nf0 = new Intl.NumberFormat(locale, { maximumFractionDigits: 0 });
  const nf2 = new Intl.NumberFormat(locale, { maximumFractionDigits: 2 });
  const u = UNITS[lang];

  /** Bytes to GiB (or MiB / TiB depending on magnitude). */
  const bytes = (b: number): string => {
    if (b >= 1024 * GIB) return `${nf2.format(b / (1024 * GIB))} ${u.tib}`;
    if (b >= GIB || b === 0) return `${nf1.format(b / GIB)} ${u.gib}`;
    return `${nf0.format(b / 1024 ** 2)} ${u.kib}`;
  };
  const int = (n: number) => nf0.format(n);
  const num = (n: number) => nf2.format(n);
  const params = (b: number): string => (b >= 1000 ? `${nf2.format(b / 1000)} T` : `${nf2.format(b)} B`);
  const tokens = (n: number): string => {
    if (n >= 1e9) return `${nf1.format(n / 1e9)} G`;
    if (n >= 1e6) return `${nf1.format(n / 1e6)} M`;
    if (n >= 1e3) return `${nf0.format(n / 1e3)} k`;
    return nf0.format(n);
  };
  const rate = (tokPerSec: number): string => {
    if (tokPerSec >= 1e5) return `${nf0.format(tokPerSec / 1e3)} k tok/s`;
    if (tokPerSec >= 100) return `${nf0.format(tokPerSec)} tok/s`;
    return `${nf1.format(tokPerSec)} tok/s`;
  };
  const duration = (seconds: number): string => {
    if (seconds < 1) return `${nf0.format(seconds * 1000)} ms`;
    if (seconds < 120) return `${nf1.format(seconds)} s`;
    if (seconds < 7200) return `${nf1.format(seconds / 60)} min`;
    return `${nf1.format(seconds / 3600)} h`;
  };
  return { bytes, int, num, params, tokens, rate, duration, gib: u.gib };
}
export type Formatters = ReturnType<typeof formatters>;
