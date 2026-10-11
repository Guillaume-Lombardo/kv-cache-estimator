import { GIB } from "./calc";

const nf1 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const nf0 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });
const nf2 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 2 });

/** Octets → "12,3 Gio" (ou Mio / Tio selon l'ordre de grandeur). */
export function fmtBytes(bytes: number): string {
  if (bytes >= 1024 * GIB) return `${nf2.format(bytes / (1024 * GIB))} Tio`;
  if (bytes >= GIB || bytes === 0) return `${nf1.format(bytes / GIB)} Gio`;
  return `${nf0.format(bytes / 1024 ** 2)} Mio`;
}

export const fmtInt = (n: number) => nf0.format(n);
export const fmtNum = (n: number) => nf2.format(n);

export function fmtParams(b: number): string {
  return b >= 1000 ? `${nf2.format(b / 1000)} T` : `${nf2.format(b)} B`;
}

export function fmtTokens(n: number): string {
  if (n >= 1e9) return `${nf1.format(n / 1e9)} G`;
  if (n >= 1e6) return `${nf1.format(n / 1e6)} M`;
  if (n >= 1e3) return `${nf0.format(n / 1e3)} k`;
  return nf0.format(n);
}

export function fmtRate(tokPerSec: number): string {
  if (tokPerSec >= 1e5) return `${nf0.format(tokPerSec / 1e3)} k tok/s`;
  if (tokPerSec >= 100) return `${nf0.format(tokPerSec)} tok/s`;
  return `${nf1.format(tokPerSec)} tok/s`;
}

export function fmtDuration(seconds: number): string {
  if (seconds < 1) return `${nf0.format(seconds * 1000)} ms`;
  if (seconds < 120) return `${nf1.format(seconds)} s`;
  if (seconds < 7200) return `${nf1.format(seconds / 60)} min`;
  return `${nf1.format(seconds / 3600)} h`;
}
