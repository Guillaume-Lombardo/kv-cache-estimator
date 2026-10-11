export const LANGS = ["fr", "en"] as const;
export type Lang = (typeof LANGS)[number];
export const DEFAULT_LANG: Lang = "fr";

export const THEMES = ["system", "light", "dark"] as const;
export type Theme = (typeof THEMES)[number];

/** Cookies read on the server so the first render already uses the chosen language and theme. */
export const LANG_COOKIE = "kv-lang";
export const THEME_COOKIE = "kv-theme";

export const TITLE = { fr: "Estimateur KV cache", en: "KV cache estimator" } as const;
export const DESCRIPTION = {
  fr: "Mémoire des poids et du KV cache des LLM open-weight sur GPU NVIDIA",
  en: "Weight and KV cache memory for open-weight LLMs on NVIDIA GPUs",
} as const;

export const isLang = (v: unknown): v is Lang => LANGS.includes(v as Lang);
export const isTheme = (v: unknown): v is Theme => THEMES.includes(v as Theme);

/** Picks the French or English variant: `const t = tr(lang); t("Poids", "Weights")`. */
export const tr =
  (lang: Lang) =>
  <T>(fr: T, en: T): T =>
    lang === "en" ? en : fr;
export type Translate = ReturnType<typeof tr>;

/** Language from an Accept-Language header, French when nothing matches. */
export function langFromAcceptLanguage(header: string | null | undefined): Lang {
  for (const part of (header ?? "").split(",")) {
    const code = part.trim().slice(0, 2).toLowerCase();
    if (isLang(code)) return code;
  }
  return DEFAULT_LANG;
}

/** Per-language text, as accepted in the configuration file. */
export type LocalizedText = { fr: string; en: string };

/** Type of a value after localize(): every LocalizedText becomes a string. */
export type Localized<T> = T extends string | number | boolean | null | undefined
  ? T
  : T extends LocalizedText
    ? string
    : T extends readonly (infer U)[]
      ? Localized<U>[]
      : { [K in keyof T]: Localized<T[K]> };

const isLocalizedText = (v: object): v is LocalizedText => {
  const keys = Object.keys(v);
  return keys.length === 2 && typeof (v as LocalizedText).fr === "string" && typeof (v as LocalizedText).en === "string";
};

/** Resolves every { fr, en } text in a value (configuration, model) to one language. */
export function localize<T>(value: T, lang: Lang): Localized<T> {
  if (value === null || typeof value !== "object") return value as Localized<T>;
  if (Array.isArray(value)) return value.map((v) => localize(v, lang)) as Localized<T>;
  if (isLocalizedText(value)) return value[lang] as Localized<T>;
  return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, localize(v, lang)])) as Localized<T>;
}
