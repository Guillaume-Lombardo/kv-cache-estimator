"use client";

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { formatters, type Formatters } from "@/lib/format";
import { LANG_COOKIE, THEME_COOKIE, TITLE, tr, type Lang, type Theme, type Translate } from "@/lib/i18n";

interface Prefs {
  lang: Lang;
  theme: Theme;
  setLang: (l: Lang) => void;
  setTheme: (t: Theme) => void;
  /** t("Français", "English") */
  t: Translate;
  fmt: Formatters;
}

const PrefsContext = createContext<Prefs | null>(null);

function writeCookie(name: string, value: string) {
  document.cookie = `${name}=${value}; path=/; max-age=31536000; samesite=lax`;
}

/** Applies the theme to <html>: no attribute follows the system setting (prefers-color-scheme). */
function applyTheme(theme: Theme) {
  if (theme === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
}

export function PrefsProvider({ initialLang, initialTheme, children }: { initialLang: Lang; initialTheme: Theme; children: ReactNode }) {
  const [lang, setLangState] = useState(initialLang);
  const [theme, setThemeState] = useState(initialTheme);
  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    writeCookie(LANG_COOKIE, l);
    document.documentElement.lang = l;
    document.title = TITLE[l];
  }, []);
  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    writeCookie(THEME_COOKIE, t);
    applyTheme(t);
  }, []);
  const value = useMemo(() => ({ lang, theme, setLang, setTheme, t: tr(lang), fmt: formatters(lang) }), [lang, theme, setLang, setTheme]);
  return <PrefsContext.Provider value={value}>{children}</PrefsContext.Provider>;
}

export function usePrefs(): Prefs {
  const p = useContext(PrefsContext);
  if (!p) throw new Error("usePrefs() outside PrefsProvider");
  return p;
}

const ICON_PROPS = { width: 14, height: 14, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" } as const;

const THEME_OPTIONS: { id: Theme; icon: ReactNode; fr: string; en: string }[] = [
  {
    id: "light",
    fr: "Thème clair",
    en: "Light theme",
    icon: (
      <svg {...ICON_PROPS} aria-hidden>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
      </svg>
    ),
  },
  {
    id: "dark",
    fr: "Thème sombre",
    en: "Dark theme",
    icon: (
      <svg {...ICON_PROPS} aria-hidden>
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
      </svg>
    ),
  },
  {
    id: "system",
    fr: "Thème du système",
    en: "System theme",
    icon: (
      <svg {...ICON_PROPS} aria-hidden>
        <rect x="2" y="3" width="20" height="14" rx="2" />
        <path d="M8 21h8M12 17v4" />
      </svg>
    ),
  },
];

/** Language switch and light / dark / system theme selector. */
export function PrefsToggles() {
  const { lang, setLang, theme, setTheme, t } = usePrefs();
  const other: Lang = lang === "fr" ? "en" : "fr";
  return (
    <div className="prefs">
      <button
        type="button"
        className="btn lang-toggle"
        onClick={() => setLang(other)}
        lang={other}
        title={other === "en" ? "Switch to English" : "Passer en français"}
        aria-label={other === "en" ? "Switch to English" : "Passer en français"}
      >
        <span className={lang === "fr" ? "on" : ""}>FR</span>
        <span className={lang === "en" ? "on" : ""}>EN</span>
      </button>
      <div className="segmented" role="radiogroup" aria-label={t("Thème", "Theme")}>
        {THEME_OPTIONS.map((o) => (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={theme === o.id}
            className={theme === o.id ? "on" : ""}
            onClick={() => setTheme(o.id)}
            title={t(o.fr, o.en)}
            aria-label={t(o.fr, o.en)}
          >
            {o.icon}
          </button>
        ))}
      </div>
    </div>
  );
}
