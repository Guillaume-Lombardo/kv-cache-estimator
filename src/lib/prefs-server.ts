import { cookies, headers } from "next/headers";
import { isLang, isTheme, LANG_COOKIE, langFromAcceptLanguage, THEME_COOKIE, type Lang, type Theme } from "./i18n";

/** Language and theme for the server render: cookie first, then the browser's Accept-Language. */
export async function readPrefs(): Promise<{ lang: Lang; theme: Theme }> {
  const jar = await cookies();
  const lang = jar.get(LANG_COOKIE)?.value;
  const theme = jar.get(THEME_COOKIE)?.value;
  return {
    lang: isLang(lang) ? lang : langFromAcceptLanguage((await headers()).get("accept-language")),
    theme: isTheme(theme) ? theme : "system",
  };
}
