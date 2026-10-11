import type { Metadata } from "next";
import { PrefsProvider } from "@/components/Prefs";
import { DESCRIPTION, TITLE } from "@/lib/i18n";
import { readPrefs } from "@/lib/prefs-server";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const { lang } = await readPrefs();
  return { title: TITLE[lang], description: DESCRIPTION[lang] };
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const { lang, theme } = await readPrefs();
  return (
    <html lang={lang} data-theme={theme === "system" ? undefined : theme}>
      <body>
        <PrefsProvider initialLang={lang} initialTheme={theme}>
          {children}
        </PrefsProvider>
      </body>
    </html>
  );
}
