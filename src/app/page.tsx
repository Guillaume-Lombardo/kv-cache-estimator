import { Estimator } from "@/components/Estimator";
import { tr } from "@/lib/i18n";
import { loadConfig } from "@/lib/load-config";
import { readPrefs } from "@/lib/prefs-server";
import type { QueryParams } from "@/lib/scenario";

// Read the configuration on the server at startup (see load-config.ts).
export const dynamic = "force-dynamic";

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

export default async function Page({ searchParams }: { searchParams: SearchParams }) {
  const result = await loadConfig();
  if (!result.ok) {
    const t = tr((await readPrefs()).lang);
    return (
      <main className="config-error">
        <h1>{t("Fichier de configuration invalide", "Invalid configuration file")}</h1>
        <p>
          <code>{result.path}</code>
        </p>
        <pre>{result.error}</pre>
        <p>{t("Corrigez le fichier (ou KV_ESTIMATOR_CONFIG) puis relancez l'application.", "Fix the file (or KV_ESTIMATOR_CONFIG), then restart the application.")}</p>
      </main>
    );
  }

  const sp = await searchParams;
  const initial: QueryParams = Object.fromEntries(Object.entries(sp).map(([k, v]) => [k, Array.isArray(v) ? v[0] : v]));
  return <Estimator source={result.config} configPath={result.path} initial={initial} />;
}
