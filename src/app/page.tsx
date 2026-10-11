import { Estimator } from "@/components/Estimator";
import { loadConfig } from "@/lib/load-config";
import type { QueryParams } from "@/lib/scenario";

// Read the configuration on the server at startup (see load-config.ts).
export const dynamic = "force-dynamic";

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

export default async function Page({ searchParams }: { searchParams: SearchParams }) {
  const result = await loadConfig();
  if (!result.ok) {
    return (
      <main className="config-error">
        <h1>Fichier de configuration invalide</h1>
        <p>
          <code>{result.path}</code>
        </p>
        <pre>{result.error}</pre>
        <p>Corrigez le fichier (ou KV_ESTIMATOR_CONFIG) puis relancez l&apos;application.</p>
      </main>
    );
  }

  const sp = await searchParams;
  const initial: QueryParams = Object.fromEntries(Object.entries(sp).map(([k, v]) => [k, Array.isArray(v) ? v[0] : v]));
  return <Estimator config={result.config} configPath={result.path} initial={initial} />;
}
