import { Estimator, type InitialState } from "@/components/Estimator";
import { loadConfig } from "@/lib/load-config";

// Le fichier de configuration est lu côté serveur, au démarrage (voir load-config.ts).
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
  const str = (k: string) => (typeof sp[k] === "string" ? (sp[k] as string) : undefined);
  const num = (k: string) => {
    const v = Number(str(k));
    return Number.isFinite(v) && str(k) !== undefined ? v : undefined;
  };
  const initial: InitialState = {
    model: str("model"),
    ctx: num("ctx"),
    gpu: str("gpu"),
    n: num("n"),
    w: str("w"),
    kv: str("kv"),
    r: num("r"),
    p: str("p"),
  };

  return <Estimator config={result.config} configPath={result.path} initial={initial} />;
}
