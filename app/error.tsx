'use client';
export default function ErrorPage({ reset }: { reset: () => void }) {
  return <main className="error-page"><h1>Le catalogue n’a pas pu être chargé.</h1><p>Vérifiez le fichier ESTIMATOR_CONFIG et lancez <code>npm run validate:config</code>. Les détails sont dans les journaux du serveur.</p><button onClick={reset}>Réessayer</button></main>;
}
