// Print the JSON report of a scenario, with the same parameters as the page URL.
//
// Usage:
//   pnpm run estimate model=qwen3-32b gpu=h100 n=2 ctx=65536 [lang=en] [rps=4 in=2000 out=500 ttft=2 itl=50]
//   pnpm run estimate "model=qwen3-32b&gpu=h100"   (a copied query string also works)
//
// Reads KV_ESTIMATOR_CONFIG like the server. Exit code 2 on unknown identifiers.
import { isLang, localize } from "../src/lib/i18n";
import { configPath, parseConfigFile } from "../src/lib/load-config";
import { buildReport, unknownIdentifiers } from "../src/lib/report";
import { scenarioFromQuery } from "../src/lib/scenario";

async function main() {
  const q: Record<string, string> = {};
  for (const arg of process.argv.slice(2)) for (const [k, v] of new URLSearchParams(arg.replace(/^\?/, ""))) q[k] = v;
  const result = await parseConfigFile(configPath());
  if (!result.ok) {
    console.error(`${result.path}\n${result.error}`);
    process.exit(1);
  }
  const lang = isLang(q.lang) ? q.lang : "en";
  const cfg = localize(result.config, lang);
  const unknown = unknownIdentifiers(cfg, cfg.models, q);
  if (unknown.length) {
    console.error(`Unknown identifiers: ${unknown.join(", ")}`);
    process.exit(2);
  }
  console.log(JSON.stringify(buildReport(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, q), lang), null, 2));
}

main();
