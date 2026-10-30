import { isLang, langFromAcceptLanguage, localize } from "@/lib/i18n";
import { loadConfig } from "@/lib/load-config";
import { buildReport, unknownIdentifiers } from "@/lib/report";
import { scenarioFromQuery } from "@/lib/scenario";

export const dynamic = "force-dynamic";

/**
 * GET /api/estimate?model=…&gpu=…&n=…&ctx=… — same parameters as the page URL, plus lang=fr|en.
 * Returns the JSON report of the scenario (see src/lib/report.ts).
 */
export async function GET(request: Request) {
  const result = await loadConfig();
  if (!result.ok) return Response.json({ error: "invalid configuration", detail: result.error }, { status: 500 });
  const q = Object.fromEntries(new URL(request.url).searchParams);
  const lang = isLang(q.lang) ? q.lang : langFromAcceptLanguage(request.headers.get("accept-language"));
  const cfg = localize(result.config, lang);
  const unknown = unknownIdentifiers(cfg, cfg.models, q);
  if (unknown.length) return Response.json({ error: "unknown identifiers", unknown }, { status: 400 });
  return Response.json(buildReport(cfg, cfg.models, scenarioFromQuery(cfg, cfg.models, q), lang));
}
