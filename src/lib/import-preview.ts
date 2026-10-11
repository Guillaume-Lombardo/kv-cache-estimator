import { convertHfConfig, type ImportOptions, type ImportResult } from "./hf-import";
import { LANGS, tr, type Lang } from "./i18n";

type Preview = { result: ImportResult | null; error: string | null };
export type ImportPreview = Record<Lang, Preview>;

/** Capture both translations of one submitted import, independent of later form edits. */
export function buildImportPreview(
  raw: string,
  options: Omit<ImportOptions, "lang">,
  existingIds: ReadonlySet<string>,
): ImportPreview {
  return Object.fromEntries(LANGS.map((lang) => {
    let json: unknown;
    try {
      // The converter normalizes its input, so give each language its own parsed object.
      json = JSON.parse(raw);
    } catch {
      return [lang, { result: null, error: tr(lang)("JSON invalide : vérifiez la syntaxe du fichier.", "Invalid JSON: check the file syntax.") }];
    }
    try {
      const result = convertHfConfig(json, { ...options, lang });
      if (existingIds.has(result.model.id)) result.model.id = `${result.model.id}-import`;
      return [lang, { result, error: null }];
    } catch (e) {
      return [lang, { result: null, error: (e as Error).message }];
    }
  })) as ImportPreview;
}
