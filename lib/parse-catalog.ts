import { parseDocument } from 'yaml';
import { catalogSchema } from './schema';
/** Shared parser for server startup and CLI. Never fetches remote includes. */
export function parseCatalog(text: string, filename: string) {
  let raw: unknown;
  if (/\.ya?ml$/i.test(filename)) {
    const document = parseDocument(text, { uniqueKeys: true });
    if (document.errors.length) throw new Error(document.errors.map(e => e.message).join('; '));
    raw = document.toJS({ maxAliasCount: 50 });
  } else raw = JSON.parse(text);
  const result = catalogSchema.safeParse(raw);
  if (!result.success) throw new Error(result.error.issues.map(i => `${i.path.join('.') || 'racine'}: ${i.message}`).join('; '));
  return result.data;
}
