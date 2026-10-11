import 'server-only';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { catalogSchema, type Catalog } from './schema';
let loaded: Promise<Catalog> | undefined;
/** Load once per process. Restart the server to use a modified catalogue. */
export function loadCatalog(): Promise<Catalog> {
  return loaded ??= (async () => {
    const file = process.env.ESTIMATOR_CONFIG || path.join(process.cwd(), 'config/catalog.json');
    const parsed = catalogSchema.safeParse(JSON.parse(await readFile(file, 'utf8')));
    if (!parsed.success) throw new Error(`Configuration invalide (${file}): ${parsed.error.issues.map(i => `${i.path.join('.')}: ${i.message}`).join('; ')}`);
    return parsed.data;
  })();
}
