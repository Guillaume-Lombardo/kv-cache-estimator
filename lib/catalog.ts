import 'server-only';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import type { Catalog } from './schema';
import { parseCatalog } from './parse-catalog';
let loaded: Promise<Catalog> | undefined;
/** Load once per process. Restart the server to use a modified catalogue. */
export function loadCatalog(): Promise<Catalog> {
  return loaded ??= (async () => {
    const file = process.env.ESTIMATOR_CONFIG || path.join(process.cwd(), 'config/catalog.json');
    return parseCatalog(await readFile(file, 'utf8'), file);
  })();
}
