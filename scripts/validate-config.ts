import { readFileSync } from 'node:fs';
import { parseCatalog } from '../lib/parse-catalog';
const file = process.argv[2] || process.env.ESTIMATOR_CONFIG || 'config/catalog.json';
try {
  const data = parseCatalog(readFileSync(file, 'utf8'), file);
  console.log(`Configuration valide : ${data.models.length} modèles, ${data.gpus.length} GPU.`);
} catch (error) {
  console.error(`Configuration invalide (${file}): ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
}
