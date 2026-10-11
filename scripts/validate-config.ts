import { readFileSync } from 'node:fs';
import { catalogSchema } from '../lib/schema';
const file = process.argv[2] || process.env.ESTIMATOR_CONFIG || 'config/catalog.json';
const result = catalogSchema.safeParse(JSON.parse(readFileSync(file, 'utf8')));
if (!result.success) { console.error(result.error.issues); process.exit(1); }
console.log(`Configuration valide : ${result.data.models.length} modèles, ${result.data.gpus.length} GPU.`);
