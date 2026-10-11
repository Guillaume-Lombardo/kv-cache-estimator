// Validate the configuration file without starting the application.
// Usage: pnpm run check-config [path/to/file.yaml]
import { configPath, parseConfigFile } from "../src/lib/load-config";

const file = process.argv[2] ?? configPath();
parseConfigFile(file).then((r) => {
  if (!r.ok) {
    console.error(`✗ ${r.path}\n${r.error}`);
    process.exit(1);
  }
  console.log(`✓ ${r.path} : ${r.config.gpus.length} GPUs, ${r.config.models.length} modèles`);
});
