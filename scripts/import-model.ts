// Convertit un config.json Hugging Face (ou params.json Mistral) en entrée YAML pour config/estimator.yaml.
//
// Usage :
//   npm run import-model -- chemin/config.json [--params 235] [--active 22] [--id mon-modele]
//                           [--name "Mon modèle"] [--family "Qwen (Alibaba)"] [--source org/repo] [--append]
//
// --append ajoute l'entrée à la fin du fichier de configuration (section models), puis le valide.
import { readFile, appendFile } from "node:fs/promises";
import { convertHfConfig, modelToYaml, type ImportOptions } from "../src/lib/hf-import";
import { configPath, parseConfigFile } from "../src/lib/load-config";

const args = process.argv.slice(2);
const file = args.find((a) => !a.startsWith("--") && !isValue(a));
function isValue(a: string) {
  const i = args.indexOf(a);
  return i > 0 && args[i - 1].startsWith("--") && !["--append"].includes(args[i - 1]);
}
const opt = (k: string) => {
  const i = args.indexOf(`--${k}`);
  return i >= 0 ? args[i + 1] : undefined;
};

async function main() {
  if (!file) {
    console.error("Usage : npm run import-model -- chemin/config.json [--params N] [--active N] [--id …] [--append]");
    process.exit(2);
  }
  const json = JSON.parse(await readFile(file, "utf8"));
  const options: ImportOptions = {
    id: opt("id"),
    name: opt("name"),
    family: opt("family"),
    source: opt("source"),
    revision: opt("revision"),
    paramsB: opt("params") ? Number(opt("params")) : undefined,
    activeParamsB: opt("active") ? Number(opt("active")) : undefined,
  };
  const { model, warnings, paramsSource } = convertHfConfig(json, options);
  const yaml = modelToYaml(model);

  console.error(`# ${model.name} — paramètres : ${model.paramsB} B (${paramsSource})`);
  for (const w of warnings) console.error(`# ⚠ ${w}`);

  if (args.includes("--append")) {
    const target = configPath();
    const current = await parseConfigFile(target);
    if (current.ok && current.config.models.some((m) => m.id === model.id)) {
      console.error(`✗ id déjà présent dans ${target} : ${model.id} (utilisez --id)`);
      process.exit(1);
    }
    await appendFile(target, `\n${yaml}`);
    const after = await parseConfigFile(target);
    if (!after.ok) {
      console.error(`✗ Le fichier n'est plus valide après ajout :\n${after.error}`);
      process.exit(1);
    }
    console.error(`✓ Ajouté à ${target}`);
  } else {
    process.stdout.write(yaml + "\n");
  }
}

main().catch((e) => {
  console.error(`✗ ${(e as Error).message}`);
  process.exit(1);
});
