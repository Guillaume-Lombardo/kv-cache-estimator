import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { buildImportPreview } from "./import-preview";

const raw = readFileSync(path.join(__dirname, "../../config/upstream/Qwen__Qwen3-8B.json"), "utf8");

describe("submitted import preview", () => {
  it("switches warning language without changing the imported model", () => {
    const preview = buildImportPreview(raw, {}, new Set(["qwen3-8b"]));
    expect(preview.fr.result!.warnings.join(" ")).toContain("Pas d'extension RoPE");
    expect(preview.en.result!.warnings.join(" ")).toContain("No RoPE extension");
    expect(preview.en.result!.model).toEqual(preview.fr.result!.model);
    expect(preview.en.result!.model.id).toBe("qwen3-8b-import");
  });

  it("keeps submitted options when form values are edited before a language switch", () => {
    const options = { paramsB: 9, source: "example/submitted-model" };
    const ids = new Set<string>();
    const preview = buildImportPreview(raw, options, ids);
    options.paramsB = 100;
    options.source = "example/pending-edit";
    ids.add(preview.fr.result!.model.id);
    expect(preview.en.result!.model.paramsB).toBe(9);
    expect(preview.en.result!.model.source).toBe("example/submitted-model");
    expect(preview.en.result!.model.id).toBe(preview.fr.result!.model.id);
  });

  it("switches conversion errors and JSON syntax errors to the selected language", () => {
    const invalidModel = buildImportPreview("{}", {}, new Set());
    expect(invalidModel.fr.error).toContain("introuvables");
    expect(invalidModel.en.error).toContain("not found");
    const invalidJson = buildImportPreview("{", {}, new Set());
    expect(invalidJson.fr.error).toContain("JSON invalide");
    expect(invalidJson.en.error).toContain("Invalid JSON");
    expect(invalidModel.en.result).toBeNull();
    expect(invalidJson.en.result).toBeNull();
  });
});
