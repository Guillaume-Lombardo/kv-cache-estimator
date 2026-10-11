import { readFile } from "node:fs/promises";
import path from "node:path";
import { parse as parseYaml } from "yaml";
import { z } from "zod";
import { configSchema, type ConfigSource } from "./config-schema";

export type ConfigResult =
  | { ok: true; config: ConfigSource; path: string }
  | { ok: false; error: string; path: string };

/** Configuration path: KV_ESTIMATOR_CONFIG, falling back to config/estimator.yaml. */
export function configPath(): string {
  const p = process.env.KV_ESTIMATOR_CONFIG ?? "config/estimator.yaml";
  return path.isAbsolute(p) ? p : path.join(/*turbopackIgnore: true*/ process.cwd(), p);
}

export async function parseConfigFile(file: string): Promise<ConfigResult> {
  let raw: unknown;
  try {
    const text = await readFile(file, "utf8");
    raw = file.endsWith(".json") ? JSON.parse(text) : parseYaml(text);
  } catch (e) {
    return { ok: false, path: file, error: `Lecture impossible : ${(e as Error).message}` };
  }
  const parsed = configSchema.safeParse(raw);
  if (!parsed.success) return { ok: false, path: file, error: z.prettifyError(parsed.error) };
  return { ok: true, path: file, config: parsed.data };
}

let cached: Promise<ConfigResult> | undefined;

/**
 * Read once at startup in production; read on every request in development
 * so the file can be edited without restarting the server.
 */
export function loadConfig(): Promise<ConfigResult> {
  if (process.env.NODE_ENV !== "production") return parseConfigFile(configPath());
  cached ??= parseConfigFile(configPath());
  return cached;
}
