/**
 * Download a model's config.json (or Mistral params.json) and parameter count from the Hugging Face Hub,
 * as the { repo, revision, config, parameterMetadata } envelope that convertHfConfig reads.
 * Used by /api/hf-model and `pnpm run import-model --hf org/name`.
 */

export const HF_ORIGIN = "https://huggingface.co";

/** org/name, as on the Hub: no path traversal, no extra segments. */
const REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const REVISION_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$/;

export const isHfRepo = (repo: string) => REPO_RE.test(repo) && !repo.includes("..");

export interface HfEnvelope {
  repo: string;
  /** Commit SHA the files were read at. */
  revision: string;
  config: unknown;
  /** safetensors summary from the Hub API: { parameters: { BF16: n, … }, total: n }. */
  parameterMetadata?: unknown;
}

export class HfFetchError extends Error {
  constructor(
    message: string,
    /** HTTP status to return to the browser. */
    readonly status: number,
  ) {
    super(message);
  }
}

export async function fetchHfModel(
  repo: string,
  { revision, token, fetchImpl = fetch, timeoutMs = 15000 }: { revision?: string; token?: string; fetchImpl?: typeof fetch; timeoutMs?: number } = {},
): Promise<HfEnvelope> {
  if (!isHfRepo(repo)) throw new HfFetchError(`Invalid repository: ${repo} (expected org/name)`, 400);
  if (revision !== undefined && (!REVISION_RE.test(revision) || revision.includes(".."))) throw new HfFetchError(`Invalid revision: ${revision}`, 400);
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const get = async (url: string) => {
    let res: Response;
    try {
      res = await fetchImpl(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    } catch (e) {
      throw new HfFetchError(`Hugging Face unreachable (${(e as Error).message})`, 502);
    }
    if (res.status === 401 || res.status === 403)
      throw new HfFetchError(`${repo}: access denied (gated or private repository; set HF_TOKEN on the server)`, 403);
    return res;
  };

  const api = `${HF_ORIGIN}/api/models/${repo}${revision ? `/revision/${encodeURIComponent(revision)}` : ""}?expand[]=safetensors&expand[]=sha`;
  const info = await get(api);
  if (info.status === 404) throw new HfFetchError(`${repo}: repository not found`, 404);
  if (!info.ok) throw new HfFetchError(`${repo}: Hub API error ${info.status}`, 502);
  const meta = (await info.json()) as { sha?: string; safetensors?: unknown };
  const sha = meta.sha ?? revision ?? "main";

  for (const file of ["config.json", "params.json"]) {
    const res = await get(`${HF_ORIGIN}/${repo}/resolve/${encodeURIComponent(sha)}/${file}`);
    if (res.status === 404) continue;
    if (!res.ok) throw new HfFetchError(`${repo}/${file}: error ${res.status}`, 502);
    let config: unknown;
    try {
      config = await res.json();
    } catch {
      throw new HfFetchError(`${repo}/${file}: invalid JSON`, 502);
    }
    return { repo, revision: sha, config, parameterMetadata: meta.safetensors };
  }
  throw new HfFetchError(`${repo}: neither config.json nor params.json found`, 404);
}
