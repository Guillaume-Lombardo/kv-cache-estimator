import { fetchHfModel, HfFetchError } from "@/lib/hf-fetch";

export const dynamic = "force-dynamic";

/**
 * GET /api/hf-model?repo=org/name[&revision=…] — config.json and parameter count from the Hugging Face Hub,
 * fetched by the server (HF_TOKEN, if set, opens gated repositories). Only huggingface.co is contacted.
 */
export async function GET(request: Request) {
  const params = new URL(request.url).searchParams;
  const repo = params.get("repo") ?? "";
  try {
    const envelope = await fetchHfModel(repo, { revision: params.get("revision") || undefined, token: process.env.HF_TOKEN || undefined });
    return Response.json(envelope);
  } catch (e) {
    const status = e instanceof HfFetchError ? e.status : 500;
    return Response.json({ error: (e as Error).message }, { status });
  }
}
