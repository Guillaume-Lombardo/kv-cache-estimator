import type { NextConfig } from "next";

/**
 * URL prefix the application is served under, e.g. `/kv-estimator` behind a reverse proxy.
 * Empty (the default) serves it at the root. Next.js applies it at build time: pages, `/_next`
 * assets and redirects all carry the prefix, so changing it requires a new build.
 */
function basePathFromEnv(): string {
  const value = process.env.KV_ESTIMATOR_BASE_PATH ?? "";
  if (value === "") return "";
  const segments = value.split("/").slice(1);
  const valid =
    value.startsWith("/") &&
    segments.every((s) => /^[A-Za-z0-9._~-]+$/.test(s) && s !== "." && s !== "..");
  if (!valid) {
    throw new Error(
      `Invalid KV_ESTIMATOR_BASE_PATH ${JSON.stringify(value)}: use an empty value to serve at the root, ` +
        `or a path that starts with "/" and has no trailing slash, such as "/kv-estimator".`,
    );
  }
  return value;
}

const nextConfig: NextConfig = {
  // Read the configuration on the server at startup (see src/lib/load-config.ts).
  output: "standalone",
  basePath: basePathFromEnv(),
};

export default nextConfig;
