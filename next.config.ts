import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Read the configuration on the server at startup (see src/lib/load-config.ts).
  output: "standalone",
};

export default nextConfig;
