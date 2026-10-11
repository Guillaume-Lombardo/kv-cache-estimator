import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Le fichier de configuration est lu côté serveur au démarrage (voir src/lib/load-config.ts).
  output: "standalone",
};

export default nextConfig;
