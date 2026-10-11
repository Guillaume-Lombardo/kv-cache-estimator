import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Estimateur KV cache",
  description: "Mémoire des poids et du KV cache des LLM open-weight sur GPU NVIDIA",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
