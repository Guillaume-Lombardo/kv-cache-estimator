import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'KV Lab — Estimateur mémoire LLM', description: 'Dimensionnement des poids et des caches LLM sur GPU. Catalogue configurable, architectures GQA, MLA et hybrides.' };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="fr"><body>{children}</body></html>;
}
