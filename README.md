# Estimateur KV cache

Application Next.js / TypeScript (page unique) qui estime la mémoire GPU nécessaire pour servir un LLM open-weight : poids du modèle, KV cache par requête et nombre de sessions simultanées.

## Démarrage

```bash
npm install
npm run dev            # http://localhost:3000
# ou en production
npm run build && npm start
```

La configuration (GPUs, modèles, formats) est lue côté serveur au démarrage depuis `config/estimator.yaml`. En dev, le fichier est relu à chaque requête.

```bash
KV_ESTIMATOR_CONFIG=/etc/kv/estimator.yaml npm start   # autre fichier (YAML ou JSON)
npm run check-config                                    # valider le fichier sans lancer l'app
npm run check-config -- autre-fichier.yaml
```

Si le fichier est invalide, la page affiche l'erreur précise (champ et chemin). Le schéma est strict : une clé inconnue ou mal orthographiée est rejetée.

## Entrées

| Paramètre | Contrôle |
|---|---|
| Modèle | champ de recherche textuel (nom, famille, `moe`, `vision`, `mla`, `hybride`…) |
| Contexte | slider 8k → 1M par doublement ; les crans au-delà du contexte natif sont marqués (orange : extension YaRN, rouge : hors limite) |
| Type de carte | V100, L40S, H100, H200, GH200 (96 / 144 Go), B200, GB200, B300, GB300 |
| Nombre de cartes | 1 à 8, plus le mode de parallélisme (TP, PP, DP-attention pour les MoE) |
| Format des poids | BF16, FP16, FP8, INT8, INT4, MXFP4, NVFP4 |
| Format du KV cache | BF16, FP16, FP8, NVFP4 |
| Réserve mémoire | % de chaque GPU laissé aux activations et graphes CUDA (≈ `--gpu-memory-utilization`) |

L'état est conservé dans l'URL : il suffit de copier le lien pour partager une configuration.

## Résultats

- poids du modèle (total et par GPU)
- KV par requête (vue logique, et mémoire réelle s'il y a réplication entre GPUs)
- modèle + 1 KV, et si cela tient
- KV total disponible en Gio et en nombre de sessions au contexte choisi
- répartition de la mémoire d'un GPU, détail par type de couche, sessions pour chaque taille de contexte

## Comment les limitations sont signalées

- **Option grisée et barrée (✕)** : format non supporté par le GPU (BF16 ou FP8 sur V100, KV NVFP4 hors Blackwell…). Impossible à sélectionner, la raison s'affiche au survol.
- **Bordure pointillée orange (⚠)** : supporté avec des limites (FP4 émulé en W4A16 sur Hopper/Ada, INT8 bridé sur Blackwell Ultra, TP invalide pour ce nombre de cartes, absence de NVLink…).
- Une ligne de résumé sous chaque groupe : « Sur V100 : ✕ BF16, FP8… · ⚠ INT8, INT4 limités ».
- **Ajustement automatique** : si on change de carte et que le format choisi n'existe plus, l'app bascule sur le meilleur format disponible et le signale (🔁).
- **Panneau « Limitations et avertissements »** classé par gravité (⛔ bloquant, ⚠️ attention, ℹ️ info) : poids qui ne tiennent pas, TP non divisible par le nombre de têtes, KV répliqué en TP, cache MLA dupliqué, pas de NVLink, dépassement d'un nœud, contexte au-delà du natif, absence de checkpoint officiel, paramètres de modèle peu fiables…

## Modèle de calcul

- **Poids** = paramètres × bits effectifs du format (échelles de quantification incluses). Les embeddings, le lm_head et l'encodeur vision restent en 16 bits quand on quantifie.
- **GQA/MHA** : couches × têtes KV × (dim K + dim V) × octets × tokens.
- **Fenêtre glissante** : idem, plafonné à la taille de la fenêtre (Gemma 3, MiMo-V2-Flash).
- **MLA** : couches × (rang latent + dim RoPE) × octets × tokens (Kimi K2, Mistral Large 3, GLM-5…), plus l'indexeur DSA s'il existe.
- **Linéaire / SSM** (Mamba-2, Gated DeltaNet, KDA) : état constant par requête, quel que soit le contexte.
- **TP** : têtes KV réparties sur les GPUs, et répliquées quand TP > nombre de têtes KV. Le latent MLA est copié sur chaque GPU.
- **PP** : couches réparties, pas de duplication.
- **DP-attention + EP** (MoE) : attention répliquée sur chaque GPU, experts répartis, chaque requête vit sur un seul GPU.
- **Sessions** = ⌊(mémoire GPU × (1 − réserve) − poids par GPU) ÷ KV par requête et par GPU⌋, toutes au contexte maximal. C'est le pire cas : le prefix caching et le partage de préfixes ne sont pas comptés.

## Fichier de configuration

Voir les commentaires dans `config/estimator.yaml`. En résumé :

```yaml
gpus:
  - id: h100
    name: H100 SXM
    memoryGiB: 79.6                     # mémoire vue par nvidia-smi
    interconnect: { label: "NVLink 4", nvlink: true }
    maxGpusPerNode: 8
    nvlinkDomain: 8
    weightFormats:                      # native | limited | unsupported
      fp8: native
      nvfp4: { level: limited, note: "W4A16 via Marlin" }
    kvCacheFormats: { bf16: native, fp8: native }

models:
  - id: qwen3-32b
    name: Qwen3-32B
    family: Qwen (Alibaba)
    paramsB: 32.8
    hiddenSize: 5120
    vocabSize: 151936
    attentionHeads: 64
    layers:
      - { kind: full, count: 64, kvHeads: 8, headDim: 128 }
    maxContext: 32768
    maxContextExtended: 131072
    nativeWeightFormat: bf16
    officialWeightFormats: [bf16, fp8, int4]
```

Types de couches : `full`, `sliding` (`window`), `mla` (`kvLoraRank`, `ropeHeadDim`), `linear` (`stateElements`).

### Fiabilité des données

Les paramètres d'architecture ont été saisis à la main (pas d'accès à Hugging Face au moment de la rédaction). Chaque modèle porte un champ `confidence` :

- `high` : architecture bien établie (Qwen3, Gemma 3, Mixtral, Llama/Nemotron, Kimi K2…)
- `medium` / `low` : modèles récents dont certains champs sont à confirmer (Mistral Large 3, Ministral 3, GLM-5, GLM-4.7-Flash, MiMo-V2-Flash, Qwen3.5, Kimi Linear, Nemotron 3 Nano…)

L'interface affiche un badge « à vérifier » sur ces modèles. Comparez avec le `config.json` du dépôt indiqué dans `source`, puis corrigez le YAML.

## Tests

```bash
npm test          # vitest : formules GQA, MLA, SWA, linéaire, réplication TP, limitations
npm run typecheck
```
