# Estimateur KV cache

Application Next.js / TypeScript (page unique) qui dimensionne le service d'un LLM open-weight sur GPU NVIDIA : poids du modèle, KV cache par requête, nombre de sessions simultanées, débit approximatif. Elle sert aussi au dimensionnement inverse et à la comparaison de scénarios. Moteur de référence : **vLLM** (et llm-d, qui orchestre des instances vLLM).

## Démarrage

```bash
npm install
npm run dev            # http://localhost:3000
# ou en production
npm run build && npm start
# ou en conteneur
docker build -t kv-cache-estimator .
docker run -p 3000:3000 -v $PWD/mon-catalogue.yaml:/config/estimator.yaml \
  -e KV_ESTIMATOR_CONFIG=/config/estimator.yaml kv-cache-estimator
```

La configuration (formats, moteurs, GPUs, modèles) est lue côté serveur au démarrage depuis `config/estimator.yaml`. En dev, elle est relue à chaque requête.

```bash
KV_ESTIMATOR_CONFIG=/etc/kv/estimator.yaml npm start   # autre fichier (YAML ou JSON)
npm run check-config                                    # valider sans lancer l'app
```

Si le fichier est invalide, la page affiche l'erreur précise. Le schéma est strict : une clé inconnue ou mal orthographiée est rejetée.

## Ajouter un modèle depuis son config.json

```bash
npm run import-model -- chemin/config.json                       # affiche l'entrée YAML
npm run import-model -- chemin/config.json --append              # l'ajoute au catalogue et le valide
npm run import-model -- config.json --source org/repo --params 235 --active 22 --id mon-modele
```

On peut faire la même chose dans l'interface, onglet **Importer un modèle** : coller ou charger le fichier, relire l'entrée YAML, la copier, ou tester le modèle tout de suite (il est gardé dans le navigateur).

Le convertisseur reconnaît les architectures suivantes :

- GQA / MHA
- fenêtre glissante via `layer_types` (Gemma 3/4, gpt-oss), `hybrid_layer_pattern` (MiMo) ou `sliding_window_pattern`
- attention par chunks (Llama 4)
- MLA (DeepSeek, Kimi, Mistral Large 3, GLM), y compris l'indexeur DSA avec IndexShare et pooling
- Mamba-2 (`hybrid_override_pattern`, Nemotron-H)
- Gated DeltaNet (Qwen3-Next, Qwen3.5)
- KDA (Kimi Linear, Kimi K3, GLM-5.3-Flash)
- modèles élagués par NAS (`block_configs`)
- le `params.json` de Mistral

Origine du nombre de paramètres, par ordre de priorité :

1. l'option `--params` ;
2. les métadonnées safetensors, si le fichier est l'enveloppe `{repo, revision, config, parameterMetadata}` de `config/upstream/` ;
3. une estimation à partir des dimensions (attention, MLP, experts, couches linéaires) ;
4. le nom du dépôt, si l'estimation n'est pas fiable (tenseurs empaquetés).

Les paramètres actifs des MoE sont estimés. Le format natif est déduit de `quantization_config`, et les états SSM passent en FP32 si le modèle l'impose (`mamba_ssm_cache_dtype`).

`config/upstream/` contient 24 `config.json` d'origine, avec leur révision. Les tests vérifient que le convertisseur retrouve exactement les couches du catalogue pour chacun.

## Entrées

| Paramètre | Contrôle |
|---|---|
| Modèle | recherche textuelle (nom, famille, `moe`, `vision`, `mla`, `hybride`…) |
| Contexte | 8k → 1M par doublement ; crans au-delà du natif marqués (orange : YaRN, rouge : hors limite) |
| Carte | V100, L40S, H100, H200, GH200 (96 / 144 Go), B200, GB200, B300, GB300 |
| Nombre de cartes | 1 à 8, puis 16, 32, 64 (et 72 en NVL72) |
| Parallélisme | TP / PP avec choix du degré de TP (PP = cartes ÷ TP), ou DP-attention + EP pour les MoE |
| Moteur | vLLM / llm-d, ZML (llmd) : un format doit être supporté par la carte **et** par le moteur |
| Formats | poids BF16, FP16, FP8, INT8, INT4, MXFP4, NVFP4 ; KV BF16, FP16, FP8, NVFP4 |
| Réserve | % de chaque GPU laissé aux activations et graphes CUDA (≈ `--gpu-memory-utilization`) |
| Options avancées | mémoire visible par GPU (valeur `nvidia-smi`), offload du KV en RAM CPU |

L'état complet est dans l'URL, il suffit de copier le lien pour partager une configuration.

## Résultats

- **Indicateurs principaux** : poids du modèle (total et par GPU), KV par requête (logique et réel s'il y a réplication), modèle + 1 KV, KV total disponible en Gio et en sessions.
- **Débit estimé** : une session seule, pleine charge, temps de prefill du contexte.
- Répartition de la mémoire d'un GPU, limitations et avertissements.
- **Détail** : KV par type de couche, sessions pour chaque contexte, commande `vllm serve` indicative.
- **Débit** : tok/s par session et agrégés selon le nombre de sessions actives, limite (mémoire ou calcul), offload CPU.
- **Dimensionnement** : pour un objectif de N sessions, nombre minimal de cartes par type de GPU et meilleur parallélisme ; contexte maximal pour N sessions sur la configuration actuelle.
- **Comparaison** : jusqu'à 6 scénarios épinglés côte à côte.
- **Export JSON** du scénario et des résultats.

## Signalement des limitations

- **Option grisée et barrée (✕)** : non supportée par la carte ou le moteur (BF16 / FP8 sur V100, KV NVFP4 hors Blackwell, DP-attention sur un modèle dense, TP qui ne divise pas les têtes…). Raison au survol.
- **Bordure pointillée orange (⚠)** : possible mais limité (FP4 émulé sur Hopper/Ada, INT8 bridé sur Blackwell Ultra, KV NVFP4 expérimental dans vLLM, TP sans NVLink, multi-nœuds, profil moteur non vérifié).
- **Ligne de résumé** sous chaque groupe de formats.
- **Ajustement automatique (🔁)** : quand on change de carte, de moteur ou de modèle et que le choix n'est plus possible, l'app bascule sur le meilleur choix et le signale. Elle revient au format natif dès qu'il redevient disponible.
- **Panneau d'avertissements** classé par gravité (⛔ / ⚠️ / ℹ️).

## Modèle de calcul

- **Poids** = paramètres × bits effectifs (échelles incluses : INT4 g128 = 4,25 bits, NVFP4 = 4,5 bits…). Embeddings, lm_head et encodeur vision restent en 16 bits si on quantifie. `nativeBitsPerParam` corrige les checkpoints natifs particuliers (Kimi K2.5 en INT4 g32 = 4,5 bits).
- **GQA/MHA** : couches × têtes KV × (dim K + dim V) × octets × tokens.
- **Fenêtre glissante** : plafonnée à la fenêtre si le moteur le gère (`slidingWindowOptimized`), sinon contexte complet.
- **MLA** : couches × (rang latent + dim RoPE) × octets × tokens ; indexeur DSA en FP8 (+ échelle) en plus.
- **Linéaire / SSM** : état SSM (`ssmStateBytes`, FP32 si imposé) + état de convolution, constant par requête.
- **TP** : têtes KV réparties, répliquées si TP > têtes KV ; latent MLA copié sur chaque rang TP. **PP** : couches réparties, sans duplication. **DP-attention + EP** : attention répliquée sur chaque GPU, experts répartis, chaque requête sur un seul GPU.
- **Sessions** = ⌊(mémoire GPU × (1 − réserve) − poids par GPU) ÷ KV par requête et par GPU⌋, au contexte maximal (pire cas). Prefix caching ignoré.
- **Offload CPU** : sessions supplémentaires gardées en RAM (en attente, préemptées, prefix cache). Elles n'augmentent pas la concurrence de décodage. Temps de rechargement = KV par GPU ÷ lien CPU↔GPU.
- **Débit** : temps d'une étape = max(octets lus ÷ (bande passante × efficacité), FLOPs ÷ (TFLOPS × MFU)) + latence des collectives. Pour un MoE, les octets lus incluent les experts touchés par le batch, plus le KV des sessions actives. En PP, l'étape traverse les étages. Le prefill est calculé en FLOPs (attention causale incluse). Réglages dans `performance:` du YAML.

## Fichier de configuration

Sections : `defaults`, `performance`, `largeGpuCounts`, `weightFormats`, `kvCacheFormats`, `engines`, `gpus`, `models`. Voir les commentaires de `config/estimator.yaml`.

```yaml
engines:
  - id: vllm
    name: vLLM / llm-d
    features: { slidingWindowOptimized: true, dpAttention: true, kvOffload: true }
    weightFormats: { bf16: native, fp8: native, nvfp4: native }            # absent = non supporté
    kvCacheFormats: { bf16: native, fp8: native, nvfp4: { level: limited, note: "expérimental" } }
    gpus: { v100: { level: limited, note: "support non garanti" } }

gpus:
  - id: h100
    name: H100 SXM
    memoryGiB: 79.6                     # mémoire vue par nvidia-smi
    bandwidthTBs: 3.35
    tflops: { bf16: 989, fp8: 1979 }    # dense
    interconnect: { label: "NVLink 4", nvlink: true, bandwidthGBs: 450 }
    maxGpusPerNode: 8
    nvlinkDomain: 8
    hostLinkGBs: 55
    hostLinkLabel: PCIe Gen5 x16
    weightFormats: { fp8: native, nvfp4: { level: limited, note: "W4A16 via Marlin" } }
    kvCacheFormats: { bf16: native, fp8: native }

models:
  - id: qwen3-32b
    name: Qwen3-32B
    family: Qwen (Alibaba)
    source: Qwen/Qwen3-32B
    revision: 9216db5781bf21249d130ec9da846c4624c16137
    paramsB: 32.76
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

Types de couches : `full`, `sliding` (`window`), `mla` (`kvLoraRank`, `ropeHeadDim`, `extraBytesPerToken`), `linear` (`ssmElements`, `convElements`).

### Catalogue et fiabilité

64 modèles : Mistral, Z.ai (GLM), Moonshot (Kimi), Xiaomi (MiMo), Qwen, NVIDIA, Google (Gemma), DeepSeek, OpenAI (gpt-oss), Meta (Llama).

- Les modèles marqués **« config vérifiée »** (champ `revision`) ont été vérifiés contre leur `config.json` d'origine. Les copies sont dans `config/upstream/`, reprises de la branche `feat/kv-cache-estimator`.
- Les autres viennent de connaissances publiques. Le champ `confidence` (`high`, `medium`, `low`) l'indique, et l'interface affiche un badge pour les deux derniers niveaux.
- Les paramètres actifs de certains modèles récents (Kimi K3, GLM-5.3, MiMo-V2.6-Pro) sont estimés.
- Le profil **ZML (llmd)** est indicatif et doit être complété.

## Tests

```bash
npm test          # vitest : formules, réplication TP/PP/DP, offload, débit, dimensionnement,
                  # convertisseur sur les 24 config.json, balayage de toutes les combinaisons
npm run typecheck
```

La CI GitHub (`.github/workflows/check.yml`) lance la validation du catalogue, le typecheck, les tests et le build.
