# Estimateur KV cache

Application Next.js / TypeScript (page unique) qui dimensionne le service d'un LLM open-weight sur GPU NVIDIA : poids du modèle, KV cache par requête, nombre de sessions simultanées, débit approximatif. Elle sert aussi au dimensionnement inverse et à la comparaison de scénarios. Moteur de référence : **vLLM** (et llm-d, qui orchestre des instances vLLM).

## Démarrage

Prérequis : Node.js 22.13 ou supérieur et Corepack. La version de pnpm est fixée dans `package.json`.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm run dev            # http://localhost:3000
# ou en production
pnpm run build && pnpm start
# ou en conteneur
docker build -t kv-cache-estimator .
docker run -p 3000:3000 -v $PWD/mon-catalogue.yaml:/config/estimator.yaml \
  -e KV_ESTIMATOR_CONFIG=/config/estimator.yaml kv-cache-estimator
```

La configuration (formats, moteurs, GPUs, modèles) est lue côté serveur au démarrage depuis `config/estimator.yaml`. En dev, elle est relue à chaque requête.

```bash
KV_ESTIMATOR_CONFIG=/etc/kv/estimator.yaml pnpm start   # autre fichier (YAML ou JSON)
pnpm run check-config                                    # valider sans lancer l'app
```

Si le fichier est invalide, la page affiche l'erreur précise. Le schéma est strict : une clé inconnue ou mal orthographiée est rejetée.

## Ajouter un modèle depuis son config.json

```bash
pnpm run import-model chemin/config.json                       # affiche l'entrée YAML
pnpm run import-model chemin/config.json --append              # l'ajoute au catalogue et le valide
pnpm run import-model config.json --source org/repo --params 235 --active 22 --id mon-modele
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
| Moteur | vLLM / llm-d, ZML / LLMD : un format doit être supporté par la carte **et** par le moteur |
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

## Profils moteur

Chaque moteur déclare, avec une note et une source pour chaque point :

- **Formats** : niveau de support pour chaque format de poids et de KV (natif, limité, **non vérifié**, non supporté). Un format « non vérifié » reste sélectionnable mais il est signalé.
- **Format en mémoire** : `memoryFormat` indique si le moteur garde le format du fichier ou s'il le déquantifie au chargement (ex. fichier FP8 → BF16 en VRAM). La VRAM est calculée sur le format en mémoire. Si ce point n'est pas vérifié, le calcul garde le format du fichier et l'avertissement donne aussi la taille en BF16.
- **Capacités à trois états** (oui / non / non vérifié) : KV paginé, pool KV préalloué (les deux peuvent coexister), prefix caching, continuous batching, attention glissante, libération des blocs hors fenêtre, TP, PP, DP-attention, offload.
- **Fenêtres glissantes** : l'économie de VRAM n'est appliquée que si le moteur supporte l'attention glissante **et** libère effectivement les blocs hors fenêtre. Sans support de l'attention glissante, c'est une erreur (résultats faux). Si la libération n'est pas confirmée, le contexte complet est compté.

**ZML / LLMD** (et non llm-d) :

- **Capacités serveur**, d'après la [documentation LLMD](https://zml.ai/llmd/) : PagedAttention, continuous batching, tensor parallel, prefix caching.
- **Formats, précision du KV et fenêtres glissantes**, d'après le code de la [bibliothèque ZML @3500e88](https://github.com/zml/zml/tree/3500e88cf84e0e98d37b0f6048e6af9d33bc6108) (9 octobre 2026), backend CUDA, car le code de LLMD n'est pas public :
  - poids FP8 / MXFP4 / NVFP4 : présents et gardés quantifiés en mémoire, mais leur chargement par LLMD n'est pas vérifié ;
  - INT4 / INT8 : non implémentés (TODO dans `zml/nn.zig`) ;
  - KV FP8 : refusé par FA2/FA3, désactivé dans le wrapper Triton ;
  - fenêtre glissante : appliquée par les noyaux, mais aucune libération de pages dans la bibliothèque. Aucune économie de VRAM n'est donc appliquée.
- Le profil indique ce qui reste à vérifier (pool préalloué, PP, DP-attention, offload).

## Modèle de calcul

- **Poids** = paramètres × bits effectifs (échelles incluses : INT4 g128 = 4,25 bits, NVFP4 = 4,5 bits…). Embeddings, lm_head et encodeur vision restent en 16 bits si on quantifie. `nativeBitsPerParam` corrige les checkpoints natifs particuliers (Kimi K2.5 en INT4 g32 = 4,5 bits).
- **GQA/MHA** : couches × têtes KV × (dim K + dim V) × octets × tokens.
- **Fenêtre glissante** : plafonnée à la fenêtre si le moteur le gère (`slidingWindowOptimized`), sinon contexte complet.
- **MLA** : couches × (rang latent + dim RoPE) × octets × tokens ; indexeur DSA en FP8 (+ échelle) en plus.
- **Linéaire / SSM** : état SSM (`ssmStateBytes`, FP32 si imposé) + état de convolution, constant par requête.
- **TP** : têtes KV réparties, répliquées si TP > têtes KV ; latent MLA copié sur chaque rang TP.
- **PP** : couches entières par étage. On dimensionne le rang le plus chargé : ⌈couches ÷ PP⌉ couches, en prenant les plus coûteuses en KV puisque l'ordre exact n'est pas connu. Ce rang porte aussi les embeddings d'entrée ou de sortie (copiés sur les deux bords s'ils sont liés). PP supérieur au nombre de couches = erreur.
- **DP-attention + EP** : seuls les experts routés (`expertParamsB`) sont répartis. Attention, embeddings, couches denses et experts partagés sont présents sur chaque GPU, à leur précision réelle (BF16 pour les tensors non quantifiés). Sans `expertParamsB`, la capacité n'est pas validée (erreur). Chaque requête vit sur un seul GPU.
- **Sessions** = ⌊(mémoire GPU × (1 − réserve) − poids du GPU le plus chargé) ÷ KV par requête de ce GPU⌋, au contexte maximal (pire cas). Prefix caching ignoré.
- **KV total disponible** : mémoire libre physique totale (N × mémoire utilisable − poids physiques, réplications comprises). Le budget prudent (mémoire libre du GPU le plus chargé × N), qui sert au calcul des sessions, est affiché à côté quand il diffère (PP inégal).
- **Configuration invalide** (au moins une erreur bloquante) : les indicateurs gardent les chiffres, mais ils sont présentés comme une « capacité mémoire théorique », en rouge et sans statut de réussite.
- **Modèle + 1 KV** : occupation physique réelle (réplications du KV en TP, poids répliqués en DP-attention ou en PP). La somme logique est affichée à côté si elle diffère. Le verdict « tient » porte sur le GPU le plus chargé.
- **Recommandations** (dimensionnement, contexte maximal, meilleure colonne de la comparaison) : seules les configurations sans erreur bloquante sont retenues (formats, parallélisme, contexte du modèle, capacité). La mémoire visible personnalisée ne s'applique qu'à la carte pour laquelle elle a été saisie.
- **Commande indicative** : elle utilise le dépôt du format natif ou celui déclaré dans `checkpoints`. Sinon, le checkpoint reste à fournir et la commande le signale.
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

## Tests

```bash
pnpm test          # vitest : formules, réplication TP/PP/DP, offload, débit, dimensionnement,
                  # convertisseur sur les 24 config.json, balayage de toutes les combinaisons
pnpm run typecheck
```

La CI GitHub (`.github/workflows/check.yml`) lance la validation du catalogue, le typecheck, les tests et le build.
