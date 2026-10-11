# KV Lab

Application single page Next.js / TypeScript pour estimer la mémoire d’inférence d’un LLM sur 1 à 8 GPU identiques. L’interface est en français. Les calculs se font dans le navigateur ; aucun poids de modèle n’est téléchargé et aucune API d’inférence n’est appelée.

## Lancer

Prérequis : Node.js 22 ou supérieur et npm.

```bash
npm ci
npm run dev
```

Ouvrir http://localhost:3000.

Pour une exécution en production :

```bash
npm run build
npm start
```

## Charger son catalogue avant le lancement

Le fichier par défaut est `config/catalog.json`. Pour le remplacer, définir `ESTIMATOR_CONFIG` avant de démarrer le serveur. Le fichier est lu et validé au démarrage des workers Node, puis conservé en mémoire. Redémarrer après modification. Aucune configuration n’est envoyée par le navigateur et aucune URL distante n’est chargée comme configuration.

Linux / macOS :

```bash
export ESTIMATOR_CONFIG=/chemin/absolu/catalog.json
npm run validate:config
npm run dev
```

PowerShell :

```powershell
$env:ESTIMATOR_CONFIG = 'C:\configuration\catalog.json'
npm run validate:config
npm run dev
```

On peut aussi vérifier un fichier sans le sélectionner pour l’application :

```bash
npm run validate:config -- /chemin/catalog.json
```

`ESTIMATOR_CONFIG` est une variable serveur, pas une variable `NEXT_PUBLIC_*`. Elle peut être changée entre le build et le lancement. Le catalogue est public pour les visiteurs de l’application : ne pas y mettre de secrets.

## Ce que l’application affiche

- Recherche textuelle et sélection du modèle.
- Contexte : 8k, 16k, 32k, 64k, 128k, 256k, 512k et 1M tokens. Les multiples sont binaires : 8k = 8 192 et 1M = 1 048 576.
- GPU, nombre de GPU, interconnexion, format des poids, format du KV cache et réserve par GPU.
- Poids bruts et estimés, cache logique et cache physique par session, modèle + une session, mémoire restante pour les caches et nombre de sessions complètes.
- Décomposition mémoire, comparaison des huit contextes et export JSON d’un scénario.
- Options incompatibles grisées, formats conditionnels marqués `*`, avertissements détaillés et sources consultables.

Le curseur permet de dépasser le contexte publié pour voir son coût mémoire. Dans ce cas, la configuration n’est pas validée et le nombre de sessions utilisables est zéro. La limite n’est jamais étendue silencieusement avec RoPE ou YaRN.

## Catalogue initial

Photographie vérifiée le **11 octobre 2026**. La sélection est représentative, pas exhaustive. Elle contient 24 modèles calculables et une annonce indisponible.

| Famille | Modèles |
| --- | --- |
| Mistral | Ministral 3 8B, Mistral Small 3.2 24B, Devstral Small 2 24B, Medium 3.5 128B, Large 3 675B |
| Z.ai | GLM 4.7 Flash, GLM 4.7, GLM 5.3, GLM 5.3 Flash |
| Moonshot | Kimi K2, Kimi K2.5, Kimi K3 |
| Xiaomi | MiMo 7B RL, MiMo V2.6 Flash RL, MiMo V2.6 Pro RL |
| Qwen | Qwen3 8B, Qwen3 32B, Qwen3 235B A22B, Qwen3.5 35B A3B |
| NVIDIA | Llama 3.3 Nemotron Super 49B, Nemotron 3 Nano 30B A3B, Nemotron 3 Super 120B A12B |
| Google | Gemma 4 31B, Gemma 4 26B A4B |

Mistral Large 4 figure comme indisponible : les poids sont annoncés pour le 31 octobre 2026, et le calcul ne suppose pas une architecture KV inconnue. Les références de configuration consultées sont conservées dans `config/upstream/`. Les fiches du catalogue contiennent les liens officiels et les révisions utilisées. Les tailles de paramètres sont estimées à partir des métadonnées flottantes ou d’une valeur nominale explicitement indiquée ; les comptes de tensors empaquetés ne sont pas assimilés à des nombres de paramètres logiques.

Les GPU couvrent V100, L40S, H100, H200, GH200, B200, GB200, B300 et GB300. Les variantes PCIe/SXM et 96/144 Go de GH200 sont distinctes. Le compteur désigne les GPU physiques : un superchip GB200/GB300 contient deux GPU. La RAM du CPU Grace est exclue.

Les capacités nominales exprimées en Go sont converties en Gio. Ce choix conservateur peut différer de la mémoire réellement exposée par le pilote. Le champ avancé permet de la remplacer par la valeur de `nvidia-smi` en MiB divisée par 1024. B300 et GB300 disposent d’un avertissement spécifique : certaines fiches indiquent 270/279 Go utilisables plutôt que 288 Go nominaux. Ne pas confondre une réserve système déjà retranchée de la capacité utilisable avec la réserve d’exécution de cette application.

## Méthode de calcul

Toutes les tailles sont affichées en Gio, soit 2³⁰ octets.

```text
Poids bruts = paramètres totaux × octets par paramètre
Poids estimés = poids bruts × (1 + surcoût poids)
Budget caches = max(0, GPU × VRAM/GPU − GPU × réserve/GPU − poids estimés)
Sessions = floor(budget caches / cache physique d’une session)
```

Les poids d’un MoE comprennent tous ses experts. Les paramètres actifs ne suffisent pas à dimensionner la mémoire.

### GQA / MHA / MQA

```text
Cache logique = Σ couches × tokens retenus × têtes KV × (dimension K + dimension V) × octets KV
```

Pour une fenêtre glissante optimisée, `tokens retenus = min(contexte, fenêtre)`. Sinon, chaque couche réserve le contexte complet. Les couches globales et locales peuvent avoir des nombres de têtes et des dimensions différents, comme Gemma 4 et MiMo V2.6.

Le partage de projections K/V de Gemma 4 n’est pas assimilé à un cache unique : K et V subissent des transformations différentes.

### MLA et architectures hybrides

```text
MLA compressé = couches × contexte × (rang latent + dimension RoPE) × octets KV
Cache hybride = cache attention + états récurrents fixes + index auxiliaires
```

Le MLA suppose un backend conservant les latents compressés. Un backend matérialisant des K/V étendus utilisera davantage de mémoire.

Les états Mamba, KDA et DeltaNet sont estimés en FP32 avec convolutions FP16. Ils ne changent pas avec le format KV. Les index DSA utilisent une borne conservatrice en FP16, sans gains de pooling ou de partage IndexShare. Ces modèles sont marqués expérimentaux ; le catalogue explique leurs hypothèses. Les allocations de préfixes récurrents, snapshots intermédiaires et caches MTP ne sont pas modélisées.

### Multi-GPU

Deux modes sont proposés :

| Mode | Intérêt | Limite |
| --- | --- | --- |
| Tensor parallel | Vérifie la divisibilité des têtes de requête et KV ; compte les réplications KV | Suppose les poids uniformément répartis, sans vérification du placement de chaque couche/expert |
| Partition idéale | Donne une borne optimiste avec partage parfait de la mémoire | Ne valide aucun plan TP, PP ou EP |

En TP, les têtes KV sont répliquées quand il y a davantage de GPU que de têtes. Le latent MLA et les index sont supposés répliqués sur les rangs TP. Les états récurrents sont intégralement répliqués par prudence, même si certains moteurs les répartissent. Les poids sont répartis sur les cartes et ne sont pas multipliés par leur nombre.

NVLink n’est pas une condition nécessaire au multi-GPU. L40S reste utilisable via PCIe. La disponibilité théorique de NVLink sur une puce ne prouve pas que le serveur possède les ponts ou NVSwitch nécessaires. GH200 C2C relie le CPU au GPU et ne garantit pas une liaison GPU-GPU.

### Quantification et réserve

Les formats sont des projections de mémoire, pas une liste de checkpoints disponibles. INT4 ne désigne pas un format binaire unique. Les méthodes AWQ, GPTQ, GGUF, bitsandbytes, Marlin et les backends d’attention n’ont pas les mêmes contraintes. Les profils utilisent donc trois états : natif, conditionnel, exclu.

NVFP4 sur Hopper/Ada est conditionnel, car une déquantification spécialisée peut exister même sans Tensor Cores FP4 natifs. Le profil Volta exclut BF16 et FP8 natifs ; des chemins logiciels hors profil peuvent néanmoins exister.

Surcoûts initiaux des poids : FP8 2 %, INT8 3 %, INT4 10 %, NVFP4 15 %, ajustables. Surcoûts KV : FP8 2 %, INT8 4 %, INT4 12 %. Ce sont des hypothèses, à remplacer par des mesures du checkpoint et du moteur retenus.

La réserve initiale est **8 Gio par GPU**. Elle couvre approximativement les activations, les buffers, CUDA graphs et la fragmentation. Le pic de préremplissage dépend du batch, du chunked prefill et du backend ; aucune réserve fixe ne garantit l’absence d’OOM.

Une session utilise tout le contexte sélectionné, entrée et sortie comprises. Le simulateur n’estime ni tokens/s, ni latence, ni limite de concurrence du scheduler. Aucun offload CPU, partage de préfixe, beam search ou décodage spéculatif n’est pris en compte.

## Étendre le fichier JSON

Le schéma exécutable se trouve dans `lib/schema.ts`. Racine : `schemaVersion`, `updatedAt`, `title`, `gpus`, `models`.

Un modèle décrit ses caches avec un ou plusieurs blocs :

| Type | Champs spécifiques | Usage |
| --- | --- | --- |
| `gqa` | `layers`, `heads`, `keyDim`, `valueDim`, `window` optionnel | Attention globale ou locale |
| `mla` | `layers`, `latentDim`, `ropeDim` | Latents compressés |
| `state` | `bytes` | État récurrent fixe par session |
| `index` | `layers`, `bytesPerToken` | Cache d’index ou autre structure par token |

Exemple de blocs pour Mistral Small 3.2 :

```json
[
  {
    "kind": "gqa",
    "label": "Attention globale",
    "layers": 40,
    "heads": 8,
    "keyDim": 128,
    "valueDim": 128
  }
]
```

Les champs `weightFormats` et `kvFormats` définissent les formats autorisés par le modèle. Chaque GPU contient une entrée `weights` / `kv` pour chaque format, avec `status` (`native`, `conditional`, `unsupported`) et `reason`. Des restrictions propres à une version de moteur peuvent ainsi être imposées dans un catalogue interne sans modifier l’interface.

La validation refuse les identifiants dupliqués, dimensions négatives, champs manquants, URL non HTTPS et modèles calculables sans blocs de cache.

## Tests et structure

```bash
npm run validate:config
npm test
npm run typecheck
npm run build
```

Les tests vérifient des résultats numériques connus, les fenêtres glissantes, la réplication TP, le MLA, les états fixes, les limites, les configurations invalides et 9 216 combinaisons modèle/GPU/contexte/nombre de GPU. La compilation de production a été vérifiée. La vérification visuelle et les interactions dans un navigateur restent à réaliser dans un environnement qui le permet.

```text
app/                  Interface et styles
lib/schema.ts         Validation du catalogue
lib/catalog.ts        Chargement serveur au démarrage
lib/estimator.ts      Calcul pur, indépendant de React
config/catalog.json   Catalogue modifiable
config/upstream/      Configurations sources consultées
scripts/              Validation en ligne de commande
tests/                Tests du calcul et du schéma
```

## Pistes d’évolution

- Importer la VRAM réellement disponible et les mesures de cache de vLLM/SGLang pour calibrer les hypothèses.
- Ajouter des profils versionnés de moteurs et une gestion explicite TP × PP × EP.
- Comparer côte à côte plusieurs scénarios et produire un plan de capacité à partir de longueurs de sessions différentes.
