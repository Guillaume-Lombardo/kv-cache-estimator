import type { GPU, Model, WeightFormat, KVFormat } from './schema';
export const GiB = 2 ** 30;
export const CONTEXTS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
export const WEIGHT_FORMATS: Record<WeightFormat, { label: string; bytes: number; overhead: number }> = {
  fp16: { label: 'FP16 · 16 bits', bytes: 2, overhead: 0 },
  bf16: { label: 'BF16 · 16 bits', bytes: 2, overhead: 0 },
  fp8: { label: 'FP8 · 8 bits', bytes: 1, overhead: .02 },
  int8: { label: 'INT8 · 8 bits', bytes: 1, overhead: .03 },
  int4: { label: 'INT4 · 4 bits', bytes: .5, overhead: .10 },
  nvfp4: { label: 'NVFP4 · 4 bits', bytes: .5, overhead: .15 },
};
export const KV_FORMATS: Record<KVFormat, { label: string; bytes: number; overhead: number }> = {
  fp16: { label: 'FP16 · 16 bits', bytes: 2, overhead: 0 },
  bf16: { label: 'BF16 · 16 bits', bytes: 2, overhead: 0 },
  fp8: { label: 'FP8 · 8 bits', bytes: 1, overhead: .02 },
  int8: { label: 'INT8 · 8 bits', bytes: 1, overhead: .04 },
  int4: { label: 'INT4 · 4 bits', bytes: .5, overhead: .12 },
};
export type Settings = {
  context: number; gpuCount: number; weight: WeightFormat; kv: KVFormat;
  reserveGiB: number; memoryGiB: number; weightOverhead: number;
  interconnect: 'pcie' | 'nvlink'; sliding: boolean; parallelism: 'tp' | 'ideal';
};
export type Notice = { level: 'error' | 'warning' | 'info'; text: string };
export function estimate(model: Model, gpu: GPU, s: Settings) {
  if (!Number.isInteger(s.gpuCount) || s.gpuCount < 1 || s.gpuCount > 8 || !CONTEXTS.includes(s.context) || !Number.isFinite(s.memoryGiB) || s.memoryGiB <= 0 || !Number.isFinite(s.reserveGiB) || s.reserveGiB < 0 || !Number.isFinite(s.weightOverhead) || s.weightOverhead < 0 || s.weightOverhead > 100) throw new Error('Paramètres de simulation invalides');
  const notices: Notice[] = [];
  const add = (level: Notice['level'], text: string) => notices.push({ level, text });
  const count = s.gpuCount;
  const tp = s.parallelism === 'tp';
  const w = WEIGHT_FORMATS[s.weight], kv = KV_FORMATS[s.kv];
  const rawWeights = model.parameters * w.bytes / GiB;
  const weights = rawWeights * (1 + s.weightOverhead / 100);
  const total = s.memoryGiB * count;
  const reserved = s.reserveGiB * count;
  const available = Math.max(0, total - reserved - weights);
  if (s.reserveGiB >= s.memoryGiB) add('error', 'La réserve atteint ou dépasse la mémoire de chaque GPU.');
  if (model.status === 'unavailable') add('error', model.unavailableReason || 'Architecture indisponible.');
  if (model.status === 'experimental') add('warning', 'Architecture hybride : états et caches auxiliaires estimés. Valider les allocations avec le moteur retenu.');
  if (s.context > model.maxContext) add('error', `Contexte hors limite publiée (${model.maxContext.toLocaleString('fr-FR')} tokens). L’extrapolation mémoire est affichée, mais aucune capacité de déploiement n’est validée.`);
  for (const [kind, format, capability, allowed] of [
    ['Poids', s.weight, gpu.weights[s.weight], model.weightFormats.includes(s.weight)],
    ['KV cache', s.kv, gpu.kv[s.kv], model.kvFormats.includes(s.kv)],
  ] as const) {
    if (!allowed || capability.status === 'unsupported') add('error', `${kind} ${format.toUpperCase()} : ${!allowed ? 'format exclu par le catalogue du modèle.' : capability.reason}`);
    else if (capability.status === 'conditional') add('warning', `${kind} ${format.toUpperCase()} : ${capability.reason}`);
  }
  if (tp && model.queryHeads % count !== 0) add('error', `${model.queryHeads} têtes de requête ne se divisent pas par ${count} GPU : TP standard non compatible. Le mode partition idéale reste une borne théorique.`);
  if (count > 1 && !gpu.interconnects.includes(s.interconnect)) add('error', 'Interconnexion indisponible pour cette variante GPU.');
  if (count > 1 && s.interconnect === 'pcie') add('warning', 'Multi-GPU sur PCIe : faisable selon le moteur, mais communications plus lentes. Vérifier le P2P et la topologie NUMA.');
  if (count > 1 && s.interconnect === 'nvlink') add('warning', 'NVLink nécessite un serveur et une topologie effectivement câblés. La VRAM n’est pas automatiquement unifiée.');
  if (!tp && count > 1) add('warning', 'Partition idéale : poids et caches parfaitement répartis. Cette borne optimiste ne valide ni TP, ni PP, ni EP.');
  const parts = model.blocks.map(b => {
    let logicalBytes = 0, physicalBytes = 0;
    if (b.kind === 'gqa') {
      const tokens = s.sliding && b.window ? Math.min(s.context, b.window) : s.context;
      logicalBytes = b.layers * tokens * b.heads * (b.keyDim + b.valueDim) * kv.bytes * (1 + kv.overhead);
      const physicalHeads = tp ? Math.max(1, Math.ceil(b.heads / count)) * count : b.heads;
      physicalBytes = logicalBytes * physicalHeads / b.heads;
      if (tp && b.heads % count !== 0 && count % b.heads !== 0) add('error', `${b.label} : ${b.heads} têtes KV ne se partitionnent pas uniformément sur ${count} GPU.`);
      else if (tp && count > b.heads) add('warning', `${b.label} : réplication des ${b.heads} têtes KV sur ${count} GPU prise en compte.`);
    } else if (b.kind === 'mla') {
      logicalBytes = b.layers * s.context * (b.latentDim + b.ropeDim) * kv.bytes * (1 + kv.overhead);
      physicalBytes = logicalBytes * (tp ? count : 1);
    } else if (b.kind === 'state') {
      logicalBytes = b.bytes; physicalBytes = logicalBytes * (tp ? count : 1);
    } else {
      logicalBytes = b.layers * s.context * b.bytesPerToken; physicalBytes = logicalBytes * (tp ? count : 1);
    }
    return { label: b.label, kind: b.kind, logicalGiB: logicalBytes / GiB, physicalGiB: physicalBytes / GiB };
  });
  if (model.blocks.some(b => b.kind === 'mla')) add('info', 'MLA : latent compressé + composante RoPE. En TP, le cache latent est supposé répliqué sur chaque GPU. Le moteur doit conserver le MLA compressé.');
  if (model.blocks.some(b => b.kind === 'state')) add('info', 'Les états récurrents ont une taille fixe, indépendante du format KV. En TP, leur réplication intégrale est une hypothèse conservatrice.');
  if (s.sliding && model.blocks.some(b => b.kind === 'gqa' && b.window)) add('warning', 'Fenêtre glissante optimisée : suppose que le moteur libère les tokens hors fenêtre. Désactiver pour une allocation complète.');
  if (s.weight !== 'fp16' && s.weight !== 'bf16') add('warning', 'Quantification des poids : projection de capacité, pas une garantie de checkpoint disponible. Les couches non quantifiées et les métadonnées dépendent du format réel.');
  if (s.reserveGiB === 0) add('warning', 'Réserve nulle : activations, préremplissage, CUDA graphs et espace de travail ne sont pas couverts.');
  if (weights > total - reserved) add('error', `Les poids dépassent le budget mémoire de ${(weights - total + reserved).toFixed(1)} Gio.`);
  const logicalKV = parts.reduce((a, p) => a + p.logicalGiB, 0);
  const kvPerSession = parts.reduce((a, p) => a + p.physicalGiB, 0);
  const memorySessions = kvPerSession > 0 ? Math.floor(available / kvPerSession) : 0;
  if (memorySessions === 0 && weights <= total - reserved && kvPerSession > 0) add('error', 'Mémoire insuffisante pour une session au contexte choisi.');
  const compatible = !notices.some(n => n.level === 'error');
  return { rawWeights, weights, total, reserved, available, logicalKV, kvPerSession, modelAndKV: weights + kvPerSession, parts, memorySessions, sessions: compatible ? memorySessions : 0, compatible, notices };
}
