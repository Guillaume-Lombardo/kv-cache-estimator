import { z } from 'zod';
const pos = z.number().finite().positive();
const integer = pos.int();
const source = z.strictObject({ label: z.string().min(1), url: z.url().refine(v => v.startsWith('https://'), 'HTTPS requis') });
export const weightIds = ['fp16', 'bf16', 'fp8', 'int8', 'int4', 'nvfp4'] as const;
export const kvIds = ['fp16', 'bf16', 'fp8', 'int8', 'int4'] as const;
const weightId = z.enum(weightIds);
const kvId = z.enum(kvIds);
const support = z.strictObject({ status: z.enum(['native', 'conditional', 'unsupported']), reason: z.string().min(1) });
const block = z.discriminatedUnion('kind', [
  z.strictObject({ kind: z.literal('gqa'), label: z.string(), layers: integer, heads: integer, keyDim: integer, valueDim: integer, window: integer.optional() }),
  z.strictObject({ kind: z.literal('mla'), label: z.string(), layers: integer, latentDim: integer, ropeDim: z.number().int().nonnegative() }),
  z.strictObject({ kind: z.literal('state'), label: z.string(), bytes: pos }),
  z.strictObject({ kind: z.literal('index'), label: z.string(), layers: integer, bytesPerToken: pos }),
]);
const model = z.strictObject({
  id: z.string().min(1), name: z.string().min(1), family: z.string().min(1), architecture: z.string().min(1),
  parameters: pos, parameterNote: z.string().min(1), activeParameters: pos.optional(), maxContext: integer,
  weightStorage: z.partialRecord(weightId, z.strictObject({ baseBytes: pos, note: z.string().min(1), source })).optional(),
  queryHeads: integer, status: z.enum(['estimated', 'experimental', 'unavailable']),
  unavailableReason: z.string().optional(), blocks: z.array(block),
  weightFormats: z.array(weightId).min(1), kvFormats: z.array(kvId).min(1),
  notes: z.array(z.string()), sources: z.array(source).min(1), revision: z.string(),
}).superRefine((m, ctx) => {
  if (m.activeParameters !== undefined && m.activeParameters > m.parameters) ctx.addIssue({ code: 'custom', message: 'Les paramètres actifs ne peuvent pas dépasser le total.', path: ['activeParameters'] });
  m.blocks.forEach((b, index) => {
    if (b.kind === 'gqa' && m.queryHeads % b.heads !== 0) ctx.addIssue({ code: 'custom', message: 'Le nombre de têtes de requête doit être divisible par le nombre de têtes KV.', path: ['blocks', index, 'heads'] });
  });
  for (const format of Object.keys(m.weightStorage || {}) as WeightFormat[]) {
    if (!m.weightFormats.includes(format)) ctx.addIssue({ code: 'custom', message: 'Stockage renseigné pour un format exclu du modèle.', path: ['weightStorage', format] });
  }

  if (m.status !== 'unavailable' && !m.blocks.length) ctx.addIssue({ code: 'custom', message: 'Un modèle calculable doit définir ses caches.', path: ['blocks'] });
  if (m.status === 'unavailable' && !m.unavailableReason) ctx.addIssue({ code: 'custom', message: 'Motif requis.', path: ['unavailableReason'] });
});
export const catalogSchema = z.strictObject({
  schemaVersion: z.literal(1), updatedAt: z.iso.date(), title: z.string().min(1),
  gpus: z.array(z.strictObject({
    id: z.string().min(1), name: z.string().min(1), architecture: z.string().min(1), memoryGiB: pos,
    memoryNote: z.string().min(1), interconnects: z.array(z.enum(['pcie', 'nvlink'])).min(1),
    notes: z.array(z.string()), sources: z.array(source).min(1),
    weights: z.record(weightId, support), kv: z.record(kvId, support),
  })).min(1), models: z.array(model).min(1),
}).superRefine((c, ctx) => {
  for (const key of ['models', 'gpus'] as const) {
    const ids = new Set<string>();
    c[key].forEach((item, i) => {
      if (ids.has(item.id)) ctx.addIssue({ code: 'custom', message: 'Identifiant dupliqué', path: [key, i, 'id'] });
      ids.add(item.id);
    });
  }
  if (!c.models.some(m => m.status !== 'unavailable')) ctx.addIssue({ code: 'custom', message: 'Au moins un modèle calculable est requis.', path: ['models'] });
});
export type Catalog = z.infer<typeof catalogSchema>;
export type Model = Catalog['models'][number];
export type GPU = Catalog['gpus'][number];
export type WeightFormat = typeof weightIds[number];
export type KVFormat = typeof kvIds[number];
