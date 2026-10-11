import type { Catalog } from './schema';
import { CONTEXTS, WEIGHT_FORMATS, type Settings } from './estimator';
export type Scenario = { modelId: string; gpuId: string; settings: Settings };
/** URL v1 uses explicit units; links from other schemas are not silently reinterpreted. */
export function readScenario(q: URLSearchParams, catalog: Catalog, fallback: Scenario): Scenario {
  if (q.get('v') !== '1') return fallback;
  const model = catalog.models.find(m => m.id === q.get('model') && m.status !== 'unavailable') || catalog.models.find(m => m.id === fallback.modelId)!;
  const gpu = catalog.gpus.find(g => g.id === q.get('gpu')) || catalog.gpus.find(g => g.id === fallback.gpuId)!;
  const numeric = (key: string, value: number, min: number, max: number, integer = false) => {
    const raw = q.get(key); if (raw === null || !raw.trim()) return value;
    const n = Number(raw); return Number.isFinite(n) && n >= min && n <= max && (!integer || Number.isInteger(n)) ? n : value;
  };
  const weight = model.weightFormats.find(f => f === q.get('weight') && gpu.weights[f].status !== 'unsupported') || model.weightFormats.find(f => f === fallback.settings.weight && gpu.weights[f].status !== 'unsupported') || model.weightFormats.find(f => gpu.weights[f].status !== 'unsupported') || model.weightFormats[0];
  const kv = model.kvFormats.find(f => f === q.get('kv') && gpu.kv[f].status !== 'unsupported') || model.kvFormats.find(f => f === fallback.settings.kv && gpu.kv[f].status !== 'unsupported') || model.kvFormats.find(f => gpu.kv[f].status !== 'unsupported') || model.kvFormats[0];
  const ctx = numeric('context', fallback.settings.context, CONTEXTS[0], CONTEXTS.at(-1)!, true);
  const link = q.get('link');
  return { modelId: model.id, gpuId: gpu.id, settings: {
    ...fallback.settings, weight, kv, context: CONTEXTS.includes(ctx) ? ctx : fallback.settings.context,
    gpuCount: numeric('gpus', fallback.settings.gpuCount, 1, 8, true),
    reserveGiB: numeric('reserveGiB', fallback.settings.reserveGiB, 0, 1024),
    memoryGiB: numeric('memoryGiB', gpu.memoryGiB, 1, 2048),
    weightOverhead: numeric('weightOverhead', WEIGHT_FORMATS[weight].overhead * 100, 0, 100),
    interconnect: (link === 'nvlink' || link === 'pcie') && gpu.interconnects.includes(link) ? link : gpu.interconnects[0],
    parallelism: q.get('parallelism') === 'ideal' ? 'ideal' : 'tp',
    sliding: q.get('sliding') === '0' ? false : q.get('sliding') === '1' ? true : fallback.settings.sliding,
  } };
}
export function writeScenario(s: Scenario) {
  return new URLSearchParams({ v: '1', model: s.modelId, gpu: s.gpuId, context: String(s.settings.context), gpus: String(s.settings.gpuCount), weight: s.settings.weight, kv: s.settings.kv, reserveGiB: String(s.settings.reserveGiB), memoryGiB: String(s.settings.memoryGiB), weightOverhead: String(s.settings.weightOverhead), link: s.settings.interconnect, parallelism: s.settings.parallelism, sliding: s.settings.sliding ? '1' : '0' });
}
