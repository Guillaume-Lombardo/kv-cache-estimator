import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { catalogSchema, type Model } from '../lib/schema';
import { estimate, GiB, CONTEXTS, type Settings } from '../lib/estimator';
const raw = JSON.parse(readFileSync('config/catalog.json', 'utf8'));
const catalog = catalogSchema.parse(raw);
const gpu = catalog.gpus.find(g => g.id === 'h200')!;
const model = catalog.models.find(m => m.id.includes('Mistral-Small-3.2'))!;
const s: Settings = { context: 32768, gpuCount: 1, weight: 'bf16', kv: 'bf16', reserveGiB: 8, memoryGiB: 80, weightOverhead: 0, interconnect: 'nvlink', sliding: true, parallelism: 'tp' };
test('GQA oracle: 40 layers × 32768 tokens × 8 heads × 256 × 2 = 5 GiB', () => {
  const r = estimate(model, gpu, s);
  assert.equal(r.kvPerSession, 5);
  assert.equal(r.weights, 24011361280 * 2 / GiB);
  assert.equal(r.available, 80 - 8 - r.weights);
  assert.equal(r.sessions, Math.floor(r.available / 5));
});
test('doubling context doubles full attention cache but leaves weights unchanged', () => {
  const a = estimate(model, gpu, s), b = estimate(model, gpu, { ...s, context: 65536 });
  assert.equal(b.weights, a.weights); assert.equal(b.kvPerSession, a.kvPerSession * 2);
});
test('Gemma global/local dimensions and bounded sliding cache', () => {
  const m = catalog.models.find(m => m.id === 'google/gemma-4-31B')!;
  const r = estimate(m, gpu, s);
  const expected = (10 * 32768 * 4 * 1024 * 2 + 50 * 1024 * 16 * 512 * 2) / GiB;
  assert.equal(r.kvPerSession, expected);
  assert.ok(estimate(m, gpu, { ...s, sliding: false }).kvPerSession > r.kvPerSession);
});
test('TP replicates KV heads when GPU count exceeds KV heads', () => {
  const m: Model = { ...model, blocks: [{ kind: 'gqa', label: 'GQA', layers: 1, heads: 2, keyDim: 128, valueDim: 128 }] };
  const a = estimate(m, gpu, s), b = estimate(m, gpu, { ...s, gpuCount: 8 });
  assert.equal(b.logicalKV, a.logicalKV); assert.equal(b.kvPerSession, a.kvPerSession * 4);
});
test('MLA counts one compressed latent + RoPE, replicated per TP rank', () => {
  const m: Model = { ...model, blocks: [{ kind: 'mla', label: 'MLA', layers: 61, latentDim: 512, ropeDim: 64 }] };
  const r = estimate(m, gpu, { ...s, gpuCount: 4 });
  assert.equal(r.kvPerSession, 61 * 32768 * 576 * 2 * 4 / GiB);
  assert.equal(estimate(m, gpu, { ...s, gpuCount: 4, parallelism: 'ideal' }).kvPerSession, r.kvPerSession / 4);
});
test('fixed recurrent state is neither proportional to context nor KV precision', () => {
  const m: Model = { ...model, blocks: [{ kind: 'state', label: 'SSM', bytes: 4 * GiB }] };
  assert.equal(estimate(m, gpu, s).kvPerSession, 4);
  assert.equal(estimate(m, gpu, { ...s, kv: 'int4', context: 65536 }).kvPerSession, 4);
});
test('FP8 KV includes explicit metadata overhead', () => {
  assert.equal(estimate(model, gpu, { ...s, kv: 'fp8' }).kvPerSession, 5 / 2 * 1.02);
});
test('format, context and TP incompatibilities produce zero validated sessions', () => {
  for (const scenario of [{ ...s, context: 1048576 }, { ...s, gpuCount: 3 }]) {
    const r = estimate(model, gpu, scenario); assert.equal(r.compatible, false); assert.equal(r.sessions, 0);
  }
  const v100 = catalog.gpus.find(g => g.id === 'v100-pcie-16')!;
  assert.equal(estimate(model, v100, s).compatible, false);
  const l40 = catalog.gpus.find(g => g.id === 'l40s')!;
  assert.equal(estimate(model, l40, { ...s, gpuCount: 2 }).compatible, false);
});
test('OOM, full reservation and unavailable models never yield usable sessions', () => {
  for (const scenario of [{ ...s, memoryGiB: 20 }, { ...s, reserveGiB: 80 }, { ...s, reserveGiB: 79 }]) {
    const r = estimate(model, gpu, scenario); assert.equal(r.sessions, 0); assert.ok(r.available >= 0);
  }
  assert.equal(estimate(catalog.models.find(m => m.status === 'unavailable')!, gpu, s).sessions, 0);
});
test('bad scalar settings rejected rather than generating NaN', () => {
  for (const scenario of [{ ...s, gpuCount: 0 }, { ...s, gpuCount: 9 }, { ...s, context: 0 }, { ...s, memoryGiB: NaN }, { ...s, reserveGiB: -1 }]) assert.throws(() => estimate(model, gpu, scenario));
});
test('configuration rejects duplicates, absent caches, invalid dimensions and insecure URLs', () => {
  const mutations = [
    (x: typeof raw) => x.models.push(x.models[0]),
    (x: typeof raw) => x.models[0].blocks = [],
    (x: typeof raw) => x.gpus[0].memoryGiB = -1,
    (x: typeof raw) => x.gpus[0].sources[0].url = 'javascript:alert(1)',
    (x: typeof raw) => delete x.gpus[0].weights.fp16,
  ];
  for (const mutate of mutations) { const copy = structuredClone(raw); mutate(copy); assert.equal(catalogSchema.safeParse(copy).success, false); }
});
test('all catalogue combinations return finite and nonnegative memory and coherent sessions', () => {
  for (const m of catalog.models.filter(m => m.status !== 'unavailable')) for (const g of catalog.gpus) for (const context of CONTEXTS) for (const count of [1, 2, 3, 8]) {
    const r = estimate(m, g, { ...s, context, gpuCount: count, memoryGiB: g.memoryGiB, interconnect: g.interconnects[0] });
    for (const key of ['weights', 'kvPerSession', 'logicalKV', 'available', 'sessions'] as const) assert.ok(Number.isFinite(r[key]) && r[key] >= 0, `${m.id} ${g.id} ${key}`);
    if (r.sessions > 0) { assert.ok(r.weights + r.reserved + r.sessions * r.kvPerSession <= r.total + 1e-9); assert.ok(r.weights + r.reserved + (r.sessions + 1) * r.kvPerSession > r.total); }
  }
});
