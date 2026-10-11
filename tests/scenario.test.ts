import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stringify } from 'yaml';
import { parseCatalog } from '../lib/parse-catalog';
import { readScenario, writeScenario, type Scenario } from '../lib/scenario';
const text = readFileSync('config/catalog.json', 'utf8');
const catalog = parseCatalog(text, 'catalog.json');
const fallback: Scenario = { modelId: 'mistralai/Mistral-Small-3.2-24B-Instruct-2506', gpuId: 'h200', settings: { context: 65536, gpuCount: 4, weight: 'int4', kv: 'fp8', reserveGiB: 9.5, memoryGiB: 130.125, weightOverhead: 13, interconnect: 'nvlink', sliding: false, parallelism: 'ideal' } };
test('shareable URL preserves every setting, including custom GiB capacity and overhead', () => {
  assert.deepEqual(readScenario(writeScenario(fallback), catalog, fallback), fallback);
});
test('invalid URL values do not cause NaN, unsupported selections or out-of-range controls', () => {
  const q = new URLSearchParams('v=1&gpus=3.2&context=12000&memoryGiB=Infinity&reserveGiB=-20&weight=invalid&kv=invalid&model=absent&gpu=absent&weightOverhead=NaN');
  const restored = readScenario(q, catalog, fallback);
  assert.equal(restored.settings.gpuCount, fallback.settings.gpuCount);
  assert.equal(restored.settings.context, fallback.settings.context);
  assert.equal(restored.settings.reserveGiB, fallback.settings.reserveGiB);
  assert.ok(Number.isFinite(restored.settings.memoryGiB));
  assert.equal(restored.modelId, fallback.modelId);
});
test('restoring V100 falls back from unsupported BF16/FP8 and rejects NVLink on PCIe', () => {
  const q = writeScenario(fallback); q.set('gpu', 'v100-pcie-16'); q.set('weight', 'bf16'); q.set('kv', 'bf16');
  const restored = readScenario(q, catalog, fallback);
  const gpu = catalog.gpus.find(g => g.id === restored.gpuId)!;
  assert.notEqual(gpu.weights[restored.settings.weight].status, 'unsupported');
  assert.notEqual(gpu.kv[restored.settings.kv].status, 'unsupported');
  assert.equal(restored.settings.interconnect, 'pcie');
});
test('legacy or unversioned URL is not interpreted with different reserve units', () => {
  assert.deepEqual(readScenario(new URLSearchParams('r=10&n=8'), catalog, fallback), fallback);
});
test('JSON and YAML produce identical validated catalogues', () => {
  assert.deepEqual(parseCatalog(stringify(JSON.parse(text)), 'catalog.yaml'), catalog);
});
test('catalogue rejects typo fields, duplicate YAML keys and nested unknown fields', () => {
  const raw = JSON.parse(text); raw.gpus[0].memoryGB = 32;
  assert.throws(() => parseCatalog(JSON.stringify(raw), 'catalog.json'), /memoryGB/);
  assert.throws(() => parseCatalog('schemaVersion: 1\nschemaVersion: 1', 'catalog.yaml'), /unique/);
  assert.throws(() => parseCatalog('{"schemaVersion":', 'catalog.json'));
});
