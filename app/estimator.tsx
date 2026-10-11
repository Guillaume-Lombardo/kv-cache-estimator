'use client';
import { useEffect, useState } from 'react';
import ModelCombobox from './model-combobox';
import { readScenario, writeScenario } from '@/lib/scenario';
import { Cpu, Database, Layers3, SlidersHorizontal, ArrowUpRight, Download, Info, TriangleAlert, Check, ChevronDown, Link2, CircuitBoard, Gauge, ExternalLink } from 'lucide-react';
import type { Catalog, WeightFormat, KVFormat } from '@/lib/schema';
import { CONTEXTS, estimate, WEIGHT_FORMATS, KV_FORMATS, type Settings } from '@/lib/estimator';
const number = (n: number, digits = 1) => n.toLocaleString('fr-FR', { maximumFractionDigits: digits, minimumFractionDigits: digits });
const tokens = (n: number) => n >= 1048576 ? '1M' : `${number(n / 1024, 0)}k`;
const label = (s: string) => s.replaceAll('-', ' ');
export default function Estimator({ catalog }: { catalog: Catalog }) {
  const first = catalog.models.find(m => m.id.includes('Mistral-Small-3.2')) || catalog.models.find(m => m.status !== 'unavailable')!;
  const initialGPU = catalog.gpus.find(g => g.id === 'h200') || catalog.gpus[0];
  const chooseWeight = (m: typeof first, g: typeof initialGPU) => (['bf16', 'fp16', 'fp8', 'int8', 'int4', 'nvfp4'] as const).find(f => m.weightFormats.includes(f) && g.weights[f].status !== 'unsupported') || m.weightFormats[0];
  const chooseKV = (m: typeof first, g: typeof initialGPU) => (['bf16', 'fp16', 'fp8', 'int8', 'int4'] as const).find(f => m.kvFormats.includes(f) && g.kv[f].status !== 'unsupported') || m.kvFormats[0];
  const [modelId, setModelId] = useState(first.id);
  const [gpuId, setGpuId] = useState(initialGPU.id);
  const [ready, setReady] = useState(false);
  const [message, setMessage] = useState('');
  const [settings, setSettings] = useState<Settings>({ context: 32768, gpuCount: 1, weight: chooseWeight(first, initialGPU), kv: chooseKV(first, initialGPU), reserveGiB: 8, memoryGiB: initialGPU.memoryGiB, weightOverhead: WEIGHT_FORMATS[chooseWeight(first, initialGPU)].overhead * 100, interconnect: initialGPU.interconnects.includes('nvlink') ? 'nvlink' : 'pcie', sliding: true, parallelism: 'tp' });
  const model = catalog.models.find(m => m.id === modelId)!;
  const gpu = catalog.gpus.find(g => g.id === gpuId)!;
  const result = estimate(model, gpu, settings);
  useEffect(() => {
    const fallback = { modelId, gpuId, settings };
    const restore = () => {
      const restored = readScenario(new URLSearchParams(window.location.search), catalog, fallback);
      setModelId(restored.modelId); setGpuId(restored.gpuId); setSettings(restored.settings); setReady(true);
    };
    restore();
    window.addEventListener('popstate', restore);
    return () => window.removeEventListener('popstate', restore);
    // The initial defaults are the fallback for invalid/missing URL fields.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalog]);
  useEffect(() => {
    if (!ready) return;
    const url = new URL(window.location.href);
    url.search = writeScenario({ modelId, gpuId, settings }).toString();
    window.history.replaceState(window.history.state, '', url);
  }, [ready, modelId, gpuId, settings]);
  async function shareScenario() {
    const url = new URL(window.location.href);
    url.search = writeScenario({ modelId, gpuId, settings }).toString();
    try { await navigator.clipboard.writeText(url.href); setMessage('Lien du scénario copié. Le destinataire doit utiliser le même catalogue.'); }
    catch { setMessage('Copiez l’adresse dans la barre du navigateur : elle contient le scénario.'); }
  }
  const set = <K extends keyof Settings>(key: K, value: Settings[K]) => setSettings(s => ({ ...s, [key]: value }));
  function switchSelection(id: string, kind: 'model' | 'gpu') {
    const m = kind === 'model' ? catalog.models.find(v => v.id === id)! : model;
    const g = kind === 'gpu' ? catalog.gpus.find(v => v.id === id)! : gpu;
    const weight = m.weightFormats.includes(settings.weight) && g.weights[settings.weight].status !== 'unsupported' ? settings.weight : chooseWeight(m, g);
    const kv = m.kvFormats.includes(settings.kv) && g.kv[settings.kv].status !== 'unsupported' ? settings.kv : chooseKV(m, g);
    setSettings(s => ({ ...s, weight, kv, weightOverhead: weight === s.weight ? s.weightOverhead : WEIGHT_FORMATS[weight].overhead * 100, ...(kind === 'gpu' ? { memoryGiB: g.memoryGiB, interconnect: g.interconnects.includes('nvlink') ? 'nvlink' : 'pcie' } : {}) }));
    if (kind === 'model') setModelId(id); else setGpuId(id);
    setMessage(weight !== settings.weight || kv !== settings.kv ? `Formats ajustés : poids ${weight.toUpperCase()}, KV ${kv.toUpperCase()}.` : '');
  }
  function exportResult() {
    const blob = new Blob([JSON.stringify({ catalogDate: catalog.updatedAt, model: model.id, modelRevision: model.revision, gpu: gpu.id, settings, result, assumptions: model.notes }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'kv-estimation.json'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  const segments = [{ name: 'Poids du modèle', value: result.weights, cls: 'weights' }, { name: '1 session · caches', value: result.kvPerSession, cls: 'cache' }, { name: 'Réserve', value: result.reserved, cls: 'reserve' }];
  const freeAfterOne = Math.max(0, result.total - result.modelAndKV - result.reserved);
  const over = Math.max(0, result.modelAndKV + result.reserved - result.total);
  const plotTotal = Math.max(result.total, result.modelAndKV + result.reserved);
  return <>
    <header className="topbar"><a className="brand" href="/" aria-label="KV Lab accueil"><span className="brand-mark"><Layers3 size={21}/></span><strong>KV<span>LAB</span></strong><span className="brand-divider"/><span className="brand-description">LLM memory planner</span></a><div className="top-meta"><span className="status-dot"/>Calcul local <span className="version">v1.0</span><a href="https://github.com/Guillaume-Lombardo/kv-cache-estimator" target="_blank" rel="noreferrer" aria-label="Dépôt GitHub"><ArrowUpRight size={19}/></a></div></header>
    <main className="workspace">
      <section className="intro"><div><div className="eyebrow">INFÉRENCE / DIMENSIONNEMENT</div><h1>Combien de mémoire<br className="mobile-break"/> pour votre modèle ?</h1><p>Explorez le coût des poids et du KV cache, avant de réserver vos GPU.</p></div><div className="intro-actions"><button className="export" onClick={shareScenario} disabled={!ready}><Link2 size={16}/>Partager</button><button className="export" onClick={exportResult}><Download size={16}/>Exporter le calcul</button></div></section>
      <div className="dashboard">
        <aside className="panel controls" aria-label="Paramètres de simulation">
          <div className="panel-heading"><SlidersHorizontal size={17}/><h2>Configuration</h2><span className="small-tag">LIVE</span></div>
          <div className="control-section"><div className="section-label"><span>01</span>MODÈLE</div>
            <ModelCombobox models={catalog.models} value={modelId} onChange={id => switchSelection(id, 'model')}/>
            <div className="model-chips"><span>{number(model.parameters / 1e9, 1)}B paramètres</span><span>{model.architecture}</span></div>
            <div className="label-row"><label htmlFor="context">Fenêtre de contexte</label><strong>{tokens(settings.context)} <small>tokens</small></strong></div>
            <input id="context" type="range" min="0" max="7" step="1" value={CONTEXTS.indexOf(settings.context)} aria-valuetext={`${settings.context} tokens`} onChange={e => set('context', CONTEXTS[+e.target.value])}/>
            <div className="range-ticks">{CONTEXTS.map((n, i) => <button type="button" key={n} className={`${settings.context === n ? 'active' : ''} ${n > model.maxContext ? 'beyond' : ''}`} onClick={() => set('context', CONTEXTS[i])} aria-label={`${n} tokens${n > model.maxContext ? ', hors limite du modèle' : ''}`}>{tokens(n)}</button>)}</div>
            <div className="field-hint">Entrée + sortie · limite publiée : {number(model.maxContext, 0)} tokens.</div>
          </div>
          <div className="control-section"><div className="section-label"><span>02</span>MATÉRIEL</div>
            <label htmlFor="gpu">Type de GPU</label><select id="gpu" value={gpuId} onChange={e => switchSelection(e.target.value, 'gpu')}>{catalog.gpus.map(g => <option value={g.id} key={g.id}>{g.name}</option>)}</select>
            <label id="count-label">Nombre de GPU physiques</label><div className="count-buttons" role="group" aria-labelledby="count-label">{Array.from({ length: 8 }, (_, i) => i + 1).map(n => <button type="button" aria-pressed={settings.gpuCount === n} key={n} onClick={() => set('gpuCount', n)} className={settings.gpuCount === n ? 'selected' : ''}>{n}</button>)}</div>
            <label htmlFor="link">Interconnexion entre GPU</label><select id="link" disabled={settings.gpuCount === 1} value={settings.interconnect} onChange={e => set('interconnect', e.target.value as Settings['interconnect'])}><option value="pcie" disabled={!gpu.interconnects.includes('pcie')}>PCIe</option><option value="nvlink" disabled={!gpu.interconnects.includes('nvlink')}>NVLink / NVSwitch{!gpu.interconnects.includes('nvlink') ? ' — indisponible' : ''}</option></select>
            <div className="field-hint">{settings.gpuCount === 1 ? 'Aucune communication inter-GPU.' : !gpu.interconnects.includes('nvlink') ? 'Ce GPU ne dispose pas de NVLink.' : 'À sélectionner uniquement si le serveur est câblé ainsi.'}</div>
          </div>
          <div className="control-section"><div className="section-label"><span>03</span>PRÉCISION & RÉSERVE</div>
            <div className="two-fields"><div><label htmlFor="weight">Poids</label><select id="weight" value={settings.weight} onChange={e => { const w = e.target.value as WeightFormat; setSettings(s => ({ ...s, weight: w, weightOverhead: WEIGHT_FORMATS[w].overhead * 100 })); }}>{Object.entries(WEIGHT_FORMATS).map(([id, f]) => <option key={id} value={id} disabled={gpu.weights[id as WeightFormat].status === 'unsupported' || !model.weightFormats.includes(id as WeightFormat)}>{f.label}{gpu.weights[id as WeightFormat].status === 'unsupported' ? ' ×' : gpu.weights[id as WeightFormat].status === 'conditional' ? ' *' : ''}</option>)}</select></div>
            <div><label htmlFor="kv">KV cache</label><select id="kv" value={settings.kv} onChange={e => set('kv', e.target.value as KVFormat)}>{Object.entries(KV_FORMATS).map(([id, f]) => <option key={id} value={id} disabled={gpu.kv[id as KVFormat].status === 'unsupported' || !model.kvFormats.includes(id as KVFormat)}>{f.label}{gpu.kv[id as KVFormat].status === 'unsupported' ? ' ×' : gpu.kv[id as KVFormat].status === 'conditional' ? ' *' : ''}</option>)}</select></div></div>
            <div className="field-hint">* dépend du moteur / checkpoint · × exclu du profil.</div>
            <details className="format-reasons"><summary>Pourquoi certains formats sont-ils limités ?</summary>{(['weights', 'kv'] as const).map(kind => <div key={kind}><strong>{kind === 'weights' ? 'Poids' : 'KV cache'}</strong>{Object.entries(gpu[kind]).filter(([, v]) => v.status !== 'native').map(([f, v]) => <p key={f}><b>{f.toUpperCase()} · {v.status === 'unsupported' ? 'exclu' : 'conditionnel'}</b> : {v.reason}</p>)}</div>)}</details>
            <div className="label-row"><label htmlFor="reserve">Réserve par GPU</label><div className="number-unit"><input id="reserve" type="number" min="0" max="1024" step="1" value={settings.reserveGiB} onChange={e => set('reserveGiB', Math.max(0, Math.min(1024, Number(e.target.value) || 0)))}/><span>Gio</span></div></div>
            <div className="field-hint">Activations, CUDA graphs, buffers et fragmentation. Valeur de départ : 8 Gio, à mesurer sur votre charge.</div>
            <details className="advanced"><summary>Réglages avancés <ChevronDown size={14}/></summary><div>
              <label htmlFor="memory">VRAM utilisable mesurée / GPU (Gio)</label><input id="memory" type="number" min="1" max="2048" step="0.1" value={+settings.memoryGiB.toFixed(3)} onChange={e => set('memoryGiB', Math.max(1, Math.min(2048, Number(e.target.value) || 1)))}/><p className="field-hint">{gpu.memoryNote}</p>
              <label htmlFor="overhead">Surcoût des poids quantifiés (%)</label><input id="overhead" type="number" min="0" max="100" value={settings.weightOverhead} onChange={e => set('weightOverhead', Math.max(0, Math.min(100, Number(e.target.value) || 0)))}/><p className="field-hint">Métadonnées et couches non quantifiées : hypothèse réglable, pas une mesure du checkpoint.</p>
              <label htmlFor="parallel">Répartition mémoire</label><select id="parallel" value={settings.parallelism} onChange={e => set('parallelism', e.target.value as Settings['parallelism'])}><option value="tp">Tensor parallel · réplications incluses</option><option value="ideal">Partition idéale · borne optimiste</option></select>
              <label className="checkbox"><input type="checkbox" checked={settings.sliding} onChange={e => set('sliding', e.target.checked)}/>Libérer le cache hors fenêtre glissante</label>
            </div></details><p role="status" className="field-hint">{message}</p>
          </div><div className="catalog-footer"><Database size={14}/>{catalog.models.filter(m => m.status !== 'unavailable').length} modèles · {catalog.gpus.length} variantes GPU<br/>Catalogue du {catalog.updatedAt}</div>
        </aside>
        <div className="results">
          <div className="results-title"><div><span className="eyebrow">ESTIMATION MÉMOIRE</span><h2>{label(model.name)}</h2></div><span className={`badge ${result.compatible ? 'good' : 'bad'}`}>{result.compatible ? <Check size={13}/> : <TriangleAlert size={13}/>}{result.compatible ? 'Budget compatible' : 'Configuration à revoir'}</span></div>
          <p className="scenario-line">{settings.gpuCount} × {gpu.name} <span>·</span> {tokens(settings.context)} tokens <span>·</span> {settings.weight.toUpperCase()} / KV {settings.kv.toUpperCase()}</p>
          <section className="metrics" aria-label="Résultats mémoire" aria-live="polite" aria-atomic="true">
            <Metric icon={<Layers3 size={16}/>} title="Poids du modèle" value={result.weights} note={`${number(result.rawWeights)} Gio bruts + ${number(settings.weightOverhead, 0)} % de surcoût`}/>
            <Metric icon={<Database size={16}/>} title="Cache / session" value={result.kvPerSession} note={`${number(result.logicalKV)} Gio logiques · réplications incluses`}/>
            <Metric icon={<Cpu size={16}/>} title="Modèle + 1 session" value={result.modelAndKV} note="Somme sur les GPU · hors réserve"/>
          </section>
          <section className="per-gpu" aria-label="Répartition mémoire par GPU"><div className="per-gpu-title">Par GPU <small>répartition supposée du mode sélectionné</small></div><dl><div><dt>Poids</dt><dd>{number(result.weights / settings.gpuCount)} Gio</dd></div><div><dt>Cache / session</dt><dd>{number(result.kvPerSession / settings.gpuCount)} Gio</dd></div><div><dt>Réserve</dt><dd>{number(settings.reserveGiB)} Gio</dd></div><div><dt>Budget caches</dt><dd>{number(result.available / settings.gpuCount)} Gio</dd></div></dl></section>
          <section className="panel capacity-panel"><div className="capacity-main"><div className="capacity-label"><span className="eyebrow">CAPACITÉ À CONTEXTE PLEIN</span><div className="session-value">{number(result.sessions, 0)}<span>session{result.sessions === 1 ? '' : 's'}</span></div><p>{result.compatible ? 'Borne mémoire, sans garantie de débit ou de latence.' : 'Capacité non validée : voir les points de vigilance.'}</p></div><div className="available"><Database size={20}/><span>Budget total pour les caches</span><strong>{number(result.available)} <small>Gio</small></strong><span>après poids et réserve</span></div></div>
            <div className="budget-head"><span>Répartition avec une session</span><strong>{number(result.total)} Gio disponibles sur {settings.gpuCount} GPU</strong></div>
            <div className="memory-bar" role="img" aria-label={`Poids ${number(result.weights)} Gio ; caches ${number(result.kvPerSession)} Gio ; réserve ${number(result.reserved)} Gio ; libre ${number(freeAfterOne)} Gio ; dépassement ${number(over)} Gio`}>{segments.map(p => <div key={p.cls} className={p.cls} style={{ width: `${p.value / plotTotal * 100}%` }}/ >)}{freeAfterOne > 0 && <div className="free" style={{ width: `${freeAfterOne / plotTotal * 100}%` }}/>}</div>
            <div className="legend">{segments.map(p => <span key={p.cls}><i className={p.cls}/>{p.name}<b>{number(p.value)} Gio</b></span>)}<span><i className="free"/>{over > 0 ? 'Dépassement' : 'Libre après 1 session'}<b className={over > 0 ? 'danger-text' : ''}>{number(over || freeAfterOne)} Gio</b></span></div>
          </section>
          <section className="panel warning-panel"><div className="panel-heading"><Info size={17}/><h2>Points de vigilance</h2><span className="counter">{result.notices.length + model.notes.length + gpu.notes.length}</span></div>
            <div className="notices">{result.notices.length === 0 && <div className="notice info"><Check size={16}/><p>Aucune incompatibilité détectée par les règles du catalogue. Validez les allocations sur votre moteur.</p></div>}{result.notices.map((n, i) => <div key={i} className={`notice ${n.level}`}>{n.level === 'info' ? <Info size={16}/> : <TriangleAlert size={16}/>}<p>{n.text}</p></div>)}{[...gpu.notes, ...model.notes].map((n, i) => <div className="notice info" key={`note-${i}`}><Info size={16}/><p>{n}</p></div>)}</div>
          </section>
          <section className="panel comparison-panel"><div className="panel-heading"><Gauge size={17}/><h2>L’effet du contexte</h2><span className="panel-subtitle">Même matériel, mêmes formats</span></div><div className="table-scroll"><table><thead><tr><th>Contexte</th><th>Cache / session</th><th>Sessions estimées</th><th>Capacité relative</th></tr></thead><tbody>{CONTEXTS.map(c => {
            const r = estimate(model, gpu, { ...settings, context: c }); const max = estimate(model, gpu, { ...settings, context: CONTEXTS[0] }).sessions;
            return <tr key={c} className={c === settings.context ? 'current-row' : ''}><th><button onClick={() => set('context', c)}>{tokens(c)}{c === settings.context && <span>ACTUEL</span>}</button></th><td>{number(r.kvPerSession, 2)} <small>Gio</small></td><td>{c > model.maxContext ? <span className="muted">Hors limite</span> : r.compatible ? number(r.sessions, 0) : <span className="muted">Non validé</span>}</td><td><div className="mini-bar"><span style={{ width: `${max ? r.sessions / max * 100 : 0}%` }}/></div></td></tr>;
          })}</tbody></table></div></section>
          <section className="panel method-panel"><details><summary><span><CircuitBoard size={17}/>Hypothèses, calcul & sources</span><ChevronDown size={16}/></summary><div className="method-content"><h3>Ce que mesure le simulateur</h3><p>Une copie du modèle, des GPU identiques, des sessions indépendantes remplies au contexte choisi. Le contexte comprend le prompt, les tokens générés et les tokens multimodaux. Aucun offload CPU, partage de préfixe, beam search ou décodage spéculatif.</p><p>Les poids incluent tous les experts MoE, pas seulement les experts actifs. La réserve est déduite sur chaque GPU. Les poids sont supposés répartis uniformément ; les contraintes de placement des couches et des experts restent à vérifier.</p><div className="formula">Poids = paramètres × octets / paramètre × (1 + surcoût)<br/>Cache GQA = Σ couches × tokens retenus × têtes KV × (dK + dV) × octets<br/>Cache MLA = couches × contexte × (rang latent + dimension RoPE) × octets<br/>Sessions = ⌊ (VRAM totale − réserve − poids) / caches physiques d’une session ⌋</div><p>1 Gio = 2³⁰ octets ; 1 Go = 10⁹ octets. Ici 8k = 8 192 et 1M = 1 048 576 tokens. Les capacités GPU nominales sont converties depuis les Go constructeur. Utilisez la valeur mesurée via <code>nvidia-smi</code> pour votre machine.</p><h3>Décomposition des caches</h3><div className="table-scroll"><table><thead><tr><th>Composante</th><th>Logique</th><th>Physique · tous GPU</th></tr></thead><tbody>{result.parts.map((p, i) => <tr key={i}><td>{p.label}</td><td>{number(p.logicalGiB, 3)} Gio</td><td>{number(p.physicalGiB, 3)} Gio</td></tr>)}</tbody></table></div><p>{model.parameterNote}</p><p>Surcoûts KV estimés : FP8 +2 %, INT8 +4 %, INT4 +12 %. Les états récurrents et les index conservent leur propre précision, indépendamment du sélecteur KV. Ces valeurs ne remplacent pas une mesure du backend.</p><h3>Compatibilité du profil GPU</h3><ul className="support-list">{Object.entries(gpu.weights).map(([f, s]) => <li key={f}><strong>{f.toUpperCase()} · {s.status === 'unsupported' ? 'exclu' : s.status === 'conditional' ? 'conditionnel' : 'natif'}</strong><span>{s.reason}</span></li>)}</ul><h3>Sources du catalogue</h3><div className="sources">{[...model.sources, ...gpu.sources].map((s, i) => <a key={i} href={s.url} target="_blank" rel="noreferrer">{s.label}<ExternalLink size={13}/></a>)}</div><p className="field-hint">Révision modèle : {model.revision}. Catalogue vérifié le {catalog.updatedAt}. Sélection représentative des sept familles, non exhaustive. « Open weights » ne signifie pas licence sans restriction.</p></div></details></section>
          {catalog.models.some(m => m.status === 'unavailable') && <p className="unavailable-note">{catalog.models.filter(m => m.status === 'unavailable').map(m => `${m.name} : ${m.unavailableReason}`).join(' ')}</p>}
          <footer className="page-footer"><span>KV LAB <span> / </span> Estimer. Comparer. Mesurer.</span><span>Aucun modèle chargé · aucune requête d’inférence</span></footer>
        </div>
      </div>
    </main>
  </>;
}
function Metric({ icon, title, value, note }: { icon: React.ReactNode; title: string; value: number; note: string }) {
  return <article className="panel metric"><div className="metric-label">{icon}{title}</div><strong>{number(value)}<small>Gio</small></strong><p>{note}</p></article>;
}
