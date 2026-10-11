'use client';
// Search-and-select interaction inspired by claude/great-dijkstra-9wcc65.
import { useEffect, useId, useRef, useState } from 'react';
import { Search } from 'lucide-react';
import type { Model } from '@/lib/schema';
export default function ModelCombobox({ models, value, onChange }: { models: Model[]; value: string; onChange: (id: string) => void }) {
  const id = useId();
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const list = useRef<HTMLUListElement>(null);
  const selected = models.find(m => m.id === value)!;
  const terms = query.toLocaleLowerCase('fr').split(/\s+/).filter(Boolean);
  const filtered = models.filter(m => terms.every(t => `${m.name} ${m.id} ${m.family} ${m.architecture} ${Math.round(m.parameters / 1e9)}B`.toLocaleLowerCase('fr').includes(t)));
  const enabled = filtered.map((m, i) => m.status !== 'unavailable' ? i : -1).filter(i => i >= 0);
  useEffect(() => { list.current?.querySelector(`[data-index="${active}"]`)?.scrollIntoView({ block: 'nearest' }); }, [active, open]);
  function choose(m: Model) { if (m.status === 'unavailable') return; onChange(m.id); setOpen(false); setQuery(''); }
  function openList() { setQuery(''); setActive(Math.max(0, models.findIndex(m => m.id === value))); setOpen(true); }
  function move(direction: number) {
    const pos = enabled.indexOf(active);
    setActive(enabled[Math.max(0, Math.min(enabled.length - 1, pos + direction))] ?? -1);
  }
  return <div className="model-combo">
    <label htmlFor={id}>Rechercher et sélectionner un modèle</label>
    <div className="search-wrap"><Search size={15}/><input id={id} role="combobox" aria-expanded={open} aria-controls={`${id}-list`} aria-autocomplete="list" aria-activedescendant={open && filtered[active] ? `${id}-option-${active}` : undefined} autoComplete="off" placeholder={selected.name} value={open ? query : selected.name} onFocus={openList} onClick={() => { if (!open) openList(); }} onBlur={() => setOpen(false)} onChange={e => { setQuery(e.target.value); setActive(-1); setOpen(true); }} onKeyDown={e => {
      if (!open && ['ArrowDown', 'ArrowUp', 'Enter'].includes(e.key)) { openList(); e.preventDefault(); return; }
      if (!open) return;
      if (e.key === 'ArrowDown') move(1);
      else if (e.key === 'ArrowUp') move(-1);
      else if (e.key === 'Home') setActive(enabled[0] ?? -1);
      else if (e.key === 'End') setActive(enabled.at(-1) ?? -1);
      else if (e.key === 'Enter') { const m = filtered[active] || filtered[enabled[0]]; if (m) choose(m); }
      else if (e.key === 'Escape') setOpen(false);
      else return;
      e.preventDefault();
    }}/></div>
    {open && <ul ref={list} id={`${id}-list`} role="listbox" aria-label="Modèles disponibles" className="combo-list">
      {!filtered.length && <li role="presentation" className="combo-empty">Aucun modèle pour « {query} ».</li>}
      {filtered.map((m, i) => <li key={m.id} id={`${id}-option-${i}`} role="option" aria-selected={m.id === value} aria-disabled={m.status === 'unavailable'} data-index={i} className={`combo-option ${active === i ? 'active' : ''}`} onMouseEnter={() => setActive(i)} onMouseDown={e => e.preventDefault()} onClick={() => choose(m)}>
        <strong>{m.name.replaceAll('-', ' ')}</strong><span>{m.family} · {Math.round(m.parameters / 1e9)}B · {m.architecture}</span><small>{m.status === 'unavailable' ? m.unavailableReason : `${m.maxContext.toLocaleString('fr-FR')} tokens${m.status === 'experimental' ? ' · estimation hybride' : ''}`}</small>
      </li>)}
    </ul>}
    <div className="field-hint">Recherche par nom, famille ou architecture (MLA, MoE…).</div>
  </div>;
}
