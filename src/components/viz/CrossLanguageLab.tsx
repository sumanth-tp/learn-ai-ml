import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Strategy = 'query' | 'document' | 'embedding';
const STRATEGIES: Record<Strategy, string> = {query: 'Translate the query', document: 'Translate every document', embedding: 'Encode into a shared space'};

export default function CrossLanguageLab() {
  const dark = useDarkViz();
  const [strategy, setStrategy] = useState<Strategy>('query');
  const [documents, setDocuments] = useState(1000);
  const [context, setContext] = useState<'finance' | 'furniture'>('finance');
  const units = {query: 1, document: documents, embedding: documents + 1};
  const translation = context === 'finance' ? 'bank' : 'bench';
  const table = {columns: ['strategy', 'work units', 'what moves across languages'], rows: [
    ['translate query', units.query, 'one short query, with limited context'],
    ['translate documents', units.document, 'collection text before lexical search'],
    ['shared embeddings', units.embedding, 'documents and query encoded once each'],
  ]};

  return <VizPanel title="Choose a cross-language retrieval bridge"
    hint="Work units are illustrative counts of texts processed, not measured cost or model accuracy. The Spanish word banco needs context."
    table={table}
    controls={<div className={s.controls}>
      <label className={s.control}>strategy
        <select className={s.select} value={strategy} aria-label="Cross-language strategy" onChange={(event) => setStrategy(event.target.value as Strategy)}>
          {Object.entries(STRATEGIES).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </label>
      <label className={s.control}>documents: {documents}
        <input type="range" min="100" max="2000" step="100" value={documents} aria-label="Documents to search"
          onChange={(event) => setDocuments(Number(event.target.value))} />
      </label>
      <label className={s.control}>query context
        <select className={s.select} value={context} aria-label="Query context" onChange={(event) => setContext(event.target.value as typeof context)}>
          <option value="finance">finance</option><option value="furniture">furniture</option>
        </select>
      </label>
    </div>}>
    <div style={{display: 'grid', gap: '0.7rem'}}>
      <div>Spanish <strong>banco</strong> in {context} context → English <strong>{translation}</strong></div>
      <div style={{height: '1.5rem', background: 'var(--ifm-color-emphasis-200)', borderRadius: '0.25rem'}}>
        <div style={{height: '100%', width: `${Math.max(1, units[strategy] / (documents + 1) * 100)}%`, background: seriesColor(0, dark), borderRadius: '0.25rem'}} />
      </div>
      <div aria-live="polite">{STRATEGIES[strategy]}: <strong>{units[strategy]} illustrative work unit{units[strategy] === 1 ? '' : 's'}</strong></div>
    </div>
  </VizPanel>;
}
