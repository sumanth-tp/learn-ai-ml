import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function TfIdfLab() {
  const dark = useDarkViz();
  const [documents, setDocuments] = useState(1000);
  const [documentFrequency, setDocumentFrequency] = useState(100);
  const [termFrequency, setTermFrequency] = useState(3);
  const idf = Math.log10(documents / documentFrequency);
  const weight = termFrequency * idf;
  const cosine = 2 / Math.sqrt(6);

  return (
    <VizPanel title="Explore tf-idf weights"
      hint="The lecture uses base-10 IDF. Rarer terms weigh more; cosine then removes vector-length effects from the comparison."
      table={{columns: ['quantity', 'calculation', 'value'], rows: [
        ['document count N', 'collection size', documents],
        ['document frequency df', 'documents containing term', documentFrequency],
        ['term frequency tf', 'occurrences in one document', termFrequency],
        ['IDF', 'log10(N / df)', idf.toFixed(3)],
        ['tf-idf', 'tf × IDF', weight.toFixed(3)],
        ['fixed vector cosine', '2 / sqrt(6)', cosine.toFixed(3)],
      ]}}
      controls={<div className={s.controls}>
        <label className={s.control}>documents N: {documents}
          <input type="range" min="100" max="2000" step="100" value={documents} aria-label="Documents in collection"
            onChange={(event) => {
              const next = Number(event.target.value);
              setDocuments(next);
              setDocumentFrequency((current) => Math.min(current, next));
            }} />
        </label>
        <label className={s.control}>document frequency df: {documentFrequency}
          <input type="range" min="1" max={documents} value={documentFrequency} aria-label="Documents containing term"
            onChange={(event) => setDocumentFrequency(Number(event.target.value))} />
        </label>
        <label className={s.control}>term frequency tf: {termFrequency}
          <input type="range" min="1" max="10" value={termFrequency} aria-label="Term occurrences in document"
            onChange={(event) => setTermFrequency(Number(event.target.value))} />
        </label>
      </div>}>
      <div style={{display: 'grid', gap: '0.75rem'}}>
        <div>log₁₀({documents} / {documentFrequency}) = <strong>{idf.toFixed(3)}</strong></div>
        <div>{termFrequency} × {idf.toFixed(3)} = <strong>{weight.toFixed(3)}</strong></div>
        <div style={{height: '1.25rem', background: 'var(--ifm-color-emphasis-200)', borderRadius: '0.25rem', overflow: 'hidden'}}>
          <div style={{height: '100%', width: `${Math.min(100, weight / 10 * 100)}%`, background: seriesColor(0, dark)}} />
        </div>
        <div>For q = [1,0,1,0] and d = [1,1,1,0], cosine = <strong>{cosine.toFixed(3)}</strong>.</div>
      </div>
    </VizPanel>
  );
}
