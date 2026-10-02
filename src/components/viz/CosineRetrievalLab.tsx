import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function CosineRetrievalLab() {
  const dark = useDarkViz();
  const [second, setSecond] = useState(1);
  const [threshold, setThreshold] = useState(0.6);
  const query = [1, 0, 1, 1];
  const document = [1, second, 1, 0];
  const dot = query.reduce((sum, value, index) => sum + value * document[index], 0);
  const queryNorm = Math.sqrt(query.reduce((sum, value) => sum + value ** 2, 0));
  const documentNorm = Math.sqrt(document.reduce((sum, value) => sum + value ** 2, 0));
  const cosine = dot / (queryNorm * documentNorm);

  return (
    <VizPanel
      title="Cosine similarity and a retrieval threshold"
      hint="The default cosine is 2/3. A real retrieval result also depends on other chunks, filters, top-k and index recall."
      table={{columns: ['dimension', 'query', 'chunk', 'product'], rows: query.map((value, index) => [
        String(index + 1), String(value), String(document[index]), String(value * document[index]),
      ])}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            chunk dimension 2: {second.toFixed(1)}
            <input type="range" min="0" max="2" step="0.1" value={second} aria-label="Chunk second dimension"
              onChange={(event) => setSecond(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            illustrative threshold: {threshold.toFixed(2)}
            <input type="range" min="0.2" max="0.9" step="0.05" value={threshold} aria-label="Retrieval similarity threshold"
              onChange={(event) => setThreshold(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.75rem'}}>
        <strong>q·d = {dot.toFixed(1)}; cosine = {cosine.toFixed(3)}; {cosine >= threshold ? 'above' : 'below'} this illustrative threshold</strong>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '0.5rem'}}>
          {query.map((value, index) => (
            <div key={index} style={{borderRadius: '0.35rem', padding: '0.55rem', background: seriesColor(index, dark), color: '#fff'}}>
              dim {index + 1}<br />q {value}<br />d {document[index].toFixed(1)}
            </div>
          ))}
        </div>
      </div>
    </VizPanel>
  );
}
