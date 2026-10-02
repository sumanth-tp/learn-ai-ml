import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function IndexOverlapLab() {
  const dark = useDarkViz();
  const [pA, setPA] = useState(0.4);
  const [pB, setPB] = useState(0.5);
  const ratio = pB / pA;
  const table = {columns: ['sample', 'overlap probability', 'meaning'], rows: [
    ['random page from A', pA.toFixed(2), 'also found in B'],
    ['random page from B', pB.toFixed(2), 'also found in A'],
    ['estimated |A| / |B|', ratio.toFixed(2), 'p_B / p_A'],
  ]};

  return <VizPanel title="Estimate relative web index size"
    hint="Overlap sampling estimates a ratio, assuming both probabilities describe the same intersection under comparable sampling."
    table={table}
    controls={<div className={s.controls}>
      <label className={s.control}>P(page from A is in B): {pA.toFixed(2)}
        <input type="range" min="0.1" max="1" step="0.05" value={pA} aria-label="Overlap probability from A"
          onChange={(event) => setPA(Number(event.target.value))} />
      </label>
      <label className={s.control}>P(page from B is in A): {pB.toFixed(2)}
        <input type="range" min="0.1" max="1" step="0.05" value={pB} aria-label="Overlap probability from B"
          onChange={(event) => setPB(Number(event.target.value))} />
      </label>
    </div>}>
    <div style={{display: 'grid', gap: '0.65rem'}}>
      {[['A sample also in B', pA, 0], ['B sample also in A', pB, 1]].map(([label, probability, index]) =>
        <div key={String(label)} style={{display: 'grid', gridTemplateColumns: '12rem minmax(0, 1fr) 3rem', gap: '0.5rem', alignItems: 'center'}}>
          <span>{label}</span><div style={{height: '1.25rem', background: 'var(--ifm-color-emphasis-200)', borderRadius: '0.25rem'}}>
            <div style={{height: '100%', width: `${Number(probability) * 100}%`, background: seriesColor(Number(index), dark), borderRadius: '0.25rem'}} />
          </div><strong>{Number(probability).toFixed(2)}</strong>
        </div>)}
      <div aria-live="polite">Estimated size ratio |A| / |B| = <strong>{ratio.toFixed(2)}</strong></div>
    </div>
  </VizPanel>;
}
