import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function HarrisResponseLab() {
  const dark = useDarkViz();
  const [first, setFirst] = useState(1);
  const [second, setSecond] = useState(1);
  const [k, setK] = useState(0.04);
  const determinant = first * second;
  const trace = first + second;
  const response = determinant - k * trace ** 2;
  const shape = first < 0.1 && second < 0.1 ? 'flat' : response > 0.05 ? 'corner-like' : 'edge-like or weak';
  const rows: [string, string][] = [
    ['First eigenvalue', first.toFixed(2)],
    ['Second eigenvalue', second.toFixed(2)],
    ['Determinant', determinant.toFixed(3)],
    ['Trace', trace.toFixed(3)],
    ['k', k.toFixed(2)],
    ['Response', response.toFixed(3)],
  ];

  return (
    <VizPanel
      title="Harris response from two directions"
      hint="The qualitative label uses a teaching threshold; actual corner selection compares local responses and image-dependent scales."
      table={{columns: ['Measure', 'Value'], rows}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            First eigenvalue: {first.toFixed(1)}
            <input type="range" min="0" max="2" step="0.1" value={first} aria-label="First structure eigenvalue"
              onChange={event => setFirst(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Second eigenvalue: {second.toFixed(1)}
            <input type="range" min="0" max="2" step="0.1" value={second} aria-label="Second structure eigenvalue"
              onChange={event => setSecond(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Harris k: {k.toFixed(2)}
            <input type="range" min="0.02" max="0.10" step="0.01" value={k} aria-label="Harris k"
              onChange={event => setK(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.65rem'}}>
        {([['First direction', first, 0], ['Second direction', second, 1]] as const).map(([label, value, colour]) => (
          <div key={label}>
            <div>{label}: {value.toFixed(1)}</div>
            <div style={{height: '1.3rem', borderRadius: '0.25rem', background: dark ? '#293142' : '#e7eaf0', overflow: 'hidden'}}>
              <div style={{height: '100%', width: `${value * 50}%`, background: seriesColor(colour, dark)}} />
            </div>
          </div>
        ))}
        <output>R = {response.toFixed(3)} · {shape}</output>
      </div>
    </VizPanel>
  );
}
