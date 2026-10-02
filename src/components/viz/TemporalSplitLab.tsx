import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const values = [10, 12, 11, 13, 10, 12, 11, 13];

export default function TemporalSplitLab() {
  const dark = useDarkViz();
  const [cutoff, setCutoff] = useState(6);
  const predictions = values.slice(cutoff).map((_, offset) => values[cutoff + offset - 4]);
  const rows: [string, string, string, string][] = values.map((value, index) => [
    String(index), String(value), index < cutoff ? 'Training' : 'Held out',
    index >= cutoff ? String(values[index - 4]) : 'Not scored',
  ]);

  return <VizPanel title="Keep future observations out of training"
    hint="A seasonal baseline uses an earlier observed season. In a real evaluation, fit every transformation using only data available at each forecast origin."
    table={{columns: ['Time', 'Observed', 'Split', 'Seasonal baseline'], rows}}
    controls={<label className={s.control}>Training cutoff: {cutoff}
      <input type="range" min="4" max="7" step="1" value={cutoff} aria-label="Training cutoff" onChange={event => setCutoff(Number(event.target.value))} />
    </label>}>
    <div style={{display: 'grid', gap: '0.6rem'}}>
      <div style={{display: 'grid', gridTemplateColumns: 'repeat(8, minmax(0, 1fr))', gap: '0.25rem'}}>
        {values.map((value, index) => <div key={index} style={{textAlign: 'center', padding: '0.5rem 0', background: seriesColor(index < cutoff ? 0 : 1, dark), color: '#fff', borderRadius: '0.25rem'}}>
          <small>t{index}</small><br />{value}
        </div>)}
      </div>
      <output>Train t0–t{cutoff - 1}; hold out {values.slice(cutoff).join(', ')}. Seasonal period-four predictions: {predictions.join(', ')}.</output>
    </div>
  </VizPanel>;
}
