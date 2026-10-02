import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const runs = [
  {id: 1, f1: 0.71, latency: 35},
  {id: 2, f1: 0.76, latency: 80},
  {id: 3, f1: 0.74, latency: 45},
];

export default function ExperimentChoiceLab() {
  const dark = useDarkViz();
  const [limit, setLimit] = useState(100);
  const eligible = runs.filter((run) => run.latency <= limit);
  const selected = eligible.reduce<(typeof runs)[number] | null>((best, run) =>
    best === null || run.f1 > best.f1 ? run : best, null);

  return (
    <VizPanel
      title="Choose a tracked run under a serving constraint"
      hint="This ranking assumes comparable, leakage-safe evaluation data. Logging a high metric does not establish that assumption."
      table={{columns: ['run', 'F1', 'latency ms', 'eligible', 'selected'], rows: runs.map((run) => [
        String(run.id), run.f1.toFixed(2), String(run.latency), run.latency <= limit ? 'yes' : 'no', selected?.id === run.id ? 'yes' : 'no',
      ])}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            maximum latency: {limit} ms
            <input type="range" min="30" max="100" value={limit} aria-label="Maximum inference latency"
              onChange={(event) => setLimit(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.65rem'}}>
        <strong>{selected ? `Run ${selected.id} selected: F1 ${selected.f1.toFixed(2)}` : 'No run meets the latency limit'}</strong>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(9rem, 1fr))', gap: '0.65rem'}}>
          {runs.map((run, index) => (
            <div key={run.id} style={{padding: '0.7rem', borderRadius: '0.35rem', background: selected?.id === run.id ? seriesColor(index, dark) : 'var(--ifm-color-emphasis-200)', color: selected?.id === run.id ? '#fff' : 'inherit'}}>
              <strong>Run {run.id}</strong><br />F1 {run.f1.toFixed(2)}<br />{run.latency} ms<br />{run.latency <= limit ? 'eligible' : 'too slow'}
            </div>
          ))}
        </div>
      </div>
    </VizPanel>
  );
}
