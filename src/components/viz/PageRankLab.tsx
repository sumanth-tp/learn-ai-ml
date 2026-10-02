import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const INITIAL = [1 / 3, 1 / 3, 1 / 3];
const LABELS = ['A', 'B', 'P'];

export default function PageRankLab() {
  const dark = useDarkViz();
  const [damping, setDamping] = useState(0.85);
  const [ranks, setRanks] = useState(INITIAL);
  const [step, setStep] = useState(0);
  const incoming = [ranks[2], 0, ranks[0] + ranks[1]];
  const next = incoming.map((value) => (1 - damping) / 3 + damping * value);
  const advance = () => { setRanks(next); setStep((current) => current + 1); };
  const reset = () => { setRanks(INITIAL); setStep(0); };
  const table = {columns: ['page', 'current rank', 'incoming share', 'next rank'], rows: LABELS.map((label, index) => [
    label, ranks[index].toFixed(3), incoming[index].toFixed(3), next[index].toFixed(3),
  ])};

  return <VizPanel title="Iterate a three-page link graph"
    hint="A and B each link to P; P links to A. Each step redistributes rank, then adds teleport probability."
    table={table}
    controls={<div className={s.controls}>
      <label className={s.control}>follow-link probability d: {damping.toFixed(2)}
        <input type="range" min="0" max="1" step="0.05" value={damping} aria-label="PageRank damping"
          onChange={(event) => setDamping(Number(event.target.value))} />
      </label>
      <button type="button" onClick={advance}>Run one PageRank step</button>
      <button type="button" onClick={reset}>Reset ranks</button>
    </div>}>
    <div style={{display: 'grid', gap: '0.65rem'}}>
      <div>A → P, B → P, P → A</div>
      {LABELS.map((label, index) => <div key={label} style={{display: 'grid', gridTemplateColumns: '2rem minmax(0, 1fr) 4rem', gap: '0.5rem', alignItems: 'center'}}>
        <strong>{label}</strong><div style={{height: '1.25rem', background: 'var(--ifm-color-emphasis-200)', borderRadius: '0.25rem'}}>
          <div style={{height: '100%', width: `${ranks[index] * 100}%`, background: seriesColor(index, dark), borderRadius: '0.25rem'}} />
        </div><span>{ranks[index].toFixed(3)}</span>
      </div>)}
      <div aria-live="polite">step {step}; P rank {ranks[2].toFixed(3)}; next P rank {next[2].toFixed(3)}</div>
    </div>
  </VizPanel>;
}
