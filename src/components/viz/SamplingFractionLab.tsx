import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function SamplingFractionLab() {
  const dark = useDarkViz();
  const [sourceThousands, setSourceThousands] = useState(2000);
  const [sample, setSample] = useState(5000);
  const source = sourceThousands * 1000;
  const fraction = sample / source;

  return (
    <VizPanel
      title="How much of the source is sampled?"
      hint="The fraction says how many rows were selected. It does not establish that the selection is representative."
      table={{columns: ['measure', 'value'], rows: [
        ['source rows', source.toLocaleString()],
        ['sample rows', sample.toLocaleString()],
        ['fraction', fraction.toFixed(4)],
        ['percentage', `${(fraction * 100).toFixed(2)}%`],
      ]}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            source rows: {source.toLocaleString()}
            <input type="range" min="100" max="2000" step="100" value={sourceThousands} aria-label="Source rows in thousands"
              onChange={(event) => setSourceThousands(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            sample rows: {sample.toLocaleString()}
            <input type="range" min="1000" max="10000" step="1000" value={sample} aria-label="Sample rows"
              onChange={(event) => setSample(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.7rem'}}>
        <strong>{sample.toLocaleString()} / {source.toLocaleString()} = {fraction.toFixed(4)} ({(fraction * 100).toFixed(2)}%)</strong>
        <div style={{height: '1.5rem', borderRadius: '0.25rem', background: 'var(--ifm-color-emphasis-200)', overflow: 'hidden'}}
          role="img" aria-label={`${(fraction * 100).toFixed(2)} per cent of source rows sampled`}>
          <div style={{height: '100%', width: `${Math.min(100, fraction * 100)}%`, minWidth: '2px', background: seriesColor(0, dark)}} />
        </div>
      </div>
    </VizPanel>
  );
}
