import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function ProfilingDriftLab() {
  const dark = useDarkViz();
  const [nulls, setNulls] = useState(1500);
  const [limit, setLimit] = useState(5);
  const [high, setHigh] = useState(50);
  const nullRate = nulls / 100;
  const currentHigh = high / 100;
  const currentLow = 1 - currentHigh;
  const psi = (currentHigh - 0.5) * Math.log(currentHigh / 0.5)
    + (currentLow - 0.5) * Math.log(currentLow / 0.5);
  const passes = nullRate <= limit;

  return (
    <VizPanel
      title="Profile missingness and distribution change"
      hint="The two-bin PSI measures a shift against a 50/50 reference. It cannot identify the cause or justify retraining by itself."
      table={{columns: ['measure', 'value'], rows: [
        ['rows', '10,000'],
        ['nulls', nulls.toLocaleString()],
        ['null rate', `${nullRate.toFixed(1)}%`],
        ['allowed null rate', `${limit}%`],
        ['null rule', passes ? 'pass' : 'fail'],
        ['reference high bin', '50%'],
        ['current high bin', `${high}%`],
        ['two-bin PSI', psi.toFixed(3)],
      ]}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            nulls among 10,000: {nulls.toLocaleString()}
            <input type="range" min="0" max="2000" step="100" value={nulls} aria-label="Null values among ten thousand"
              onChange={(event) => setNulls(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            allowed null rate: {limit}%
            <input type="range" min="1" max="20" value={limit} aria-label="Allowed null rate"
              onChange={(event) => setLimit(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            current high-bin share: {high}%
            <input type="range" min="10" max="90" value={high} aria-label="Current high-bin percentage"
              onChange={(event) => setHigh(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.75rem'}}>
        <strong>Null rate {nullRate.toFixed(1)}%: rule {passes ? 'passes' : 'fails'} at {limit}%</strong>
        <div style={{height: '1.5rem', borderRadius: '0.25rem', background: seriesColor(2, dark), overflow: 'hidden'}}
          role="img" aria-label={`${nullRate.toFixed(1)} per cent missing values`}>
          <div style={{height: '100%', width: `${100 - nullRate}%`, background: seriesColor(0, dark)}} />
        </div>
        <output>Two-bin PSI against 50/50 reference: {psi.toFixed(3)}</output>
      </div>
    </VizPanel>
  );
}
