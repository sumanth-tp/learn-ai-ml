import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const values = [10, 12, 11, 13, 10, 12];

export default function LagWindowLab() {
  const dark = useDarkViz();
  const [target, setTarget] = useState(3);
  const [windowSize, setWindowSize] = useState(3);
  const prior = values.slice(target - windowSize, target);
  const mean = prior.reduce((sum, value) => sum + value, 0) / prior.length;
  const rows: [string, string, string][] = values.map((value, index) => [
    String(index), String(value), index === target ? 'Target' : index >= target - windowSize && index < target ? 'Feature window' : 'Outside window',
  ]);
  rows.push(['Lag 1', String(values[target - 1]), 'Feature']);
  rows.push(['Lag 2', String(values[target - 2]), 'Feature']);
  rows.push([`Prior ${windowSize} mean`, mean.toFixed(3), 'Feature']);

  return <VizPanel title="Build a row from the past"
    hint="The highlighted feature window ends before the target. At prediction time, any exogenous feature must also be known by the forecast origin."
    table={{columns: ['Time or feature', 'Value', 'Role'], rows}}
    controls={<div className={s.controls}>
      <label className={s.control}>Target index: {target}
        <input type="range" min="3" max="5" step="1" value={target} aria-label="Forecast target index" onChange={event => setTarget(Number(event.target.value))} />
      </label>
      <label className={s.control}>Prior window size: {windowSize}
        <input type="range" min="2" max="3" step="1" value={windowSize} aria-label="Prior window size" onChange={event => setWindowSize(Number(event.target.value))} />
      </label>
    </div>}>
    <div style={{display: 'grid', gap: '0.6rem'}}>
      <div style={{display: 'grid', gridTemplateColumns: 'repeat(6, minmax(0, 1fr))', gap: '0.25rem'}}>
        {values.map((value, index) => <div key={index} style={{textAlign: 'center', padding: '0.5rem 0', background: seriesColor(index === target ? 1 : index >= target - windowSize && index < target ? 0 : 2, dark), color: '#fff', borderRadius: '0.25rem'}}>
          <small>t{index}</small><br />{value}
        </div>)}
      </div>
      <output>For target t{target}={values[target]}: lag 1={values[target - 1]}, lag 2={values[target - 2]}, prior-{windowSize} mean={mean.toFixed(3)}.</output>
    </div>
  </VizPanel>;
}
