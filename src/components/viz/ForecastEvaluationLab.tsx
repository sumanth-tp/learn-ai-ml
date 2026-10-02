import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const scale = 5 / 3;

export default function ForecastEvaluationLab() {
  const dark = useDarkViz();
  const [error, setError] = useState(1);
  const [centre, setCentre] = useState(13);
  const [radius, setRadius] = useState(2);
  const mase = error / scale;
  const lower = centre - radius;
  const upper = centre + radius;
  const rows: [string, string][] = [
    ['Training absolute differences', '2, 1, 2'], ['Training naive scale', scale.toFixed(3)],
    ['Test MAE', error.toFixed(2)], ['MASE', mase.toFixed(3)],
    ['Illustrative centre', centre.toFixed(1)], ['Illustrative radius', radius.toFixed(1)],
    ['Illustrative interval', `[${lower.toFixed(1)}, ${upper.toFixed(1)}]`],
  ];

  return <VizPanel title="Scale error and display uncertainty"
    hint="The interval is an illustration of centre ± radius, with no calibrated coverage claim. Validate interval coverage and width on held-out forecast origins."
    table={{columns: ['Measure', 'Value'], rows}}
    controls={<div className={s.controls}>{([
      ['Test absolute error', error, setError, 0, 4, 0.1],
      ['Forecast centre', centre, setCentre, 8, 18, 0.5],
      ['Illustrative radius', radius, setRadius, 0, 4, 0.5],
    ] as const).map(([label, value, setter, min, max, step]) => <label className={s.control} key={label}>{label}: {value.toFixed(1)}
      <input type="range" min={min} max={max} step={step} value={value} aria-label={label} onChange={event => setter(Number(event.target.value))} />
    </label>)}</div>}>
    <div style={{display: 'grid', gap: '0.7rem'}}>
      <div>Test MAE {error.toFixed(2)} ÷ training naive scale {scale.toFixed(3)} = MASE {mase.toFixed(3)}</div>
      <div style={{height: '1.2rem', background: dark ? '#293142' : '#e7eaf0', borderRadius: '0.25rem', overflow: 'hidden'}}>
        <div style={{height: '100%', width: `${Math.min(mase / 2, 1) * 100}%`, background: seriesColor(0, dark)}} />
      </div>
      <output>Illustrative interval [{lower.toFixed(1)}, {upper.toFixed(1)}]; coverage is not established by choosing a radius.</output>
    </div>
  </VizPanel>;
}
