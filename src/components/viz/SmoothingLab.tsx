import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const observations = [10, 12, 11, 13];

export default function SmoothingLab() {
  const dark = useDarkViz();
  const [alpha, setAlpha] = useState(0.5);
  let level = observations[0];
  const states = observations.map((value, index) => {
    const prior = level;
    level = index === 0 ? value : alpha * value + (1 - alpha) * level;
    return {value, prior, level};
  });
  const rows: [string, string, string, string][] = states.map((item, index) => [
    String(index), String(item.value), item.prior.toFixed(2), item.level.toFixed(2),
  ]);

  return <VizPanel title="Update one smoothed level"
    hint="Simple exponential smoothing models one level. It does not explicitly model a trend or seasonal pattern."
    table={{columns: ['Time', 'Observed', 'Prior level', 'Updated level'], rows}}
    controls={<label className={s.control}>Smoothing alpha: {alpha.toFixed(2)}
      <input type="range" min="0" max="1" step="0.05" value={alpha} aria-label="Smoothing alpha" onChange={event => setAlpha(Number(event.target.value))} />
    </label>}>
    <div style={{display: 'grid', gap: '0.5rem'}}>
      {states.map((item, index) => <div key={index} style={{display: 'grid', gridTemplateColumns: '4rem 1fr', gap: '0.5rem', alignItems: 'center'}}>
        <span>t{index}: {item.level.toFixed(2)}</span>
        <div style={{height: '0.9rem', background: dark ? '#293142' : '#e7eaf0', borderRadius: '0.2rem', overflow: 'hidden'}}>
          <div style={{height: '100%', width: `${item.level / 15 * 100}%`, background: seriesColor(index, dark)}} />
        </div>
      </div>)}
      <output>Next one-step forecast: {level.toFixed(2)}.</output>
    </div>
  </VizPanel>;
}
