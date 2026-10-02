import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function FreshnessBudgetLab() {
  const dark = useDarkViz();
  const [age, setAge] = useState(90);
  const [limit, setLimit] = useState(60);
  const breach = age > limit;

  return (
    <VizPanel
      title="Freshness promise and data age"
      hint="This clock measures time since the last approved load. Event-time freshness needs a source watermark as well."
      table={{columns: ['measure', 'minutes'], rows: [
        ['age since approved load', String(age)],
        ['maximum allowed age', String(limit)],
        ['amount over limit', String(Math.max(0, age - limit))],
      ]}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            data age: {age} min
            <input type="range" min="0" max="180" step="5" value={age} aria-label="Data age in minutes"
              onChange={(event) => setAge(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            freshness limit: {limit} min
            <input type="range" min="15" max="120" step="5" value={limit} aria-label="Freshness limit in minutes"
              onChange={(event) => setLimit(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.75rem'}}>
        <strong>{breach ? `Breach: ${age} > ${limit} minutes, ${age - limit} minutes over the limit` : `Within limit: ${age} ≤ ${limit} minutes`}</strong>
        <div style={{height: '1.3rem', borderRadius: '0.3rem', background: 'var(--ifm-color-emphasis-200)', overflow: 'hidden'}}>
          <div style={{height: '100%', width: `${Math.min(100, age / 180 * 100)}%`, background: seriesColor(breach ? 3 : 1, dark)}} />
        </div>
      </div>
    </VizPanel>
  );
}
