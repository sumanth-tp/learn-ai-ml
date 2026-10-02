import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const observations = [
  {day: 1, value: 4},
  {day: 3, value: 8},
  {day: 5, value: 12},
];

export default function FeatureCutoffLab() {
  const dark = useDarkViz();
  const [day, setDay] = useState(4);
  const [ttl, setTtl] = useState(3);
  const eligible = observations.filter((item) => item.day <= day && day - item.day <= ttl);
  const selected = eligible.at(-1);
  const latest = observations.at(-1);

  return (
    <VizPanel
      title="Which feature existed at prediction time?"
      hint="Historical retrieval uses the latest eligible feature at or before the cutoff. The current latest value may be from the future."
      table={{columns: ['feature day', 'value', 'eligible at cutoff', 'selected'], rows: observations.map((item) => [
        `day ${item.day}`, String(item.value), item.day <= day && day - item.day <= ttl ? 'yes' : 'no', item.day === selected?.day ? 'yes' : 'no',
      ])}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            prediction day: {day}
            <input type="range" min="1" max="6" value={day} aria-label="Prediction day"
              onChange={(event) => setDay(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            feature age limit: {ttl} days
            <input type="range" min="1" max="5" value={ttl} aria-label="Feature age limit"
              onChange={(event) => setTtl(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.75rem'}}>
        <strong>At day {day}: as-of value {selected?.value ?? 'none'}; current latest value {latest?.value}</strong>
        <div style={{display: 'flex', flexWrap: 'wrap', gap: '0.5rem'}}>
          {observations.map((item) => (
            <span key={item.day} style={{padding: '0.45rem 0.65rem', borderRadius: '0.35rem', background: item.day === selected?.day ? seriesColor(0, dark) : 'var(--ifm-color-emphasis-200)', color: item.day === selected?.day ? '#fff' : 'inherit'}}>
              day {item.day}: {item.value}
            </span>
          ))}
        </div>
      </div>
    </VizPanel>
  );
}
