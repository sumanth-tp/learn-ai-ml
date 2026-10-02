import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function ScdHistoryLab() {
  const dark = useDarkViz();
  const [changes, setChanges] = useState(3);
  const [time, setTime] = useState(2);
  const versions = Array.from({length: changes + 1}, (_, index) => index);
  const active = Math.min(time, changes);

  return (
    <VizPanel
      title="One customer, several historical versions"
      hint="Each version is valid from its change time until the next change. The final version has an open end."
      table={{columns: ['version', 'valid from', 'valid to', 'active now'], rows: versions.map((index) => [
        `V${index + 1}`, `t${index}`, index < changes ? `t${index + 1}` : 'open', index === active ? 'yes' : 'no',
      ])}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            address changes: {changes}
            <input type="range" min="0" max="5" value={changes} aria-label="Address changes"
              onChange={(event) => {
                const next = Number(event.target.value);
                setChanges(next);
                setTime((current) => Math.min(current, next));
              }} />
          </label>
          <label className={s.control}>
            time to inspect: t{time}
            <input type="range" min="0" max={changes} value={time} aria-label="Historical time to inspect"
              onChange={(event) => setTime(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.75rem'}}>
        <strong>{changes} changes produce {versions.length} rows; V{active + 1} is valid at t{time}</strong>
        <div style={{display: 'flex', flexWrap: 'wrap', gap: '0.5rem'}}>
          {versions.map((index) => (
            <span key={index} style={{padding: '0.45rem 0.65rem', borderRadius: '0.35rem', background: index === active ? seriesColor(0, dark) : 'var(--ifm-color-emphasis-200)', color: index === active ? '#fff' : 'inherit'}}>
              V{index + 1}: [t{index}, {index < changes ? `t${index + 1})` : 'open)'}
            </span>
          ))}
        </div>
      </div>
    </VizPanel>
  );
}
