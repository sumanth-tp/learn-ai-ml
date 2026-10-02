import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function KAnonymityLab() {
  const dark = useDarkViz();
  const [firstGroup, setFirstGroup] = useState(4);
  const groups = [firstGroup, 5, 7];
  const k = Math.min(...groups);

  return (
    <VizPanel
      title="Equivalence groups and k-anonymity"
      hint="One divided by k is only a uniform-guess illustration within a known group, not a general re-identification bound."
      table={{columns: ['age and ZIP group', 'records'], rows: groups.map((size, index) => [
        `group ${index + 1}`, String(size),
      ])}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            group 1 size: {firstGroup}
            <input type="range" min="1" max="8" value={firstGroup} aria-label="First quasi-identifier group size"
              onChange={(event) => setFirstGroup(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.75rem'}}>
        <strong>Smallest group = {k}; therefore k = {k}. Uniform-guess illustration: 1/{k} = {(1 / k).toFixed(2)}.</strong>
        {groups.map((size, index) => (
          <div key={index} style={{display: 'grid', gridTemplateColumns: '5rem minmax(0, 1fr) 3rem', gap: '0.5rem', alignItems: 'center'}}>
            <span>group {index + 1}</span>
            <div style={{height: '1rem', borderRadius: '0.25rem', background: 'var(--ifm-color-emphasis-200)', overflow: 'hidden'}}>
              <div style={{height: '100%', width: `${size / 8 * 100}%`, background: seriesColor(index, dark)}} />
            </div>
            <output>{size}</output>
          </div>
        ))}
        <span>{k === 1 ? 'A unique quasi-identifier group needs suppression or generalisation before a k ≥ 2 release.' : 'Sensitive values or outside knowledge may still disclose information about a member.'}</span>
      </div>
    </VizPanel>
  );
}
