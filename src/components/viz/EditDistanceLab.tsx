import {useMemo, useState} from 'react';

import {sequentialColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

function matrix(a: string, b: string) {
  const rows = Array.from({length: a.length + 1}, () => Array<number>(b.length + 1).fill(0));
  for (let i = 0; i <= a.length; i += 1) rows[i][0] = i;
  for (let j = 0; j <= b.length; j += 1) rows[0][j] = j;
  for (let i = 1; i <= a.length; i += 1) {
    for (let j = 1; j <= b.length; j += 1) {
      rows[i][j] = Math.min(rows[i - 1][j] + 1, rows[i][j - 1] + 1,
        rows[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
  }
  return rows;
}

export default function EditDistanceLab() {
  const dark = useDarkViz();
  const [from, setFrom] = useState('cat');
  const [to, setTo] = useState('cart');
  const rows = useMemo(() => matrix(from, to), [from, to]);
  const distance = rows[from.length][to.length];
  const update = (value: string, setValue: (value: string) => void) => setValue(value.toLowerCase().replace(/[^a-z]/g, '').slice(0, 8));

  return (
    <VizPanel title="Edit distance, one cell at a time"
      hint="Each cell takes the cheapest insertion, deletion or substitution. The bottom-right cell is the distance."
      table={{columns: ['prefix', 'empty', ...to.split('')], rows: rows.map((row, i) => [i === 0 ? 'empty' : from[i - 1], ...row])}}
      controls={<div className={s.controls}>
        <label className={s.control}>from
          <input value={from} onChange={(event) => update(event.target.value, setFrom)} aria-label="Original term" maxLength={8} />
        </label>
        <label className={s.control}>to
          <input value={to} onChange={(event) => update(event.target.value, setTo)} aria-label="Candidate term" maxLength={8} />
        </label>
      </div>}>
      <div style={{overflowX: 'auto'}}>
        <div style={{display: 'grid', gridTemplateColumns: `repeat(${to.length + 2}, minmax(2rem, 2.5rem))`, gap: '0.2rem', width: 'max-content'}}>
          {['', '∅', ...to].map((letter, i) => <strong key={`head-${i}`} style={{textAlign: 'center'}}>{letter}</strong>)}
          {rows.map((row, i) => [
            <strong key={`label-${i}`} style={{textAlign: 'center'}}>{i === 0 ? '∅' : from[i - 1]}</strong>,
            ...row.map((value, j) => <span key={`${i}-${j}`} style={{textAlign: 'center', padding: '0.35rem', borderRadius: '0.25rem',
              background: sequentialColor(1 - value / Math.max(from.length, to.length, 1), dark), color: dark ? '#fff' : '#17243b'}}>{value}</span>),
          ])}
        </div>
      </div>
      <p aria-live="polite" style={{marginTop: '0.8rem', marginBottom: 0}}>distance: <strong>{distance}</strong></p>
    </VizPanel>
  );
}
