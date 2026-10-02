import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function VisualWordsLab() {
  const dark = useDarkViz();
  const [counts, setCounts] = useState([4, 1, 3]);
  const total = counts.reduce((sum, count) => sum + count, 0);
  const frequencies = counts.map(count => total ? count / total : null);
  const rows: [string, string, string][] = counts.map((count, index) => [
    `Visual word ${index + 1}`, String(count), frequencies[index] === null ? 'undefined' : frequencies[index]!.toFixed(3),
  ]);
  const update = (index: number, value: number) => setCounts(current => current.map((count, position) => position === index ? value : count));

  return (
    <VizPanel title="Build a visual-word histogram"
      hint="Moving the same local features to different image positions keeps this whole-image histogram unchanged. A spatial pyramid stores separate regional histograms."
      table={{columns: ['Codebook entry', 'Count', 'Frequency'], rows: [...rows, ['Total', String(total), total ? '1.000' : 'undefined']]}}
      controls={<div className={s.controls}>{counts.map((count, index) =>
        <label className={s.control} key={index}>Word {index + 1} count: {count}
          <input type="range" min="0" max="12" step="1" value={count} aria-label={`Visual word ${index + 1} count`}
            onChange={event => update(index, Number(event.target.value))} />
        </label>)}</div>}>
      <div style={{display: 'grid', gap: '0.6rem'}}>
        {frequencies.map((value, index) => <div key={index}>
          <div>Word {index + 1}: {value === null ? 'undefined' : value.toFixed(3)}</div>
          <div style={{height: '1rem', background: dark ? '#293142' : '#e7eaf0', borderRadius: '0.25rem', overflow: 'hidden'}}>
            <div style={{height: '100%', width: `${(value ?? 0) * 100}%`, background: seriesColor(index, dark)}} />
          </div>
        </div>)}
        <output>Counts [{counts.join(', ')}] form a {counts.length}-dimensional histogram over {total} local features.</output>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0.5rem'}}>
          <div>Layout A: words 1 and 2 left; word 3 right</div>
          <div>Layout B: word 3 left; words 1 and 2 right</div>
        </div>
      </div>
    </VizPanel>
  );
}
