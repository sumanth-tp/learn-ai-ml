import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const IDS = [1, 2, 4, 5, 11, 31];
type Operation = 'AND' | 'OR' | 'A NOT B';

export default function BooleanPostingsLab() {
  const dark = useDarkViz();
  const [a, setA] = useState<number[]>([1, 2, 4, 11, 31]);
  const [b, setB] = useState<number[]>([1, 2, 4, 5, 31]);
  const [operation, setOperation] = useState<Operation>('AND');
  const result = IDS.filter((id) => operation === 'AND' ? a.includes(id) && b.includes(id)
    : operation === 'OR' ? a.includes(id) || b.includes(id) : a.includes(id) && !b.includes(id));
  const toggle = (list: number[], setList: (value: number[]) => void, id: number) =>
    setList(list.includes(id) ? list.filter((value) => value !== id) : [...list, id].sort((x, y) => x - y));
  const button = (id: number, list: number[], setList: (value: number[]) => void, label: string, color: string) => (
    <button key={id} type="button" onClick={() => toggle(list, setList, id)} aria-pressed={list.includes(id)}
      aria-label={`${label}: document ${id}`} style={{border: `2px solid ${color}`, borderRadius: '0.4rem', padding: '0.25rem 0.5rem',
        background: list.includes(id) ? color : 'transparent', color: list.includes(id) ? '#fff' : 'inherit', cursor: 'pointer'}}>
      {id}
    </button>
  );

  return (
    <VizPanel title="Build a Boolean answer from postings"
      hint="Each row is a sorted postings list. Toggle membership, then compare AND, OR and set difference."
      table={{columns: ['document ID', 'in A', 'in B', 'in result'], rows: IDS.map((id) => [id, a.includes(id) ? 'yes' : 'no', b.includes(id) ? 'yes' : 'no', result.includes(id) ? 'yes' : 'no'])}}
      controls={<label className={s.control}>set operation
        <select className={s.select} value={operation} onChange={(event) => setOperation(event.target.value as Operation)} aria-label="Set operation">
          <option>AND</option><option>OR</option><option>A NOT B</option>
        </select>
      </label>}>
      <div style={{display: 'grid', gap: '0.9rem'}}>
        <div><strong>List A</strong><div style={{display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.3rem'}}>
          {IDS.map((id) => button(id, a, setA, 'List A', seriesColor(0, dark)))}</div></div>
        <div><strong>List B</strong><div style={{display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.3rem'}}>
          {IDS.map((id) => button(id, b, setB, 'List B', seriesColor(1, dark)))}</div></div>
        <div aria-live="polite"><strong>{operation} result:</strong> {result.length ? result.join(', ') : 'no documents'}</div>
      </div>
    </VizPanel>
  );
}
