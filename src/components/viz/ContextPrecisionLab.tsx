import {useState} from 'react';
import VizPanel, {vizStyles as s} from './VizPanel';
import {Score} from './CourseLabShared';
import c from './CourseLab.module.css';

const INITIAL = [
  {id: 'urban', text: 'Urban minimum balance: ₹10,000', relevant: true},
  {id: 'fee', text: 'Fee: ₹350 + taxes', relevant: true},
  {id: 'kyc', text: 'KYC every 8 years', relevant: false},
  {id: 'semi', text: 'Semi-urban minimum balance: ₹5,000', relevant: true},
  {id: 'rural', text: 'Rural minimum balance: ₹2,500', relevant: true},
];
export default function ContextPrecisionLab() {
  const [chunks, setChunks] = useState(INITIAL);
  let relevant = 0;
  const precisions = chunks.map((chunk, i) => {if (chunk.relevant) relevant++; return relevant / (i + 1);});
  const score = relevant ? chunks.reduce((sum, chunk, i) => sum + (chunk.relevant ? precisions[i] : 0), 0) / relevant : 0;
  function move(i: number, offset: number) {
    setChunks(values => {const next = [...values]; [next[i], next[i + offset]] = [next[i + offset], next[i]]; return next;});
  }
  return <VizPanel title="Context precision: move the noisy chunk"
    hint="Move KYC to rank 1 to see the penalty for early noise. The score averages P@k at relevant ranks only. With no relevant chunks, this exercise reports 0."
    controls={<button type="button" className={s.button} onClick={() => setChunks(INITIAL)}>Reset ranking</button>}
    table={{columns: ['Rank', 'Chunk', 'Relevant?', 'P@k', 'Included in mean?'], rows: chunks.map((chunk, i) => [i + 1, chunk.text, chunk.relevant ? 'Yes' : 'No', precisions[i].toFixed(4), chunk.relevant ? 'Yes' : 'No'])}}>
    <Score value={score} label="Context precision" detail={`${score >= 0.7 ? 'Pass' : 'Fail'} · threshold 0.70`} />
    <div className={c.stack}>{chunks.map((chunk, i) => <div className={c.row} key={chunk.id}>
      <strong className={c.rank}>#{i + 1}</strong>
      <button type="button" className={`${s.button} ${c.claim}`} aria-pressed={chunk.relevant}
        onClick={() => setChunks(values => values.map(v => v.id === chunk.id ? {...v, relevant: !v.relevant} : v))}>
        {chunk.text}<small>{chunk.relevant ? 'Relevant' : 'Noise'} · P@{i + 1} = {precisions[i].toFixed(4)}</small>
      </button>
      <button type="button" className={s.button} disabled={i === 0} aria-label={`Move ${chunk.text} up`} onClick={() => move(i, -1)}>↑</button>
      <button type="button" className={s.button} disabled={i === chunks.length - 1} aria-label={`Move ${chunk.text} down`} onClick={() => move(i, 1)}>↓</button>
    </div>)}</div>
    <table className={c.table}><thead><tr><th>Rank k</th><th>P@k</th><th>Contribution</th></tr></thead><tbody>
      {chunks.map((chunk, i) => <tr key={chunk.id}><td>{i + 1}</td><td>{precisions[i].toFixed(4)}</td><td>{chunk.relevant ? 'Included' : 'Excluded: noise'}</td></tr>)}
    </tbody></table>
  </VizPanel>;
}
