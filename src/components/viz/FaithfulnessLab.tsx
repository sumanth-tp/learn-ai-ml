import {useState} from 'react';
import VizPanel, {vizStyles as s} from './VizPanel';
import {Score, Slider} from './CourseLabShared';
import c from './CourseLab.module.css';

const CLAIMS = [
  ['Urban minimum balance: ₹10,000', 'Supported by chunk 1'],
  ['Fee: ₹350 + GST', 'Supported by chunk 1'],
  ['Online transfers have no extra charge', 'No supporting retrieved chunk'],
  ['Rural minimum balance: ₹2,500', 'Supported by chunk 2'],
];
export default function FaithfulnessLab() {
  const [grounded, setGrounded] = useState([true, true, false, true]);
  const [threshold, setThreshold] = useState(0.8);
  const count = grounded.filter(Boolean).length;
  const score = count / CLAIMS.length;
  return <VizPanel title="Faithfulness: which claims are grounded?"
    hint="Click a claim to change its simulated judgement. Faithfulness checks support in retrieved context; it does not check against a reference answer."
    controls={<><Slider label="Pass threshold" value={threshold} min={0.5} max={1} step={0.05} onChange={setThreshold} />
      <button className={s.button} type="button" onClick={() => {setGrounded([true, true, false, true]); setThreshold(0.8);}}>Reset</button></>}
    table={{columns: ['Claim', 'Original evidence', 'Simulated judgement'], rows: CLAIMS.map(([claim, evidence], i) => [claim, evidence, grounded[i] ? 'Grounded' : 'Hallucinated'])}}>
    <Score value={score} label="Faithfulness" detail={`${score >= threshold ? 'Pass' : 'Fail'} · ${count} / 4 grounded`} />
    <div className={c.stack}>{CLAIMS.map(([claim, evidence], i) => <button key={claim} type="button"
      className={`${s.button} ${c.claim}`} aria-pressed={grounded[i]}
      onClick={() => setGrounded(values => values.map((v, j) => i === j ? !v : v))}>
      <strong>{claim}</strong><span>{grounded[i] ? 'Grounded' : 'Hallucinated'}</span><small>{evidence}</small>
    </button>)}</div>
  </VizPanel>;
}
