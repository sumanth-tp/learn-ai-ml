import {useState} from 'react';
import VizPanel, {vizStyles as s} from './VizPanel';
import {Score, Slider} from './CourseLabShared';
import c from './CourseLab.module.css';

const CLAIMS = ['Urban balance: ₹10,000', 'Penalty: ₹400', 'Free internet banking', 'Interest: 3.5%'];
const MISSED = ['Fee: ₹350 + taxes', 'Free passbook'];
export default function AnswerCorrectnessLab() {
  const [truePositives, setTruePositives] = useState([true, false, false, true]);
  const [missing, setMissing] = useState([true, true]);
  const [similarity, setSimilarity] = useState(0.72);
  const [weight, setWeight] = useState(0.75);
  const tp = truePositives.filter(Boolean).length;
  const fp = CLAIMS.length - tp;
  const fn = missing.filter(Boolean).length;
  const f1 = tp / (tp + 0.5 * (fp + fn));
  const score = weight * f1 + (1 - weight) * similarity;
  return <VizPanel title="Answer correctness: facts and semantic similarity"
    hint="Toggle the simulated TP/FP judgements and whether each omitted reference claim counts as FN. This changes labels, not the answer text. F1 = TP / (TP + ½(FP + FN)); score = w₁ × F1 + (1 − w₁) × similarity."
    controls={<><Slider label="Semantic similarity" value={similarity} min={0} max={1} step={0.01} onChange={setSimilarity} />
      <Slider label="Factual weight w₁" value={weight} min={0} max={1} step={0.05} onChange={setWeight} />
      <button type="button" className={s.button} onClick={() => {setTruePositives([true, false, false, true]); setMissing([true, true]); setSimilarity(0.72); setWeight(0.75);}}>Reset</button></>}
    table={{columns: ['Quantity', 'Value'], rows: [['True positives', tp], ['False positives', fp], ['False negatives', fn], ['Factual F1', f1.toFixed(4)], ['Semantic similarity', similarity.toFixed(2)], ['Factual weight', weight.toFixed(2)], ['Semantic weight', (1 - weight).toFixed(2)], ['Answer correctness', score.toFixed(4)]]}}>
    <Score value={score} label="Answer correctness" detail={`F1 ${f1.toFixed(3)} · TP ${tp} / FP ${fp} / FN ${fn}`} />
    <p className={c.note}>Response claims</p>
    <div className={c.stack}>{CLAIMS.map((claim, i) => <button key={claim} type="button" className={`${s.button} ${c.claim}`}
      aria-pressed={truePositives[i]} onClick={() => setTruePositives(values => values.map((v, j) => i === j ? !v : v))}>
      {claim}<small>{truePositives[i] ? 'TP: matches reference' : 'FP: incorrect or unsupported'}</small>
    </button>)}</div>
    <p className={c.note}>Reference omissions</p>
    <div className={c.stack}>{MISSED.map((claim, i) => <button key={claim} type="button" className={`${s.button} ${c.claim}`}
      aria-pressed={missing[i]} onClick={() => setMissing(values => values.map((v, j) => i === j ? !v : v))}>
      {claim}<small>{missing[i] ? 'FN: missed reference claim' : 'Excluded from the simulated FN count'}</small>
    </button>)}</div>
  </VizPanel>;
}
