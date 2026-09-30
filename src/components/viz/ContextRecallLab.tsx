import {useState} from 'react';
import VizPanel from './VizPanel';
import {Score, Slider} from './CourseLabShared';
import c from './CourseLab.module.css';

const CLAIMS = [
  {claim: 'Urban minimum balance: ₹10,000', rank: 1},
  {claim: 'Fee: ₹350 + taxes', rank: 2},
  {claim: 'Rural minimum balance: ₹2,500', rank: 5},
  {claim: 'Semi-urban minimum balance: ₹5,000', rank: 4},
];
export default function ContextRecallLab() {
  const [k, setK] = useState(4);
  const supported = CLAIMS.filter(claim => claim.rank <= k).length;
  const score = supported / CLAIMS.length;
  const band = score < 0.4 ? 'Low' : score <= 0.7 ? 'Medium' : 'High';
  return <VizPanel title="Context recall: how much of the reference was retrieved?"
    hint="Rank 3 is the KYC chunk, which supports none of these reference claims. Increasing k from 2 to 3 therefore adds noise without improving recall."
    controls={<Slider label="Retrieved chunks k" min={1} max={5} value={k} onChange={setK} />}
    table={{columns: ['Reference claim', 'Supporting rank', 'Retrieved?'], rows: CLAIMS.map(claim => [claim.claim, claim.rank, claim.rank <= k ? 'Yes' : 'No'])}}>
    <Score value={score} label="Context recall" detail={`${band} · ${supported} / 4 reference claims supported`} />
    <div className={c.stack}>{CLAIMS.map(claim => <div key={claim.claim} className={c.chip}>
      <strong>{claim.claim}</strong><br />Rank {claim.rank}: {claim.rank <= k ? 'Supported by retrieved context' : 'Missing from the top-k context'}
    </div>)}</div>
  </VizPanel>;
}
