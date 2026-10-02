import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const RANKINGS = {
  BM25: [0, 4, 3, 1, 2, 5],
  'dense toy vectors': [3, 2, 0, 1, 4, 5],
  'hybrid RRF': [0, 3, 2, 4, 1, 5],
  reranked: [0, 2, 3, 4, 1, 5],
};
type Stage = keyof typeof RANKINGS;
const RELEVANT = new Set([0, 2]);

export default function RetrievalPipelineLab() {
  const dark = useDarkViz();
  const [stage, setStage] = useState<Stage>('reranked');
  const ranking = RANKINGS[stage];
  let seen = 0;
  let precisionSum = 0;
  const rows = ranking.map((id, index) => {
    const relevant = RELEVANT.has(id);
    if (relevant) { seen += 1; precisionSum += seen / (index + 1); }
    return [index + 1, id, relevant ? 'yes' : 'no', (seen / (index + 1)).toFixed(3)];
  });
  const pAtTwo = ranking.slice(0, 2).filter((id) => RELEVANT.has(id)).length / 2;
  const ap = precisionSum / RELEVANT.size;

  return <VizPanel title="Compare the four retrieval stages"
    hint="The dense vectors and final reranker are hand-designed teaching proxies. All four lists come from the same six-document script and relevance labels."
    table={{columns: ['rank', 'document ID', 'relevant', 'precision at rank'], rows}}
    controls={<div className={s.controls}>
      <label className={s.control}>stage
        <select className={s.select} value={stage} aria-label="Retrieval stage" onChange={(event) => setStage(event.target.value as Stage)}>
          {Object.keys(RANKINGS).map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
      </label>
    </div>}>
    <div style={{display: 'grid', gap: '0.6rem'}}>
      {ranking.slice(0, 3).map((id, index) => <div key={id} style={{borderLeft: `0.35rem solid ${seriesColor(RELEVANT.has(id) ? 0 : 1, dark)}`, background: 'var(--ifm-color-emphasis-100)', padding: '0.45rem 0.7rem'}}>
        #{index + 1} document {id}: {RELEVANT.has(id) ? 'relevant' : 'not relevant'}
      </div>)}
      <div aria-live="polite">top two: [{ranking.slice(0, 2).join(', ')}]; P@2 <strong>{pAtTwo.toFixed(2)}</strong>; AP <strong>{ap.toFixed(3)}</strong></div>
    </div>
  </VizPanel>;
}
