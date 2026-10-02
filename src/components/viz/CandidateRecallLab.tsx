import {useState} from 'react';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';
import {seriesColor} from './palette';

const candidates = [{id: 'A', score: 0.9, relevant: false}, {id: 'B', score: 0.8, relevant: true}, {id: 'C', score: 0.6, relevant: false}, {id: 'D', score: 0.4, relevant: true}];
export default function CandidateRecallLab() {
  const dark = useDarkViz();
  const [limit, setLimit] = useState(2);
  const found = candidates.slice(0, limit).filter(item => item.relevant).length;
  return <VizPanel title="Candidate recall sets a ceiling"
    hint="This is an exact ordered toy list. A downstream ranker cannot recover relevant items that retrieval omitted."
    controls={<label className={s.control}>Candidates retrieved: {limit}<input aria-label="Candidate count" type="range" min="1" max="4" step="1" value={limit} onChange={event => setLimit(Number(event.target.value))} /></label>}
    table={{columns: ['Item', 'Retrieval score', 'Relevant', 'Retrieved'], rows: candidates.map((item, index) => [item.id, item.score.toFixed(1), item.relevant ? 'yes' : 'no', index < limit ? 'yes' : 'no'])}}>
    <div style={{display: 'grid', gap: '0.5rem'}}>
      {candidates.map((item, index) => <div key={item.id} style={{padding: '0.45rem', borderLeft: `0.35rem solid ${seriesColor(item.relevant ? 0 : 1, dark)}`, opacity: index < limit ? 1 : 0.5, background: 'var(--ifm-color-emphasis-100)'}}>
        {item.id} · score {item.score.toFixed(1)} · {item.relevant ? 'relevant' : 'not labelled relevant'} · {index < limit ? 'retrieved' : 'omitted'}
      </div>)}
      <output>Recall among two labelled relevant items: {found}/2 = {(found / 2).toFixed(2)}.</output>
    </div>
  </VizPanel>;
}
