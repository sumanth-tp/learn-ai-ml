import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Result = {id: string; grade: number};
const INITIAL: Result[] = [
  {id: 'A', grade: 3}, {id: 'B', grade: 0}, {id: 'C', grade: 2},
  {id: 'D', grade: 0}, {id: 'E', grade: 1},
];

function measures(results: Result[]) {
  let relevantSeen = 0;
  let precisionSum = 0;
  const details = results.map((result, index) => {
    if (result.grade > 0) relevantSeen += 1;
    const precision = relevantSeen / (index + 1);
    if (result.grade > 0) precisionSum += precision;
    return {precision, gain: (2 ** result.grade - 1) / Math.log2(index + 2)};
  });
  const totalRelevant = results.filter((result) => result.grade > 0).length;
  const firstRelevant = results.findIndex((result) => result.grade > 0);
  const ideal = [...results].sort((a, b) => b.grade - a.grade);
  const dcg = details.reduce((sum, detail) => sum + detail.gain, 0);
  const idcg = ideal.reduce((sum, result, index) => sum + (2 ** result.grade - 1) / Math.log2(index + 2), 0);
  return {details, ap: totalRelevant ? precisionSum / totalRelevant : 0,
    mrr: firstRelevant < 0 ? 0 : 1 / (firstRelevant + 1), ndcg: idcg ? dcg / idcg : 0,
    pAtThree: results.slice(0, 3).filter((result) => result.grade > 0).length / 3};
}

export default function RankingMetricsLab() {
  const dark = useDarkViz();
  const [results, setResults] = useState(INITIAL);
  const {details, ap, mrr, ndcg, pAtThree} = measures(results);
  const move = (index: number, direction: -1 | 1) => {
    const nextIndex = index + direction;
    if (nextIndex < 0 || nextIndex >= results.length) return;
    const next = [...results];
    [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
    setResults(next);
  };

  return (
    <VizPanel title="Rank-aware retrieval metrics"
      hint="Move results up or down. Average Precision rewards every relevant hit at its current rank; NDCG also uses graded relevance."
      table={{columns: ['rank', 'document', 'grade', 'P@rank', 'discounted gain'], rows: results.map((result, index) => [index + 1, result.id, result.grade, details[index].precision.toFixed(3), details[index].gain.toFixed(3)])}}
      controls={<div className={s.controls}>
        <span>AP: <strong>{ap.toFixed(3)}</strong></span>
        <span>P@3: <strong>{pAtThree.toFixed(3)}</strong></span>
        <span>MRR: <strong>{mrr.toFixed(3)}</strong></span>
        <span>NDCG: <strong>{ndcg.toFixed(3)}</strong></span>
      </div>}>
      <ol style={{listStyle: 'none', padding: 0, margin: 0, display: 'grid', gap: '0.5rem'}}>
        {results.map((result, index) => <li key={result.id} style={{display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap',
          borderLeft: `0.35rem solid ${seriesColor(result.grade > 0 ? 0 : 1, dark)}`, padding: '0.4rem 0.6rem', background: 'var(--ifm-color-emphasis-100)'}}>
          <strong>#{index + 1} {result.id}</strong>
          <span>grade {result.grade}</span>
          <span>P@{index + 1} {details[index].precision.toFixed(3)}</span>
          <span style={{marginLeft: 'auto', display: 'flex', gap: '0.3rem'}}>
            <button type="button" onClick={() => move(index, -1)} disabled={index === 0} aria-label={`Move ${result.id} up`}>↑</button>
            <button type="button" onClick={() => move(index, 1)} disabled={index === results.length - 1} aria-label={`Move ${result.id} down`}>↓</button>
          </span>
        </li>)}
      </ol>
    </VizPanel>
  );
}
