import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import {RERANK} from './rerankData';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Model = 'general' | 'tuned';

const W = 640;
const ROW = 26;
const SHOWN = 10;
const COLUMN_X = [10, 340];
const COLUMN_W = 290;
const QUERIES = RERANK.nrel.length;

const ndcg10 = (rel: number[], nrel: number) => {
  let dcg = 0;
  for (let i = 0; i < Math.min(10, rel.length); i++) dcg += rel[i] / Math.log2(i + 2);
  let ideal = 0;
  for (let i = 0; i < Math.min(nrel, 10); i++) ideal += 1 / Math.log2(i + 2);
  return dcg / ideal;
};

const order = (scores: number[], depth: number) =>
  Array.from({length: depth}, (_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b);

const reranked = (qi: number, model: Model, depth: number): number[] => {
  const scores = (model === 'general' ? RERANK.general : RERANK.tuned)[qi];
  const head = order(scores, depth);
  const tail = Array.from({length: Math.max(0, 50 - depth)}, (_, i) => depth + i);
  return [...head, ...tail];
};

const relOf = (qi: number) => Array.from(RERANK.rel[qi]).map(Number);

const short = (text: string, max: number) => (text.length > max ? `${text.slice(0, max - 1)}…` : text);

export default function RerankLab() {
  const dark = useDarkViz();
  const [model, setModel] = useState<Model>('general');
  const [depth, setDepth] = useState(20);
  const [example, setExample] = useState(0);

  const allowed = model === 'general' ? [5, 10, 20, 30, 50] : [5, 10, 15, 20];
  const effectiveDepth = allowed.includes(depth) ? depth : 20;

  const averages = useMemo(() => {
    let before = 0;
    let after = 0;
    let up = 0;
    let down = 0;
    for (let qi = 0; qi < QUERIES; qi++) {
      const rel = relOf(qi);
      const b = ndcg10(rel, RERANK.nrel[qi]);
      const a = ndcg10(reranked(qi, model, effectiveDepth).map((j) => rel[j]), RERANK.nrel[qi]);
      before += b;
      after += a;
      if (a > b + 1e-9) up++;
      else if (a < b - 1e-9) down++;
    }
    return {before: before / QUERIES, after: after / QUERIES, up, down};
  }, [model, effectiveDepth]);

  const ex = RERANK.examples[example];
  const rel = relOf(ex.q);
  const after = reranked(ex.q, model, effectiveDepth);
  const nBefore = ndcg10(rel, RERANK.nrel[ex.q]);
  const nAfter = ndcg10(after.map((j) => rel[j]), RERANK.nrel[ex.q]);

  const good = seriesColor(2, dark);
  const plain = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;
  const height = 40 + SHOWN * ROW + 8;
  const rowY = (rank: number) => 34 + rank * ROW;
  const titleOf = (j: number) => (j < 20 ? ex.titles[j] : `candidate ${j + 1}`);

  const columns = [
    {title: 'hybrid first stage', items: Array.from({length: SHOWN}, (_, i) => i)},
    {title: `after reranking top ${effectiveDepth}`, items: after.slice(0, SHOWN)},
  ];

  const rows = columns[1].items.map((j, i) => [
    i + 1,
    short(titleOf(j), 60),
    j + 1,
    rel[j] ? 'relevant' : '-',
  ]);

  return (
    <VizPanel
      title="Rerank the hybrid shortlist: general model against one fine-tuned on SciFact"
      hint="Left is the first stage, right is the same shortlist after a cross-encoder rescored it. A mark and a green bar show a relevant document. The summary line averages all 300 test queries; the default (general model, depth 20) reproduces block 6."
      legend={[
        {label: 'relevant document', color: good},
        {label: 'not relevant', color: plain},
      ]}
      table={{columns: ['new rank', 'document', 'first-stage rank', 'label'], rows}}
      controls={
        <>
          <label className={s.control}>
            query
            <select className={s.select} value={example} onChange={(e) => setExample(Number(e.target.value))}>
              {RERANK.examples.map((e, i) => (
                <option key={e.q} value={i}>
                  {short(e.text, 54)}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            reranker
            <select className={s.select} value={model} onChange={(e) => setModel(e.target.value as Model)}>
              <option value="general">general (MS MARCO)</option>
              <option value="tuned">fine-tuned on SciFact</option>
            </select>
          </label>
          <label className={s.control}>
            depth
            <select className={s.select} value={effectiveDepth} onChange={(e) => setDepth(Number(e.target.value))}>
              {allowed.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          <span className={s.value} aria-live="polite">
            all {QUERIES} queries: nDCG@10 {averages.before.toFixed(3)} to {averages.after.toFixed(3)} ({averages.up} better, {averages.down} worse)
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`Top ten before and after reranking for the query ${ex.text}. nDCG at 10 goes from ${nBefore.toFixed(2)} to ${nAfter.toFixed(2)}`}>
        {columns.map((column, c) => (
          <g key={column.title}>
            <text className={s.axisLabel} x={COLUMN_X[c]} y={16}>
              {column.title}
            </text>
            {column.items.map((j, rank) => {
              const isRel = rel[j] === 1;
              return (
                <g key={`${c}-${rank}`}>
                  <rect x={COLUMN_X[c]} y={rowY(rank) - 14} width={COLUMN_W} height={ROW - 4} rx={4} fill={isRel ? good : plain} opacity={isRel ? 0.9 : 0.28} />
                  <title>{titleOf(j)}</title>
                  <text className={s.tick} x={COLUMN_X[c] + 6} y={rowY(rank) - 2} fill={isRel ? '#fff' : undefined}>
                    {rank + 1}. {isRel ? '✓ ' : ''}
                    {short(titleOf(j), 44)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
        <text className={s.dataLabel} x={COLUMN_X[0]} y={height - 6}>
          this query: nDCG@10 {nBefore.toFixed(3)}
        </text>
        <text className={s.dataLabel} x={COLUMN_X[1]} y={height - 6}>
          this query: nDCG@10 {nAfter.toFixed(3)}
        </text>
      </svg>
    </VizPanel>
  );
}
