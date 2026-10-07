import {useMemo, useState} from 'react';

import {fuse} from './capstoneMath';
import {FUSION_LISTS} from './fusionData';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const QUERIES = Object.keys(FUSION_LISTS);
const W = 640;
const ROW = 30;
const SHOWN = 6;
const COLUMN_X = [20, 240, 460];
const COLUMN_W = 170;

const short = (text: string, max: number) => (text.length > max ? `${text.slice(0, max - 1)}…` : text);

export default function RrfFusionLab() {
  const dark = useDarkViz();
  const [query, setQuery] = useState(QUERIES[0]);
  const [semanticWeight, setSemanticWeight] = useState(0.6);
  const [k, setK] = useState(60);

  const {semantic, keyword} = FUSION_LISTS[query];
  const fused = useMemo(() => fuse(semantic, keyword, semanticWeight, k), [semantic, keyword, semanticWeight, k]);
  const colours = [seriesColor(0, dark), seriesColor(1, dark), seriesColor(2, dark), seriesColor(3, dark), seriesColor(4, dark)];
  const ids = useMemo(() => [...new Set(fused.map((f) => f.id))], [fused]);
  const colourOf = (id: string) => colours[ids.indexOf(id) % colours.length];
  const height = 56 + SHOWN * ROW + 12;
  const rowY = (rank: number) => 48 + rank * ROW;
  const columns: {title: string; items: {id: string; text: string}[]}[] = [
    {title: `semantic (weight ${semanticWeight.toFixed(2)})`, items: semantic.slice(0, SHOWN)},
    {title: `keyword BM25 (weight ${(1 - semanticWeight).toFixed(2)})`, items: keyword.slice(0, SHOWN)},
    {title: 'fused result', items: fused.slice(0, SHOWN)},
  ];

  const rows = fused.slice(0, 8).map((f, i) => [
    `${i + 1}`,
    f.id,
    f.semanticRank === null ? 'not in list' : `${f.semanticRank}`,
    f.keywordRank === null ? 'not in list' : `${f.keywordRank}`,
    f.score.toFixed(5),
  ]);

  return (
    <VizPanel
      title="Reciprocal rank fusion of two result lists"
      hint="Each list gives a chunk the score weight / (k + rank). Scores from both lists are added. A chunk that appears high in both lists wins; a chunk found by only one list can still win if that list carries enough weight. The lines connect the same chunk across the three columns."
      legend={[{label: 'one colour per chunk', color: colours[0]}]}
      table={{columns: ['fused rank', 'chunk', 'rank in semantic list', 'rank in keyword list', 'score'], rows}}
      controls={
        <>
          <label className={s.control}>
            query
            <select className={s.select} value={query} onChange={(e) => setQuery(e.target.value)}>
              {QUERIES.map((q) => (
                <option key={q} value={q}>
                  {q}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            semantic weight
            <input type="range" min={0} max={1} step={0.05} value={semanticWeight} onChange={(e) => setSemanticWeight(Number(e.target.value))} />
            <span className={s.value}>{semanticWeight.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            k
            <input type="range" min={1} max={100} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} />
            <span className={s.value}>{k}</span>
          </label>
          <span className={s.value} aria-live="polite">
            top result: {fused[0]?.id} ({fused[0]?.score.toFixed(5)})
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`Fused ranking for the query ${query}. The top result is ${fused[0]?.id}`}>
        {columns.map((column, c) => (
          <g key={column.title}>
            <text className={s.axisLabel} x={COLUMN_X[c]} y={20}>
              {column.title}
            </text>
            {column.items.map((item, rank) => (
              <g key={item.id}>
                <rect x={COLUMN_X[c]} y={rowY(rank) - 18} width={COLUMN_W} height={ROW - 4} rx={4} fill={colourOf(item.id)} opacity={0.88} />
                <title>{item.text}</title>
                <text className={s.tick} x={COLUMN_X[c] + 6} y={rowY(rank) - 5} fill="#fff">
                  {rank + 1}. {short(item.id, 26)}
                </text>
              </g>
            ))}
          </g>
        ))}
        {[0, 1].map((c) =>
          columns[c].items.map((item, rank) => {
            const target = columns[c + 1].items.findIndex((other) => other.id === item.id);
            if (target < 0) return null;
            return (
              <line
                key={`${c}-${item.id}`}
                x1={COLUMN_X[c] + COLUMN_W}
                y1={rowY(rank) - 8}
                x2={COLUMN_X[c + 1]}
                y2={rowY(target) - 8}
                stroke={colourOf(item.id)}
                strokeWidth={1.6}
                opacity={0.6}
              />
            );
          }),
        )}
      </svg>
    </VizPanel>
  );
}
