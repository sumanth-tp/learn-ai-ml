import {useState} from 'react';

import {seriesColor} from './palette';
import {FUNNEL_CATALOGUE, FUNNEL_GRID, FUNNEL_RELEVANT, FUNNEL_USERS} from './rankingFunnelData';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 250;
const K1S = [100, 300, 1000, 2000];
const K2S = [50, 100, 200];
const UTILISATION = 0.5;

export default function RankingFunnelLab() {
  const dark = useDarkViz();
  const [k1, setK1] = useState(1000);
  const [k2, setK2] = useState(200);
  const [lightUs, setLightUs] = useState(2);
  const [heavyUs, setHeavyUs] = useState(60);
  const [qps, setQps] = useState(20000);
  const [retrievalMs, setRetrievalMs] = useState(3);

  const second = Math.min(k2, k1);
  const row = FUNNEL_GRID.find((r) => r[0] === k1 && r[1] === second) ?? FUNNEL_GRID[0];
  const [, , recallK1, recallK2, precision, ndcg] = row;
  const cpuMs = retrievalMs + (k1 * lightUs + second * heavyUs) / 1000;
  const cores = (qps * (cpuMs / 1000)) / UTILISATION;

  const stages = [
    {label: `catalogue ${FUNNEL_CATALOGUE.toLocaleString()}`, value: 1, note: 'all items'},
    {label: `retrieve ${k1.toLocaleString()}`, value: recallK1, note: `${recallK1.toFixed(3)} of the ${FUNNEL_RELEVANT} relevant items are inside`},
    {label: `light ranker keeps ${second}`, value: recallK2, note: `${recallK2.toFixed(3)} of the relevant items survive`},
    {label: 'heavy ranker shows 10', value: precision, note: `precision at 10 ${precision.toFixed(3)}, NDCG at 10 ${ndcg.toFixed(3)}`},
  ];

  const barX = 20;
  const barW = W - 2 * barX;
  const rows = FUNNEL_GRID.map(([a, b, c, d, e, f]) => [a, b, c.toFixed(3), d.toFixed(3), e.toFixed(3), f.toFixed(3)]);

  return (
    <VizPanel
      title="Ranking funnel: how deep to retrieve, how many to rank heavily"
      hint={`Results of the chapter simulation: ${FUNNEL_CATALOGUE.toLocaleString()} items, ${FUNNEL_USERS} evaluation users, relevant means the ${FUNNEL_RELEVANT} items of highest true utility. Defaults (1000 retrieved, 200 kept) give precision at 10 of 0.345 and NDCG 0.337. With the default cost parameters that is 17.0 CPU ms and 680 cores, the same arithmetic the chapter prints. The cost numbers are parameters you set, not measurements.`}
      legend={[
        {label: 'recall or precision reached at the stage', color: seriesColor(0, dark)},
        {label: 'precision at 10 shown to the user', color: seriesColor(1, dark)},
      ]}
      table={{columns: ['K1', 'K2', 'recall@K1', 'recall@K2', 'precision@10', 'NDCG@10'], rows}}
      controls={
        <>
          <label className={s.control}>
            retrieve K1
            <select className={s.select} value={k1} onChange={(e) => setK1(Number(e.target.value))}>
              {K1S.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            light ranker keeps K2
            <select className={s.select} value={second} onChange={(e) => setK2(Number(e.target.value))}>
              {K2S.filter((k) => k <= k1).map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            light cost per item (µs)
            <input className={s.select} type="number" min={0.1} step={0.5} value={lightUs} onChange={(e) => setLightUs(Math.max(0.1, Number(e.target.value)))} />
          </label>
          <label className={s.control}>
            heavy cost per item (µs)
            <input className={s.select} type="number" min={1} step={5} value={heavyUs} onChange={(e) => setHeavyUs(Math.max(1, Number(e.target.value)))} />
          </label>
          <label className={s.control}>
            retrieval cost per request (ms)
            <input className={s.select} type="number" min={0.5} step={0.5} value={retrievalMs} onChange={(e) => setRetrievalMs(Math.max(0.5, Number(e.target.value)))} />
          </label>
          <label className={s.control}>
            requests per second
            <input className={s.select} type="number" min={100} step={1000} value={qps} onChange={(e) => setQps(Math.max(100, Number(e.target.value)))} />
          </label>
          <span className={s.value} aria-live="polite">
            precision at 10 {precision.toFixed(3)}, {cpuMs.toFixed(1)} ms CPU per request, {Math.ceil(cores).toLocaleString()} cores at {qps.toLocaleString()} requests per second
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Funnel: ${stages.map((st) => `${st.label}, ${st.note}`).join('; ')}`}>
        {stages.map((st, i) => {
          const y = 8 + i * 60;
          return (
            <g key={st.label}>
              <text className={s.dataLabel} x={barX} y={y + 12}>
                {st.label}
              </text>
              <rect x={barX} y={y + 18} width={barW} height={16} rx={4} fill="none" stroke="var(--border, #aaa)" />
              <rect x={barX} y={y + 18} width={Math.max(2, barW * st.value)} height={16} rx={4} fill={seriesColor(i === 3 ? 1 : 0, dark)} />
              <text className={s.tick} x={barX} y={y + 50}>
                {st.note}
              </text>
            </g>
          );
        })}
      </svg>
    </VizPanel>
  );
}
