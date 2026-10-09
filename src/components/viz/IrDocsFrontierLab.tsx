import {useMemo, useState} from 'react';

import {FRONTIER, coresNeeded, pickConfiguration} from './irProjectsMath';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 280;
const PAD = {top: 16, right: 20, bottom: 40, left: 52};
const LOG_MIN = Math.log10(0.2);
const LOG_MAX = Math.log10(150);
const Y_MIN = 0.52;
const Y_MAX = 0.74;

export default function IrDocsFrontierLab() {
  const dark = useDarkViz();
  const [budget, setBudget] = useState(10);
  const [noise, setNoise] = useState(0.025);
  const [qps, setQps] = useState(100);

  const pick = useMemo(() => pickConfiguration(FRONTIER, budget, noise), [budget, noise]);
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const x = (ms: number) => PAD.left + ((Math.log10(ms) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * innerW;
  const y = (v: number) => PAD.top + innerH - ((v - Y_MIN) / (Y_MAX - Y_MIN)) * innerH;
  const base = seriesColor(0, dark);
  const chosenColor = seriesColor(2, dark);
  const muted = seriesColor(1, dark);

  const rows = FRONTIER.map((p) => [
    p.name,
    p.ndcg.toFixed(3),
    p.p95.toFixed(1),
    p.p95 <= budget ? 'yes' : 'no',
    String(coresNeeded(p.p50, qps)),
  ]);

  return (
    <VizPanel
      title="Pick a search configuration under a latency budget"
      hint="Points are the eight configurations measured on the 106 test queries. Set the p95 budget and the size of difference you are willing to call noise. The rule picks the cheapest configuration whose quality is within that noise of the best one that fits the budget. Defaults give hybrid at 0.705; a budget of 3 ms gives BM25 at 0.561."
      legend={[
        {label: 'configuration', color: base},
        {label: 'over the budget', color: muted},
        {label: 'chosen', color: chosenColor},
      ]}
      table={{columns: ['configuration', 'nDCG@10', 'p95 ms', 'fits budget', 'cores at this load'], rows}}
      controls={
        <>
          <label className={s.control}>
            p95 budget (ms)
            <input type="range" min={1} max={150} step={1} value={budget} onChange={(e) => setBudget(Number(e.target.value))} />
            <span className={s.value}>{budget}</span>
          </label>
          <label className={s.control}>
            noise band (nDCG)
            <input type="range" min={0} max={0.1} step={0.005} value={noise} onChange={(e) => setNoise(Number(e.target.value))} />
            <span className={s.value}>{noise.toFixed(3)}</span>
          </label>
          <label className={s.control}>
            requests per second
            <input type="range" min={10} max={500} step={10} value={qps} onChange={(e) => setQps(Number(e.target.value))} />
            <span className={s.value}>{qps}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {pick.chosen
              ? `chosen: ${pick.chosen.name}, nDCG ${pick.chosen.ndcg.toFixed(3)}, p95 ${pick.chosen.p95} ms, ${coresNeeded(pick.chosen.p50, qps)} core(s) at ${qps}/s`
              : 'nothing fits this budget'}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Quality against latency. Chosen configuration: ${pick.chosen ? pick.chosen.name : 'none'}`}>
        <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} />
        <line className={s.axis} x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + innerH} />
        {[0.55, 0.6, 0.65, 0.7].map((v) => (
          <text key={v} className={s.tick} x={PAD.left - 6} y={y(v) + 4} textAnchor="end">
            {v.toFixed(2)}
          </text>
        ))}
        {[0.3, 1, 3, 10, 30, 100].map((ms) => (
          <text key={ms} className={s.tick} x={x(ms)} y={H - 22} textAnchor="middle">
            {ms}
          </text>
        ))}
        <line x1={x(Math.max(budget, 0.2))} y1={PAD.top} x2={x(Math.max(budget, 0.2))} y2={PAD.top + innerH} stroke={muted} strokeWidth={2} strokeDasharray="5 3" />
        {FRONTIER.map((p) => {
          const isChosen = pick.chosen?.name === p.name;
          const fits = p.p95 <= budget;
          return (
            <g key={p.name}>
              <circle cx={x(p.p95)} cy={y(p.ndcg)} r={isChosen ? 8 : 5.5} fill={isChosen ? chosenColor : fits ? base : 'none'} stroke={isChosen ? chosenColor : fits ? base : muted} strokeWidth={2} />
              {isChosen && (
                <text className={s.dataLabel} x={x(p.p95)} y={y(p.ndcg) - 12} textAnchor="middle">
                  {p.name}
                </text>
              )}
            </g>
          );
        })}
        <text className={s.axisLabel} x={W / 2} y={H - 4} textAnchor="middle">
          p95 latency (ms, log scale)
        </text>
        <text className={s.axisLabel} x={14} y={PAD.top + innerH / 2} textAnchor="middle" transform={`rotate(-90 14 ${PAD.top + innerH / 2})`}>
          nDCG@10
        </text>
      </svg>
    </VizPanel>
  );
}
