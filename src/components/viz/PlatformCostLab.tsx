import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const TEAMS = ['search', 'ads', 'vision', 'nlp'] as const;
const QUOTA: Record<string, number> = {search: 24, ads: 16, vision: 16, nlp: 8};
const DEMAND: Record<string, number[]> = {"search":[13,18,17,19,20,20,24,20,21,19,18,14,15,11,9,6,5,6,6,6,18,8,9,12,14,15,18,20,28,20,22,23,20,18,17,16,11,11,11,7,8,6,5,6,4,7,11,10,15,15,18,19,17,20,21,19,18,19,16,13,12,12,9,9,8,6,4,4,6,9,9,12,12,15,16,18,18,19,20,19,19,19,17,19,14,10,10,8,5,7,4,2,7,7,11,10,12,17,16,19,21,21,21,21,34,18,18,16,16,11,8,8,6,6,5,7,5,7,6,10,15,14,16,17,20,21,19,21,20,19,16,17,13,11,8,7,7,5,6,6,8,8,9,9,13,16,17,19,19,20,20,22,20,18,15,15,15,12,8,9,8,7,6,7,7,8,16,13],"ads":[6,5,6,5,6,6,9,8,11,14,16,15,13,15,13,14,12,11,12,9,9,5,4,5,6,3,4,3,6,9,9,10,10,24,15,15,15,13,16,13,12,21,6,8,6,6,19,4,1,3,5,5,5,6,10,11,12,11,16,11,14,17,13,13,10,12,7,8,6,4,5,2,3,4,4,6,7,7,8,10,11,14,13,15,14,13,15,12,12,12,8,8,4,5,3,3,3,4,6,6,7,8,9,9,10,13,16,15,14,16,23,12,10,15,9,10,6,4,4,11,2,4,12,6,7,6,7,12,13,11,14,14,15,14,13,13,12,10,7,4,5,3,5,5,4,3,4,7,5,9,10,11,13,12,15,13,13,13,14,12,12,9,10,10,7,7,5,17],"vision":[7,7,6,2,2,2,2,0,3,4,3,5,7,7,9,10,10,10,9,11,9,8,9,6,7,6,4,4,4,3,1,1,4,4,2,3,6,6,8,9,10,8,10,24,8,10,9,8,6,4,3,4,4,4,4,4,2,2,4,6,6,5,7,7,9,8,9,10,9,10,7,4,7,7,6,4,4,3,1,2,4,3,7,3,6,8,7,7,8,9,11,10,10,8,9,7,5,5,4,5,3,3,5,14,1,5,5,5,6,7,8,8,10,8,9,10,9,9,8,6,6,4,2,2,2,2,3,4,4,6,6,5,8,8,7,9,10,10,11,10,10,8,8,7,7,3,5,5,4,3,3,2,2,3,4,6,5,8,6,8,9,11,9,7,10,9,7,6],"nlp":[9,8,9,7,9,6,21,3,4,3,3,3,1,1,13,3,3,6,6,3,6,8,9,7,10,7,7,7,7,7,6,4,5,2,2,4,2,3,16,3,3,4,5,6,7,9,8,9,9,7,6,9,8,6,5,3,2,3,1,3,1,2,5,5,3,6,4,5,7,9,7,9,8,6,7,9,6,6,6,3,5,5,4,4,1,1,1,2,3,5,7,5,7,6,18,7,6,8,8,7,6,5,5,4,4,1,3,3,4,3,3,2,1,4,22,8,3,6,7,7,8,8,7,7,5,5,4,5,2,3,3,2,3,1,3,4,6,5,7,6,7,9,20,10,9,10,10,6,9,6,6,4,4,2,2,10,2,3,9,3,3,2,6,6,7,9,22,8]};
const HOURS = 168;

export const METHODS = ['usage only', 'idle split equally', 'idle by usage', 'idle by quota'] as const;
type Method = (typeof METHODS)[number];

export function week(cluster: number) {
  const used: Record<string, number> = {search: 0, ads: 0, vision: 0, nlp: 0};
  let unmetHours = 0;
  let unmet = 0;
  const pooled: number[] = [];
  for (let h = 0; h < HOURS; h += 1) {
    const total = TEAMS.reduce((a, t) => a + DEMAND[t][h], 0);
    pooled.push(total);
    const scale = total > cluster ? cluster / total : 1;
    if (total > cluster) {
      unmetHours += 1;
      unmet += total - cluster;
    }
    for (const t of TEAMS) used[t] += DEMAND[t][h] * scale;
  }
  const usedTotal = TEAMS.reduce((a, t) => a + used[t], 0);
  return {used, usedTotal, idle: cluster * HOURS - usedTotal, unmetHours, unmet, pooled};
}

export function bills(cluster: number, price: number, method: Method) {
  const w = week(cluster);
  const quotaSum = TEAMS.reduce((a, t) => a + QUOTA[t], 0);
  const out: Record<string, {usage: number; idleShare: number}> = {};
  for (const t of TEAMS) {
    let idleShare = 0;
    if (method === 'idle split equally') idleShare = w.idle / TEAMS.length;
    if (method === 'idle by usage') idleShare = (w.idle * w.used[t]) / w.usedTotal;
    if (method === 'idle by quota') idleShare = (w.idle * QUOTA[t]) / quotaSum;
    out[t] = {usage: w.used[t] * price, idleShare: idleShare * price};
  }
  return {out, week: w, cost: cluster * HOURS * price};
}

const W = 640;
const H = 300;

export default function PlatformCostLab() {
  const dark = useDarkViz();
  const [cluster, setCluster] = useState(64);
  const [price, setPrice] = useState(1);
  const [method, setMethod] = useState<Method>('idle by usage');

  const peaks = useMemo(() => TEAMS.map((t) => Math.max(...DEMAND[t])), []);
  const sumPeaks = peaks.reduce((a, b) => a + b, 0);
  const result = bills(cluster, price, method);
  const w = result.week;
  const pooledPeak = Math.max(...w.pooled);
  const blue = seriesColor(0, dark);
  const orange = seriesColor(1, dark);
  const violet = seriesColor(3, dark);

  const barMax = Math.max(...TEAMS.map((t) => result.out[t].usage + result.out[t].idleShare), 1) * 1.1;
  const barTop = 28;
  const barH = 200;
  const lineX = 330;
  const lineW = 290;
  const yMax = Math.max(sumPeaks, cluster, pooledPeak) * 1.08;
  const ly = (v: number) => 250 - (v / yMax) * 210;
  const lx = (h: number) => lineX + (h / (HOURS - 1)) * lineW;

  const rows = [
    ...TEAMS.map((t, i) => [
      t,
      QUOTA[t],
      peaks[i],
      Math.round(w.used[t]),
      Math.round(result.out[t].usage + result.out[t].idleShare),
    ]),
    ['all', 64, sumPeaks, Math.round(w.usedTotal), Math.round(TEAMS.reduce((a, t) => a + result.out[t].usage + result.out[t].idleShare, 0))],
  ];

  const status = `pooled peak ${pooledPeak} GPUs against ${sumPeaks} for separate partitions; idle ${Math.round(w.idle)} GPU-hours (${((w.idle / (cluster * HOURS)) * 100).toFixed(1)}%); ${w.unmetHours} hours with unmet demand`;

  return (
    <VizPanel
      title="Who pays for the idle GPUs?"
      hint="One week of demand from four teams. Pooling needs far fewer GPUs than giving each team its own peak. Idle capacity then has to be charged to someone: change the method and watch each team's bill. Defaults (64 GPUs, idle by usage, price 1.0) give search 4,123, ads 2,913, vision 1,921 and nlp 1,795, the chapter's printed values."
      legend={[
        {label: 'GPU-hours used', color: blue},
        {label: 'share of idle cost', color: orange},
        {label: 'cluster size', color: violet},
      ]}
      table={{columns: ['team', 'quota', 'peak', 'GPU-hours used', 'bill'], rows}}
      controls={
        <>
          <label className={s.control}>
            cluster GPUs
            <input type="range" min={40} max={120} step={4} value={cluster} onChange={(e) => setCluster(Number(e.target.value))} />
            <span className={s.value}>{cluster}</span>
          </label>
          <label className={s.control}>
            price per GPU-hour
            <input type="range" min={0.5} max={3} step={0.1} value={price} onChange={(e) => setPrice(Number(e.target.value))} />
            <span className={s.value}>{price.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            idle cost
            <select className={s.select} value={method} onChange={(e) => setMethod(e.target.value as Method)}>
              {METHODS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Stacked bills per team and the pooled hourly demand against cluster size">
        <line className={s.axis} x1={20} y1={barTop + barH} x2={300} y2={barTop + barH} />
        {TEAMS.map((t, i) => {
          const x = 36 + i * 66;
          const u = (result.out[t].usage / barMax) * barH;
          const idle = (result.out[t].idleShare / barMax) * barH;
          return (
            <g key={t}>
              <rect x={x} y={barTop + barH - u} width={46} height={u} fill={blue} />
              <rect x={x} y={barTop + barH - u - idle} width={46} height={idle} fill={orange} />
              <text className={s.dataLabel} x={x + 23} y={barTop + barH - u - idle - 5} textAnchor="middle">
                {Math.round(result.out[t].usage + result.out[t].idleShare).toLocaleString()}
              </text>
              <text className={s.tick} x={x + 23} y={barTop + barH + 14} textAnchor="middle">
                {t}
              </text>
            </g>
          );
        })}
        <text className={s.axisLabel} x={160} y={barTop + barH + 34} textAnchor="middle">
          weekly bill by team
        </text>
        <line className={s.axis} x1={lineX} y1={250} x2={lineX + lineW} y2={250} />
        <path
          d={w.pooled.map((v, h) => `${h ? 'L' : 'M'}${lx(h).toFixed(1)},${ly(v).toFixed(1)}`).join(' ')}
          fill="none"
          stroke={blue}
          strokeWidth={1.8}
        />
        <line x1={lineX} y1={ly(cluster)} x2={lineX + lineW} y2={ly(cluster)} stroke={violet} strokeWidth={2} />
        <line x1={lineX} y1={ly(sumPeaks)} x2={lineX + lineW} y2={ly(sumPeaks)} stroke="var(--text-faint)" strokeDasharray="5 3" />
        <text className={s.tick} x={lineX + lineW} y={ly(sumPeaks) - 4} textAnchor="end">
          each team at its own peak: {sumPeaks}
        </text>
        <text className={s.tick} x={lineX + lineW} y={ly(cluster) - 4} textAnchor="end">
          cluster: {cluster}
        </text>
        <text className={s.axisLabel} x={lineX + lineW / 2} y={H - 18} textAnchor="middle">
          168 hours of pooled demand (peak {pooledPeak})
        </text>
      </svg>
    </VizPanel>
  );
}
