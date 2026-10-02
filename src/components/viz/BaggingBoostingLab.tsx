import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 350;
const XS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const YS = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1];
const BAG_ROWS = [
  [4, 5, 7, 9, 0, 1, 8, 9, 2, 3],
  [8, 4, 2, 8, 2, 4, 6, 5, 0, 0],
  [8, 7, 8, 5, 8, 3, 4, 7, 1, 3],
  [1, 4, 9, 1, 3, 4, 9, 2, 5, 2],
  [0, 7, 0, 2, 4, 4, 1, 9, 7, 9],
  [0, 7, 2, 5, 9, 2, 7, 1, 3, 9],
  [4, 5, 2, 1, 4, 6, 4, 7, 3, 6],
];

export type Stump = {error: number; threshold: number; side: number};
export type BoostRound = Stump & {
  alpha: number;
  weightsUsed: number[];
  weightsAfter: number[];
  accuracy: number;
  score: number[];
};

export function stumpPredict(threshold: number, side: number, x: number): number {
  return x > threshold ? side : -side;
}

export function bestStump(weights: number[]): Stump {
  let best: Stump | null = null;
  for (let i = 0; i <= 10; i += 1) {
    const threshold = 0.5 + i;
    for (const side of [1, -1]) {
      let error = 0;
      for (let j = 0; j < XS.length; j += 1) {
        if (stumpPredict(threshold, side, XS[j]) !== YS[j]) error += weights[j];
      }
      if (best === null || error < best.error - 1e-12) best = {error, threshold, side};
    }
  }
  return best as Stump;
}

export function ensembleAccuracy(score: number[]): number {
  let right = 0;
  for (let j = 0; j < YS.length; j += 1) {
    if (Math.sign(score[j]) === YS[j]) right += 1;
  }
  return right / YS.length;
}

export function adaBoost(rounds: number): BoostRound[] {
  let weights = XS.map(() => 1 / XS.length);
  const score = XS.map(() => 0);
  const out: BoostRound[] = [];
  for (let r = 0; r < rounds; r += 1) {
    const stump = bestStump(weights);
    const alpha = 0.5 * Math.log((1 - stump.error) / stump.error);
    const used = weights.slice();
    let next = weights.map(
      (w, j) => w * Math.exp(-alpha * YS[j] * stumpPredict(stump.threshold, stump.side, XS[j])),
    );
    const total = next.reduce((a, b) => a + b, 0);
    next = next.map((w) => w / total);
    weights = next;
    for (let j = 0; j < XS.length; j += 1) {
      score[j] += alpha * stumpPredict(stump.threshold, stump.side, XS[j]);
    }
    out.push({
      ...stump,
      alpha,
      weightsUsed: used,
      weightsAfter: next,
      accuracy: ensembleAccuracy(score),
      score: score.slice(),
    });
  }
  return out;
}

export type BagRound = Stump & {counts: number[]; distinct: number; accuracy: number; tally: number[]};

export function bagging(count: number): BagRound[] {
  const tally = XS.map(() => 0);
  const out: BagRound[] = [];
  for (let i = 0; i < count; i += 1) {
    const counts = XS.map(() => 0);
    BAG_ROWS[i].forEach((idx) => {
      counts[idx] += 1;
    });
    const stump = bestStump(counts.map((c) => c / XS.length));
    for (let j = 0; j < XS.length; j += 1) {
      tally[j] += stumpPredict(stump.threshold, stump.side, XS[j]);
    }
    out.push({
      ...stump,
      counts,
      distinct: counts.filter((c) => c > 0).length,
      accuracy: ensembleAccuracy(tally),
      tally: tally.slice(),
    });
  }
  return out;
}

const rule = (st: Stump) =>
  `x > ${st.threshold.toFixed(1)} gives ${st.side > 0 ? '+1' : '-1'}, else ${st.side > 0 ? '-1' : '+1'}`;

export default function BaggingBoostingLab() {
  const dark = useDarkViz();
  const [mode, setMode] = useState<'boost' | 'bag'>('boost');
  const [round, setRound] = useState(3);
  const [stumps, setStumps] = useState(7);

  const boost = useMemo(() => adaBoost(5), []);
  const bag = useMemo(() => bagging(7), []);

  const posColor = seriesColor(0, dark);
  const negColor = seriesColor(1, dark);
  const goodColor = seriesColor(2, dark);

  const x0 = 50;
  const step = 60;
  const px = (v: number) => x0 + (v - 1) * step;

  let sizes: number[];
  let stump: Stump | null = null;
  let score: number[] | null = null;
  let headline: string;
  let accuracy: number;

  if (mode === 'boost') {
    const current = round === 0 ? null : boost[round - 1];
    sizes = current ? current.weightsUsed : XS.map(() => 1 / XS.length);
    stump = current;
    score = current ? current.score : null;
    accuracy = current ? current.accuracy : 0;
    headline = current
      ? `round ${round}: weighted error ${current.error.toFixed(3)}, alpha ${current.alpha.toFixed(3)}, ensemble accuracy ${(current.accuracy * 100).toFixed(0)}%`
      : 'round 0: every point carries weight 0.100';
  } else {
    const current = bag[stumps - 1];
    sizes = current.counts.map((c) => c / XS.length);
    stump = current;
    score = current.tally;
    accuracy = current.accuracy;
    headline = `${stumps} bagged stumps: latest sees ${current.distinct} distinct points, vote accuracy ${(current.accuracy * 100).toFixed(0)}%`;
  }

  const radius = (w: number) => (w === 0 ? 0 : 6 + Math.sqrt(w) * 30);
  const lineY = 130;

  const table =
    mode === 'boost'
      ? {
          columns: ['round', 'stump', 'error', 'alpha', 'ensemble accuracy'],
          rows: boost.map((b, i) => [
            i + 1,
            rule(b),
            b.error.toFixed(3),
            b.alpha.toFixed(3),
            `${(b.accuracy * 100).toFixed(0)}%`,
          ]),
        }
      : {
          columns: ['stump', 'distinct points', 'rule', 'vote accuracy'],
          rows: bag.map((b, i) => [i + 1, b.distinct, rule(b), `${(b.accuracy * 100).toFixed(0)}%`]),
        };

  return (
    <VizPanel
      title="Bagging and boosting on the same ten points"
      hint={
        mode === 'boost'
          ? 'Circle size is the weight used to fit that round. After round 1 the three misclassified points carry 0.167 each, and three rounds of one-split stumps classify all ten points. Move the slider to replay the table from the chapter.'
          : 'Each stump is fitted to a bootstrap resample of the same ten points; circle size is how often a point was drawn. Seven bagged stumps stay at 70%, because averaging cannot remove the bias of a one-split model.'
      }
      legend={[
        {label: 'label +1', color: posColor},
        {label: 'label -1', color: negColor},
        {label: 'correctly voted by the ensemble', color: goodColor},
      ]}
      table={table}
      controls={
        <>
          <label className={s.control}>
            method
            <select
              className={s.select}
              value={mode}
              onChange={(e) => setMode(e.target.value as 'boost' | 'bag')}>
              <option value="boost">Boosting (AdaBoost)</option>
              <option value="bag">Bagging (bootstrap)</option>
            </select>
          </label>
          {mode === 'boost' ? (
            <label className={s.control}>
              rounds
              <input
                type="range"
                min={0}
                max={5}
                step={1}
                value={round}
                onChange={(e) => setRound(Number(e.target.value))}
              />
              <span className={s.value}>{round}</span>
            </label>
          ) : (
            <label className={s.control}>
              stumps
              <input
                type="range"
                min={1}
                max={7}
                step={1}
                value={stumps}
                onChange={(e) => setStumps(Number(e.target.value))}
              />
              <span className={s.value}>{stumps}</span>
            </label>
          )}
          <span className={s.value} aria-live="polite">
            {headline}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={headline}>
        <line className={s.axis} x1={x0 - 20} y1={lineY} x2={px(10) + 20} y2={lineY} />
        {stump && (
          <g>
            <line
              x1={px(stump.threshold)}
              y1={lineY - 60}
              x2={px(stump.threshold)}
              y2={lineY + 60}
              stroke="var(--text-strong)"
              strokeWidth={2}
              strokeDasharray="5 4"
            />
            <text className={s.dataLabel} x={px(stump.threshold)} y={lineY - 68} textAnchor="middle">
              {rule(stump)}
            </text>
          </g>
        )}
        {XS.map((v, j) => {
          const color = YS[j] === 1 ? posColor : negColor;
          const r = radius(sizes[j]);
          return (
            <g key={v}>
              {r > 0 && <circle cx={px(v)} cy={lineY} r={r} fill={color} opacity={0.35} stroke={color} strokeWidth={2} />}
              {r === 0 && <circle cx={px(v)} cy={lineY} r={3} fill="none" stroke={color} strokeWidth={1.5} />}
              <text className={s.tick} x={px(v)} y={lineY + 4} textAnchor="middle" fill="var(--text-strong)">
                {YS[j] === 1 ? '+' : '-'}
              </text>
              <text className={s.tick} x={px(v)} y={lineY + 78} textAnchor="middle">
                x = {v}
              </text>
              <text className={s.tick} x={px(v)} y={lineY + 92} textAnchor="middle">
                {mode === 'boost' ? sizes[j].toFixed(3) : `${Math.round(sizes[j] * XS.length)} draws`}
              </text>
            </g>
          );
        })}
        <text className={s.axisLabel} x={x0 - 20} y={lineY + 108} textAnchor="start">
          {mode === 'boost' ? 'weight used to fit this round' : 'times drawn into the latest resample'}
        </text>
        {score && (
          <g>
            <text className={s.axisLabel} x={x0 - 20} y={lineY + 140} textAnchor="start">
              {mode === 'boost' ? 'ensemble score (sum of alpha x vote)' : 'vote tally (sum of stump votes)'}
            </text>
            {XS.map((v, j) => {
              const right = Math.sign(score![j]) === YS[j];
              return (
                <g key={v}>
                  <rect
                    x={px(v) - 22}
                    y={lineY + 150}
                    width={44}
                    height={26}
                    rx={6}
                    fill={right ? goodColor : 'none'}
                    opacity={right ? 0.28 : 1}
                    stroke={right ? goodColor : negColor}
                    strokeWidth={1.5}
                  />
                  <text className={s.dataLabel} x={px(v)} y={lineY + 167} textAnchor="middle">
                    {score![j].toFixed(mode === 'boost' ? 2 : 0)}
                  </text>
                </g>
              );
            })}
            <text className={s.dataLabel} x={x0 - 20} y={lineY + 200} textAnchor="start">
              ensemble accuracy {(accuracy * 100).toFixed(0)}%
            </text>
          </g>
        )}
      </svg>
    </VizPanel>
  );
}
