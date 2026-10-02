import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 320;
const PAD = {top: 16, right: 16, bottom: 36, left: 44};
const MAX_ROUNDS = 300;
const RATES = [1, 0.3, 0.1, 0.03];
const CHECKPOINTS = [1, 5, 20, 100, 300];

export const XS = [0.016, 0.099, 0.17, 0.202, 0.246, 0.312, 0.351, 0.504, 0.545, 0.633, 0.746, 0.811, 0.902, 1.054, 1.191, 1.197, 1.363, 1.384, 1.436, 1.619, 1.792, 1.798, 1.861, 1.931, 2.017, 2.027, 2.147, 2.191, 2.302, 2.334, 2.35, 2.427, 2.536, 2.551, 2.642, 2.702, 2.76, 2.915, 2.999, 3.152, 3.249, 3.262, 3.429, 3.482, 3.566, 3.64, 3.692, 3.721, 3.739, 3.775, 3.822, 3.883, 3.903, 4.024, 4.032, 4.113, 4.131, 4.329, 4.377, 4.378, 4.723, 4.778, 4.88, 4.895, 4.996, 5.144, 5.179, 5.259, 5.337, 5.342, 5.477, 5.563, 5.604, 5.61, 5.653, 5.694, 5.728, 5.885, 5.971, 5.983];
export const YS = [0.053, 0.746, 0.46, 0.326, 0.485, 0.437, 0.46, 1.14, 1.162, 1.389, 0.883, 1.473, 0.924, 1.253, 0.611, 0.53, 0.635, 0.817, 0.571, 0.358, 0.247, 0.236, 0.762, 0.948, 0.746, 0.529, 1.135, 0.636, 0.858, 1.207, 0.496, 1.172, 0.909, 1.074, 0.959, 0.96, 1.001, 0.349, 0.704, 0.155, -0.054, -0.005, -0.467, -0.549, -0.871, -1.333, -1.055, -1.233, -1.406, -1.0, -1.217, -1.33, -1.329, -0.943, -0.914, -0.608, -0.926, -0.463, -0.326, -0.388, -1.092, -0.2, -0.463, -0.451, -0.537, -0.677, -0.728, -0.978, -1.436, -0.992, -1.253, -0.805, -1.147, -1.053, -1.276, -1.174, -1.028, -1.224, -0.636, -0.714];

const GRID = Array.from({length: 200}, (_, i) => (6 * i) / 199);
const truthAt = (x: number) => Math.sin(x) + 0.5 * Math.sin(3 * x);
const TRUTH = GRID.map(truthAt);

type Split = {threshold: number; left: number; right: number};

export function fitStump(xs: number[], residual: number[]): Split {
  const n = xs.length;
  const total = residual.reduce((a, b) => a + b, 0);
  let sumLeft = 0;
  let best: {score: number; at: number} | null = null;
  for (let i = 0; i < n - 1; i += 1) {
    sumLeft += residual[i];
    if (xs[i] === xs[i + 1]) continue;
    const nl = i + 1;
    const nr = n - nl;
    const sumRight = total - sumLeft;
    const score = (sumLeft * sumLeft) / nl + (sumRight * sumRight) / nr;
    if (best === null || score > best.score + 1e-12) best = {score, at: i};
  }
  const at = (best as {score: number; at: number}).at;
  let sl = 0;
  for (let i = 0; i <= at; i += 1) sl += residual[i];
  return {
    threshold: (xs[at] + xs[at + 1]) / 2,
    left: sl / (at + 1),
    right: (total - sl) / (n - at - 1),
  };
}

export type Run = {trainRmse: number[]; truthRmse: number[]; fit: number[][]};

const rmse = (a: number[], b: number[]) =>
  Math.sqrt(a.reduce((acc, v, i) => acc + (v - b[i]) ** 2, 0) / a.length);

export function boost(rate: number, rounds: number): Run {
  const start = YS.reduce((a, b) => a + b, 0) / YS.length;
  const predTrain = YS.map(() => start);
  const predGrid = GRID.map(() => start);
  const trainRmse = [rmse(YS, predTrain)];
  const truthRmse = [rmse(TRUTH, predGrid)];
  const fit = [predGrid.slice()];
  for (let m = 0; m < rounds; m += 1) {
    const residual = YS.map((y, i) => y - predTrain[i]);
    const stump = fitStump(XS, residual);
    for (let i = 0; i < XS.length; i += 1) {
      predTrain[i] += rate * (XS[i] <= stump.threshold ? stump.left : stump.right);
    }
    for (let i = 0; i < GRID.length; i += 1) {
      predGrid[i] += rate * (GRID[i] <= stump.threshold ? stump.left : stump.right);
    }
    trainRmse.push(rmse(YS, predTrain));
    truthRmse.push(rmse(TRUTH, predGrid));
    fit.push(predGrid.slice());
  }
  return {trainRmse, truthRmse, fit};
}

export default function GradientBoostingLab() {
  const dark = useDarkViz();
  const [rounds, setRounds] = useState(100);
  const [rate, setRate] = useState(0.1);

  const run = useMemo(() => boost(rate, MAX_ROUNDS), [rate]);

  const pointColor = seriesColor(0, dark);
  const fitColor = seriesColor(1, dark);
  const truthColor = seriesColor(2, dark);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const x = (v: number) => PAD.left + (v / 6) * innerW;
  const y = (v: number) => PAD.top + innerH / 2 - (v / 1.9) * (innerH / 2);

  const current = run.fit[rounds];
  const train = run.trainRmse[rounds];
  const vsTruth = run.truthRmse[rounds];

  const path = (values: number[]) =>
    values.map((v, i) => `${i ? 'L' : 'M'}${x(GRID[i]).toFixed(1)},${y(v).toFixed(1)}`).join(' ');

  const rows = CHECKPOINTS.map((m) => [
    m,
    run.trainRmse[m].toFixed(4),
    run.truthRmse[m].toFixed(4),
  ]);

  return (
    <VizPanel
      title="Gradient boosting with depth-1 trees"
      hint="Each round fits a stump to the residuals and adds a fraction of it. With a learning rate of 1.0 the error against the true curve bottoms out around 100 rounds and then rises as noise is fitted; with 0.1 it keeps improving to 300."
      legend={[
        {label: 'training points', color: pointColor},
        {label: 'boosted fit', color: fitColor},
        {label: 'true curve', color: truthColor},
      ]}
      table={{columns: ['rounds', 'train RMSE', 'RMSE against the true curve'], rows}}
      controls={
        <>
          <label className={s.control}>
            rounds
            <input
              type="range"
              min={0}
              max={MAX_ROUNDS}
              step={1}
              value={rounds}
              onChange={(e) => setRounds(Number(e.target.value))}
            />
            <span className={s.value}>{rounds}</span>
          </label>
          <label className={s.control}>
            learning rate
            <select className={s.select} value={rate} onChange={(e) => setRate(Number(e.target.value))}>
              {RATES.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </label>
          <span className={s.value} aria-live="polite">
            train RMSE {train.toFixed(4)}, against truth {vsTruth.toFixed(4)}
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Boosted fit after ${rounds} rounds at learning rate ${rate}`}>
        <line className={s.axis} x1={PAD.left} y1={y(0)} x2={W - PAD.right} y2={y(0)} />
        {[0, 1, 2, 3, 4, 5, 6].map((tick) => (
          <text key={tick} className={s.tick} x={x(tick)} y={H - 18} textAnchor="middle">
            {tick}
          </text>
        ))}
        {[-1, 0, 1].map((tick) => (
          <text key={tick} className={s.tick} x={PAD.left - 8} y={y(tick) + 3} textAnchor="end">
            {tick}
          </text>
        ))}
        <path d={path(TRUTH)} fill="none" stroke={truthColor} strokeWidth={2} strokeDasharray="6 4" />
        {XS.map((v, i) => (
          <circle key={i} cx={x(v)} cy={y(YS[i])} r={3} fill={pointColor} opacity={0.75} />
        ))}
        <path d={path(current)} fill="none" stroke={fitColor} strokeWidth={2.6} />
        <text className={s.axisLabel} x={W / 2} y={H - 2} textAnchor="middle">
          x
        </text>
      </svg>
    </VizPanel>
  );
}
