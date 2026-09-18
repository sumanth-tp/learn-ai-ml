import {useCallback, useEffect, useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

/** Sutton & Barto's 5-state random walk: true values are 1/6 … 5/6. */
const N = 5;
const TRUE_VALUES = Array.from({length: N}, (_, i) => (i + 1) / (N + 1));
const W = 640;
const H = 240;
const PAD = {top: 16, right: 14, bottom: 32, left: 46};

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function rmse(values: number[]) {
  const total = values.reduce((acc, v, i) => acc + (v - TRUE_VALUES[i]) ** 2, 0);
  return Math.sqrt(total / N);
}

/** One episode of the random walk, returning the visited states and the reward. */
function episode(rng: () => number) {
  let state = 2;                       // start in the middle
  const visited: number[] = [];
  for (let step = 0; step < 200; step += 1) {
    visited.push(state);
    state += rng() < 0.5 ? -1 : 1;
    if (state < 0) return {visited, reward: 0};
    if (state >= N) return {visited, reward: 1};
  }
  return {visited, reward: 0};
}

function runCurves(alpha: number, episodes: number, seed: number) {
  const rngTd = mulberry(seed);
  const rngMc = mulberry(seed);
  const td = Array(N).fill(0.5);
  const mc = Array(N).fill(0.5);
  const curve: {episode: number; td: number; mc: number}[] = [
    {episode: 0, td: rmse(td), mc: rmse(mc)},
  ];

  for (let e = 1; e <= episodes; e += 1) {
    // TD(0): update after every step, bootstrapping from the next estimate
    const runTd = episode(rngTd);
    let state = 2;
    for (let i = 0; i < runTd.visited.length; i += 1) {
      const current = runTd.visited[i];
      const next = i + 1 < runTd.visited.length ? runTd.visited[i + 1] : null;
      const nextValue = next === null ? runTd.reward : td[next];
      td[current] += alpha * (nextValue - td[current]);
    }
    // Monte Carlo: wait for the return, then update every visited state
    const runMc = episode(rngMc);
    for (const visited of runMc.visited) {
      mc[visited] += alpha * (runMc.reward - mc[visited]);
    }
    curve.push({episode: e, td: rmse(td), mc: rmse(mc)});
  }
  return {curve, td, mc};
}

export default function TDvsMCLab() {
  const dark = useDarkViz();
  const [alpha, setAlpha] = useState(0.1);
  const [episodes, setEpisodes] = useState(100);
  const [seed, setSeed] = useState(3);

  const {curve, td, mc} = useMemo(
    () => runCurves(alpha, episodes, seed),
    [alpha, episodes, seed],
  );

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const yMax = 0.55;
  const x = (e: number) => PAD.left + (e / episodes) * innerW;
  const y = (v: number) => PAD.top + innerH - (Math.min(v, yMax) / yMax) * innerH;
  const path = (key: 'td' | 'mc') =>
    curve.map((p, i) => `${i ? 'L' : 'M'}${x(p.episode).toFixed(1)},${y(p[key]).toFixed(1)}`).join(' ');

  const tdColor = seriesColor(0, dark);
  const mcColor = seriesColor(1, dark);
  const final = curve[curve.length - 1];

  return (
    <VizPanel
      title="TD(0) against Monte Carlo on a random walk"
      hint="Both estimate the same values from the same episodes. TD updates every step by bootstrapping; MC waits for the return. Watch TD fall faster and sit lower — and raise α to see MC's variance turn into noise."
      legend={[
        {label: 'TD(0) — bootstraps each step', color: tdColor},
        {label: 'Monte Carlo — waits for the return', color: mcColor},
      ]}
      table={{
        columns: ['state', 'true value', 'TD estimate', 'MC estimate'],
        rows: TRUE_VALUES.map((t, i) => [
          String.fromCharCode(65 + i),
          t.toFixed(3),
          td[i].toFixed(3),
          mc[i].toFixed(3),
        ]),
      }}
      controls={
        <>
          <label className={s.control}>
            α
            <input type="range" min={0.01} max={0.5} step={0.01} value={alpha}
                   onChange={(e) => setAlpha(Number(e.target.value))} />
            <span className={s.value}>{alpha.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            episodes
            <input type="range" min={20} max={300} step={10} value={episodes}
                   onChange={(e) => setEpisodes(Number(e.target.value))} />
            <span className={s.value}>{episodes}</span>
          </label>
          <button type="button" className={s.button} onClick={() => setSeed((v) => v + 1)}>
            new run
          </button>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Root mean squared error against episodes for TD and Monte Carlo">
        {[0, 0.1, 0.2, 0.3, 0.4, 0.5].map((tick) => (
          <g key={tick}>
            <line className={s.grid} x1={PAD.left} y1={y(tick)} x2={W - PAD.right} y2={y(tick)} />
            <text className={s.tick} x={PAD.left - 6} y={y(tick) + 3} textAnchor="end">
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        <text className={s.axisLabel} x={PAD.left} y={PAD.top - 4}>RMS error</text>
        <text className={s.axisLabel} x={W / 2} y={H - 2} textAnchor="middle">episodes</text>
        <path d={path('mc')} fill="none" stroke={mcColor} strokeWidth={2} />
        <path d={path('td')} fill="none" stroke={tdColor} strokeWidth={2.5} />
      </svg>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0}}>
        <span>final RMS — TD <strong>{final.td.toFixed(3)}</strong></span>
        <span>MC <strong>{final.mc.toFixed(3)}</strong></span>
        <span>{final.td < final.mc ? 'TD is ahead after this many episodes' : 'MC has caught up'}</span>
      </div>
    </VizPanel>
  );
}
