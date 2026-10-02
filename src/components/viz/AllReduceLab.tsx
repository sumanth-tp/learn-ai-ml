import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 330;
const WORKER_CHOICES = [2, 3, 4, 5, 6, 8];

export type RingFrame = {grid: number[][]; sent: [number, number][]; received: [number, number][]; label: string};

export function ringFrames(n: number): RingFrame[] {
  let grid = Array.from({length: n}, (_, w) => Array.from({length: n}, (_, c) => w + 1 + 10 * c));
  const frames: RingFrame[] = [{grid: grid.map((r) => r.slice()), sent: [], received: [], label: 'start: every worker holds its own gradient'}];
  for (let step = 1; step <= 2 * (n - 1); step += 1) {
    const reduce = step <= n - 1;
    const k = reduce ? step - 1 : step - n;
    const next = grid.map((r) => r.slice());
    const sent: [number, number][] = [];
    const received: [number, number][] = [];
    for (let w = 0; w < n; w += 1) {
      const c = reduce ? (((w - k) % n) + n) % n : (((w + 1 - k) % n) + n) % n;
      const to = (w + 1) % n;
      sent.push([w, c]);
      received.push([to, c]);
      next[to][c] = reduce ? grid[to][c] + grid[w][c] : grid[w][c];
    }
    grid = next;
    frames.push({
      grid: grid.map((r) => r.slice()),
      sent,
      received,
      label: reduce
        ? `reduce-scatter step ${k + 1} of ${n - 1}: pass a chunk to the next worker, which adds it`
        : `all-gather step ${k + 1} of ${n - 1}: pass a finished chunk to the next worker, which keeps it`,
    });
  }
  return frames;
}

const TABLE_SIZES = [2, 4, 8, 16, 64];

export default function AllReduceLab() {
  const dark = useDarkViz();
  const [n, setN] = useState(4);
  const [step, setStep] = useState(6);

  const frames = useMemo(() => ringFrames(n), [n]);
  const last = 2 * (n - 1);
  const shown = Math.min(step, last);
  const frame = frames[shown];
  const sentColor = seriesColor(1, dark);
  const recvColor = seriesColor(0, dark);
  const doneColor = DIVERGING[dark ? 'dark' : 'light'].positive;

  const left = 96;
  const top = 64;
  const cw = Math.min(64, (W - left - 24) / n);
  const ch = Math.min(34, (H - top - 40) / n);

  const isSent = (w: number, c: number) => frame.sent.some(([a, b]) => a === w && b === c);
  const isRecv = (w: number, c: number) => frame.received.some(([a, b]) => a === w && b === c);
  const finished = shown === last;

  const rows = TABLE_SIZES.map((m) => [
    m,
    2 * (m - 1),
    (2 * (m - 1) / m).toFixed(3),
    m.toFixed(1),
  ]);

  return (
    <VizPanel
      title="Ring all-reduce, step by step"
      hint="Each worker holds one number per chunk. In the first half every worker passes a chunk to its neighbour, which adds it; in the second half the finished sums travel round the ring. With four workers it takes 6 steps and every worker ends holding 10, 50, 90, 130, the same as the chapter's code."
      legend={[
        {label: 'sent this step', color: sentColor},
        {label: 'received this step', color: recvColor},
        {label: 'final sum on every worker', color: doneColor},
      ]}
      table={{columns: ['workers', 'steps 2(N-1)', 'sent per worker, gradient = 1', 'parameter server inbound'], rows}}
      controls={
        <>
          <label className={s.control}>
            workers
            <select
              className={s.select}
              value={n}
              onChange={(e) => {
                const v = Number(e.target.value);
                setN(v);
                setStep(2 * (v - 1));
              }}>
              {WORKER_CHOICES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            step
            <input type="range" min={0} max={last} step={1} value={shown} onChange={(e) => setStep(Number(e.target.value))} />
            <span className={s.value}>
              {shown} of {last}
            </span>
          </label>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Ring all-reduce with ${n} workers after step ${shown} of ${last}. ${frame.label}`}>
        <text className={s.axisLabel} x={W / 2} y={22} textAnchor="middle">
          {frame.label}
        </text>
        <text className={s.axisLabel} x={left + (cw * n) / 2} y={50} textAnchor="middle">
          chunks of the gradient
        </text>
        {Array.from({length: n}, (_, c) => (
          <text key={c} className={s.tick} x={left + c * cw + cw / 2} y={top - 4} textAnchor="middle">
            chunk {c}
          </text>
        ))}
        {frame.grid.map((row, w) => (
          <g key={w}>
            <text className={s.tick} x={left - 8} y={top + 8 + w * ch + ch / 2} textAnchor="end">
              worker {w}
            </text>
            {row.map((value, c) => {
              const sent = isSent(w, c);
              const recv = isRecv(w, c);
              return (
                <g key={c}>
                  <rect
                    x={left + c * cw + 2}
                    y={top + 8 + w * ch + 2}
                    width={cw - 4}
                    height={ch - 4}
                    rx={4}
                    fill={finished ? doneColor : recv ? recvColor : 'var(--surface-1)'}
                    fillOpacity={finished ? 0.35 : recv ? 0.35 : 1}
                    stroke={sent ? sentColor : 'var(--border, #999)'}
                    strokeWidth={sent ? 2.6 : 1}
                  />
                  <text className={s.dataLabel} x={left + c * cw + cw / 2} y={top + 8 + w * ch + ch / 2 + 4} textAnchor="middle">
                    {value}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
        <text className={s.axisLabel} x={W / 2} y={H - 10} textAnchor="middle">
          {finished
            ? `finished in ${last} steps = 2(N-1); each worker sent ${(2 * (n - 1) / n).toFixed(3)} gradients' worth of data`
            : `${last - shown} steps to go`}
        </text>
      </svg>
    </VizPanel>
  );
}
