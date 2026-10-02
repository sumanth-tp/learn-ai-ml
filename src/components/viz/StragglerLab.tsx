import {useMemo, useState} from 'react';

import {mulberry32, normals} from './dist2Math';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const STEPS = 400;
const SIZES = [1, 4, 16, 64, 256];
const BACKUPS = [0, 1, 2, 4, 8, 16];

export function stepTimes(n: number, backups: number, sigma: number): number[] {
  const rnd = mulberry32(12345);
  const out: number[] = [];
  for (let t = 0; t < STEPS; t += 1) {
    const times = normals(rnd, n + backups)
      .map((z) => Math.exp(sigma * z))
      .sort((a, b) => a - b);
    out.push(times[n - 1]);
  }
  return out;
}

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;

export default function StragglerLab() {
  const dark = useDarkViz();
  const [n, setN] = useState(64);
  const [sigma, setSigma] = useState(0.5);
  const [backups, setBackups] = useState(0);

  const bars = useMemo(() => SIZES.map((m) => mean(stepTimes(m, backups, sigma))), [backups, sigma]);
  const plain = useMemo(() => SIZES.map((m) => mean(stepTimes(m, 0, sigma))), [sigma]);
  const samples = useMemo(() => stepTimes(n, backups, sigma), [n, backups, sigma]);
  const current = mean(samples);

  const lo = Math.min(...samples);
  const hi = Math.max(...samples);
  const BINS = 20;
  const span = Math.max(hi - lo, 1e-9);
  const counts = new Array<number>(BINS).fill(0);
  samples.forEach((v) => {
    counts[Math.min(BINS - 1, Math.floor(((v - lo) / span) * BINS))] += 1;
  });
  const maxCount = Math.max(...counts, 1);

  const barColor = seriesColor(0, dark);
  const pickColor = seriesColor(1, dark);
  const leftX0 = 44;
  const leftW = 260;
  const rightX0 = 366;
  const rightW = 250;
  const top = 38;
  const bottom = H - 52;
  const plotH = bottom - top;
  const yMax = Math.max(...bars, 1.2) * 1.1;

  const rows = SIZES.map((m, i) => [m, plain[i].toFixed(3), bars[i].toFixed(3)]);

  return (
    <VizPanel
      title="Waiting for the slowest worker"
      hint="Each worker's step time varies around 1. A synchronous step ends only when the slowest needed worker finishes, so more workers means a slower step. Add backup workers and the step ends when the fastest N have finished."
      legend={[
        {label: 'mean step time at each cluster size', color: barColor},
        {label: 'the cluster size you selected', color: pickColor},
      ]}
      table={{columns: ['workers', 'no backups', `with ${backups} backup workers`], rows}}
      controls={
        <>
          <label className={s.control}>
            workers
            <select className={s.select} value={n} onChange={(e) => setN(Number(e.target.value))}>
              {SIZES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            spread sigma
            <input
              type="range"
              min={0.1}
              max={0.8}
              step={0.05}
              value={sigma}
              onChange={(e) => setSigma(Number(e.target.value))}
            />
            <span className={s.value}>{sigma.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            backup workers
            <select className={s.select} value={backups} onChange={(e) => setBackups(Number(e.target.value))}>
              {BACKUPS.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <span className={s.value} aria-live="polite">
            mean step time {current.toFixed(3)}
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`With ${n} workers, spread ${sigma.toFixed(2)} and ${backups} backup workers the mean synchronous step takes ${current.toFixed(2)} times the median worker time`}>
        <text className={s.axisLabel} x={leftX0 + leftW / 2} y={20} textAnchor="middle">
          mean step time by cluster size
        </text>
        <line className={s.axis} x1={leftX0} y1={bottom} x2={leftX0 + leftW} y2={bottom} />
        <line className={s.axis} x1={leftX0} y1={top} x2={leftX0} y2={bottom} />
        {[1, Math.round(yMax / 2), Math.floor(yMax)].filter((v, i, a) => v > 0 && a.indexOf(v) === i).map((tick) => (
          <g key={tick}>
            <line className={s.grid} x1={leftX0} y1={bottom - (tick / yMax) * plotH} x2={leftX0 + leftW} y2={bottom - (tick / yMax) * plotH} />
            <text className={s.tick} x={leftX0 - 6} y={bottom - (tick / yMax) * plotH + 3} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        {bars.map((v, i) => {
          const bw = leftW / SIZES.length - 14;
          const bx = leftX0 + (i + 0.5) * (leftW / SIZES.length) - bw / 2;
          const h = (v / yMax) * plotH;
          return (
            <g key={SIZES[i]}>
              <rect x={bx} y={bottom - h} width={bw} height={h} fill={SIZES[i] === n ? pickColor : barColor} opacity={0.9} />
              <text className={s.dataLabel} x={bx + bw / 2} y={bottom - h - 5} textAnchor="middle">
                {v.toFixed(2)}
              </text>
              <text className={s.tick} x={bx + bw / 2} y={bottom + 14} textAnchor="middle">
                {SIZES[i]}
              </text>
            </g>
          );
        })}
        <text className={s.axisLabel} x={leftX0 + leftW / 2} y={H - 8} textAnchor="middle">
          workers
        </text>

        <text className={s.axisLabel} x={rightX0 + rightW / 2} y={20} textAnchor="middle">
          400 steps with {n} workers
        </text>
        <line className={s.axis} x1={rightX0} y1={bottom} x2={rightX0 + rightW} y2={bottom} />
        {counts.map((c, i) => {
          const bw = rightW / BINS;
          const h = (c / maxCount) * (plotH - 8);
          return <rect key={i} x={rightX0 + i * bw + 1} y={bottom - h} width={bw - 2} height={h} fill={pickColor} opacity={0.85} />;
        })}
        <line
          x1={rightX0 + ((current - lo) / span) * rightW}
          x2={rightX0 + ((current - lo) / span) * rightW}
          y1={top}
          y2={bottom}
          stroke="var(--text-strong)"
          strokeWidth={2}
          strokeDasharray="4 3"
        />
        <text className={s.tick} x={rightX0} y={bottom + 14} textAnchor="start">
          {lo.toFixed(2)}
        </text>
        <text className={s.tick} x={rightX0 + rightW} y={bottom + 14} textAnchor="end">
          {hi.toFixed(2)}
        </text>
        <text className={s.axisLabel} x={rightX0 + rightW / 2} y={H - 8} textAnchor="middle">
          step time (dashed line: mean {current.toFixed(2)})
        </text>
      </svg>
    </VizPanel>
  );
}
