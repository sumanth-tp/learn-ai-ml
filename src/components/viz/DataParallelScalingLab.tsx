import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const WORKERS = [1, 2, 4, 8, 16, 64, 256];
const BATCHES = [8, 16, 32, 64, 128];
const GRAD_MB = [10, 100, 500, 2000];
const BANDWIDTH = [1, 10, 50, 100];
const BASE_LR = 0.1;

export type Scaling = {compute: number; comm: number; step: number; speedup: number; efficiency: number};

export function scaling(
  k: number,
  batch: number,
  msPerSample: number,
  gradMb: number,
  gbPerS: number,
  overlap: number,
): Scaling {
  const compute = batch * msPerSample;
  const comm = k === 1 ? 0 : (((2 * (k - 1)) / k) * (gradMb / 1000)) / gbPerS * 1000;
  const step = compute + (1 - overlap) * comm;
  const speedup = (k * compute) / step;
  return {compute, comm, step, speedup, efficiency: speedup / k};
}

export default function DataParallelScalingLab() {
  const dark = useDarkViz();
  const [k, setK] = useState(8);
  const [batch, setBatch] = useState(32);
  const [ms, setMs] = useState(5);
  const [mb, setMb] = useState(100);
  const [bw, setBw] = useState(10);
  const [overlap, setOverlap] = useState(0);

  const now = scaling(k, batch, ms, mb, bw, overlap);
  const curve = WORKERS.map((w) => scaling(w, batch, ms, mb, bw, overlap));
  const idealColor = seriesColor(1, dark);
  const lineColor = seriesColor(0, dark);

  const left = 64;
  const right = W - 24;
  const top = 30;
  const bottom = H - 48;
  const lx = (w: number) => left + (Math.log2(w) / Math.log2(256)) * (right - left);
  const ly = (v: number) => bottom - (Math.log2(v) / Math.log2(256)) * (bottom - top);
  const yTicks = [1, 4, 16, 64, 256];

  const rows = WORKERS.map((w, i) => [
    w,
    curve[i].compute.toFixed(1),
    curve[i].comm.toFixed(2),
    curve[i].step.toFixed(2),
    curve[i].speedup.toFixed(3),
    curve[i].efficiency.toFixed(3),
  ]);

  return (
    <VizPanel
      title="Data-parallel scaling calculator"
      hint="Illustrative parameters, not measured hardware. The defaults are 8 workers, local batch 32, 5 ms of compute per sample, a 100 MB gradient and a 10 GB/s ring with no overlap: step 177.50 ms, speedup 7.211, efficiency 0.901, as in the chapter's code. Raise the overlap or the bandwidth, or grow the gradient, and watch the gap to the ideal line move."
      legend={[
        {label: 'speedup with communication', color: lineColor},
        {label: 'ideal linear speedup', color: idealColor},
      ]}
      table={{columns: ['workers', 'compute ms', 'comm ms', 'step ms', 'speedup', 'efficiency'], rows}}
      controls={
        <>
          <label className={s.control}>
            workers
            <select className={s.select} value={k} onChange={(e) => setK(Number(e.target.value))}>
              {WORKERS.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            local batch
            <select className={s.select} value={batch} onChange={(e) => setBatch(Number(e.target.value))}>
              {BATCHES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            ms per sample
            <input type="range" min={1} max={20} step={1} value={ms} onChange={(e) => setMs(Number(e.target.value))} />
            <span className={s.value}>{ms}</span>
          </label>
          <label className={s.control}>
            gradient MB
            <select className={s.select} value={mb} onChange={(e) => setMb(Number(e.target.value))}>
              {GRAD_MB.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            link GB/s
            <select className={s.select} value={bw} onChange={(e) => setBw(Number(e.target.value))}>
              {BANDWIDTH.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            overlap
            <input
              type="range"
              min={0}
              max={0.95}
              step={0.05}
              value={overlap}
              onChange={(e) => setOverlap(Number(e.target.value))}
            />
            <span className={s.value}>{overlap.toFixed(2)}</span>
          </label>
          <span className={s.value} aria-live="polite">
            global batch {k * batch}, learning rate x{k} ({(BASE_LR * k).toFixed(1)} from base {BASE_LR}), step {now.step.toFixed(2)} ms,
            speedup {now.speedup.toFixed(3)}, efficiency {now.efficiency.toFixed(3)}
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Speedup of ${now.speedup.toFixed(2)} on ${k} workers, efficiency ${now.efficiency.toFixed(3)}`}>
        <line className={s.axis} x1={left} y1={bottom} x2={right} y2={bottom} />
        <line className={s.axis} x1={left} y1={top} x2={left} y2={bottom} />
        {yTicks.map((t) => (
          <g key={t}>
            <line className={s.grid} x1={left} y1={ly(t)} x2={right} y2={ly(t)} />
            <text className={s.tick} x={left - 6} y={ly(t) + 3} textAnchor="end">
              {t}x
            </text>
          </g>
        ))}
        {WORKERS.map((w) => (
          <text key={w} className={s.tick} x={lx(w)} y={bottom + 14} textAnchor="middle">
            {w}
          </text>
        ))}
        <path
          d={WORKERS.map((w, i) => `${i ? 'L' : 'M'}${lx(w).toFixed(1)},${ly(w).toFixed(1)}`).join(' ')}
          fill="none"
          stroke={idealColor}
          strokeWidth={2}
          strokeDasharray="6 4"
        />
        <path
          d={WORKERS.map((w, i) => `${i ? 'L' : 'M'}${lx(w).toFixed(1)},${ly(curve[i].speedup).toFixed(1)}`).join(' ')}
          fill="none"
          stroke={lineColor}
          strokeWidth={2.6}
        />
        <circle cx={lx(k)} cy={ly(now.speedup)} r={5} fill={lineColor} stroke="var(--surface-raised)" strokeWidth={2} />
        <text className={s.dataLabel} x={Math.min(lx(k) + 8, right - 56)} y={ly(now.speedup) - 10}>
          {now.speedup.toFixed(3)}
        </text>
        <text className={s.axisLabel} x={(left + right) / 2} y={H - 8} textAnchor="middle">
          workers (log scale)
        </text>
        <text className={s.axisLabel} x={left} y={18} textAnchor="start">
          speedup over one worker (log scale)
        </text>
      </svg>
    </VizPanel>
  );
}
