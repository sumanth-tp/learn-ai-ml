import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const PAD = {top: 20, right: 24, bottom: 40, left: 52};
const MAX_WORKERS = 32;

export function makespanSeconds(rows: number, rate: number, workers: number, share: number, chunk: number | null): number {
  let biggest = share * rows;
  if (chunk !== null) biggest = Math.min(biggest, chunk);
  return Math.max(rows / (rate * workers), biggest / rate);
}

const fmt = (v: number) => v.toLocaleString('en-GB');
const CHUNKS: {label: string; value: number | null}[] = [
  {label: 'none', value: null},
  {label: '20 million', value: 20_000_000},
  {label: '5 million', value: 5_000_000},
  {label: '1 million', value: 1_000_000},
];

export default function BatchWindowLab() {
  const dark = useDarkViz();
  const [rows, setRows] = useState(50_000_000);
  const [rate, setRate] = useState(2000);
  const [workers, setWorkers] = useState(4);
  const [windowHours, setWindowHours] = useState(2);
  const [share, setShare] = useState(0.3);
  const [chunkIndex, setChunkIndex] = useState(0);

  const chunk = CHUNKS[chunkIndex].value;
  const spanH = makespanSeconds(rows, rate, workers, share, chunk) / 3600;
  const idealH = rows / (rate * workers) / 3600;
  const needed = Math.ceil(rows / (rate * windowHours * 3600));
  const fits = spanH <= windowHours + 1e-9;
  const utilisation = idealH / spanH;

  const curve = Array.from({length: MAX_WORKERS}, (_, i) => makespanSeconds(rows, rate, i + 1, share, chunk) / 3600);
  const plain = Array.from({length: MAX_WORKERS}, (_, i) => makespanSeconds(rows, rate, i + 1, share, null) / 3600);
  const yMax = Math.max(windowHours * 1.3, Math.min(Math.max(...plain), 40));
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const x = (w: number) => PAD.left + ((w - 1) / (MAX_WORKERS - 1)) * innerW;
  const y = (h: number) => PAD.top + innerH - (Math.min(h, yMax) / yMax) * innerH;
  const path = (values: number[]) => values.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i + 1).toFixed(1)},${y(v).toFixed(1)}`).join(' ');

  const line = seriesColor(0, dark);
  const second = seriesColor(1, dark);
  const neutral = dark ? '#848c99' : '#9aa0a6';
  const status = `${spanH.toFixed(2)} h with ${workers} workers: ${fits ? 'fits' : 'does not fit'} a ${windowHours} h window; ${needed} workers needed with a perfect split; utilisation ${(utilisation * 100).toFixed(0)}%`;

  const tableRows = [1, 2, 4, 8, 16, 32].map((w) => {
    const a = makespanSeconds(rows, rate, w, share, null) / 3600;
    const b = makespanSeconds(rows, rate, w, share, 5_000_000) / 3600;
    return [w, a.toFixed(2), b.toFixed(2)];
  });

  return (
    <VizPanel
      title="Will the nightly batch finish inside its window"
      hint="Add workers and watch the time fall until it hits the longest single partition, then stop. Cut big partitions into chunks and the curve keeps falling. Defaults match block 3: 50 million rows, 4 workers, a 2 hour window and a partition holding 30% of the rows take 2.08 h and miss the window; chunking at 5 million rows takes 1.74 h and fits. The 2,000 rows per second per worker is an assumed rate, so replace it with one you measure."
      legend={[
        {label: 'makespan with current chunking', color: line},
        {label: 'makespan as partitioned', color: second},
        {label: 'window', color: neutral},
      ]}
      table={{columns: ['workers', 'hours as partitioned', 'hours chunked at 5 million'], rows: tableRows}}
      controls={
        <>
          <label className={s.control}>
            rows per night
            <select className={s.select} value={rows} onChange={(e) => setRows(Number(e.target.value))}>
              {[5_000_000, 20_000_000, 50_000_000, 200_000_000].map((v) => (
                <option key={v} value={v}>{fmt(v)}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            rows/s per worker
            <select className={s.select} value={rate} onChange={(e) => setRate(Number(e.target.value))}>
              {[500, 2000, 10000].map((v) => (
                <option key={v} value={v}>{fmt(v)}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            workers
            <input type="range" min={1} max={MAX_WORKERS} step={1} value={workers} onChange={(e) => setWorkers(Number(e.target.value))} />
            <span className={s.value}>{workers}</span>
          </label>
          <label className={s.control}>
            window (hours)
            <input type="range" min={1} max={12} step={0.5} value={windowHours} onChange={(e) => setWindowHours(Number(e.target.value))} />
            <span className={s.value}>{windowHours.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            largest partition share
            <input type="range" min={0} max={0.6} step={0.05} value={share} onChange={(e) => setShare(Number(e.target.value))} />
            <span className={s.value}>{share.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            chunk size
            <select className={s.select} value={chunkIndex} onChange={(e) => setChunkIndex(Number(e.target.value))}>
              {CHUNKS.map((c, i) => (
                <option key={c.label} value={i}>{c.label}</option>
              ))}
            </select>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Makespan against workers. ${status}`}>
        <line className={s.axis} x1={PAD.left} y1={y(0)} x2={W - PAD.right} y2={y(0)} />
        <line className={s.axis} x1={PAD.left} y1={y(0)} x2={PAD.left} y2={PAD.top} />
        {[0, 0.5, 1].map((t) => (
          <text key={t} className={s.tick} x={PAD.left - 8} y={y(t * yMax) + 4} textAnchor="end">
            {(t * yMax).toFixed(1)}
          </text>
        ))}
        {[1, 4, 8, 16, 24, 32].map((w) => (
          <text key={w} className={s.tick} x={x(w)} y={H - 18} textAnchor="middle">
            {w}
          </text>
        ))}
        <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 2} textAnchor="middle">workers</text>
        <text className={s.axisLabel} x={14} y={PAD.top + innerH / 2} textAnchor="middle" transform={`rotate(-90 14 ${PAD.top + innerH / 2})`}>hours</text>
        <line x1={PAD.left} y1={y(windowHours)} x2={W - PAD.right} y2={y(windowHours)} stroke={neutral} strokeDasharray="5 4" />
        {chunk !== null && <path d={path(plain)} fill="none" stroke={second} strokeWidth={2} strokeDasharray="6 3" />}
        <path d={path(curve)} fill="none" stroke={line} strokeWidth={2.5} />
        <circle cx={x(workers)} cy={y(spanH)} r={5} fill={line} stroke="var(--surface-raised)" strokeWidth={2} />
        <text className={s.dataLabel} x={Math.min(x(workers) + 8, W - PAD.right - 60)} y={y(spanH) - 10}>
          {spanH.toFixed(2)} h
        </text>
      </svg>
    </VizPanel>
  );
}
