import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const PAD = {top: 20, right: 24, bottom: 34, left: 40};

export const LECTURE: number[][] = [[2, 0], [4, 0], [10, 0], [12, 0], [3, 0], [11, 0], [5, 0]];
export const LECTURE_START = [[2, 0], [10, 0]];

export const BLOBS: number[][] = [[0.09, -1.68], [4.18, 1.72], [4.84, 2.42], [2.68, 4.14], [0.63, 5.24], [0.56, -0.14], [-0.72, -2.18], [-0.49, -1.39], [0.79, 1.54], [4.46, 1.12], [6.19, 0.11], [1.98, 4.11], [-1.07, -0.19], [-0.04, -0.43], [5.51, 1.2], [-1.12, -0.56], [5.12, 0.15], [1.17, 6.34], [-0.69, -0.21], [1.41, 4.48], [1.58, 5.72], [1.61, 0.39], [1.31, 5.49], [-0.83, -0.92], [5.81, 1.68], [4.3, 1.02], [1.92, 5.56], [-0.25, -0.32], [3.7, 1.52], [3.2, 0.79], [-0.07, -0.56], [5.39, 1.54], [3.23, 5.15], [-1.18, 0.8], [1.01, -0.12], [0.88, -0.99], [5.89, 1.9], [4.88, 2.42], [4.52, 0.53], [0.67, 1.78], [5.35, 2.55], [5.68, -0.49], [1.48, 4.81], [2.61, 4.16], [1.47, 4.27], [4.71, -0.39], [1.43, 4.89], [-0.92, 4.03], [2.87, 3.71], [1.7, 7.16], [3.88, 2.11], [1.34, 0.21], [-0.92, -0.64], [2.65, 5.31], [5.84, 2.07], [0.05, -0.36], [2.15, 6.17], [4.52, -0.53], [1.04, 5.38], [2.85, 5.49]];

export const PRESETS: {label: string; rows: number[]}[] = [
  {label: 'rows 30, 37, 49 (one per blob)', rows: [30, 37, 49]},
  {label: 'rows 0, 25, 27 (two in one blob)', rows: [0, 25, 27]},
  {label: 'rows 24, 31, 39 (a poor start)', rows: [24, 31, 39]},
];

export const ELBOW = [652.44, 345.74, 95.35, 79.43, 65.41, 52.99, 44.2, 36.86];

export type Frame = {centroids: number[][]; labels: number[] | null; wcss: number | null};

const sq = (a: number[], b: number[]) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

const close = (a: number[], b: number[]) =>
  Math.abs(a[0] - b[0]) <= 1e-8 + 1e-5 * Math.abs(b[0]) && Math.abs(a[1] - b[1]) <= 1e-8 + 1e-5 * Math.abs(b[1]);

export function lloyd(points: number[][], init: number[][]): Frame[] {
  const frames: Frame[] = [{centroids: init.map((c) => c.slice()), labels: null, wcss: null}];
  let centroids = init.map((c) => c.slice());
  for (let pass = 0; pass < 50; pass += 1) {
    const labels = points.map((p) => {
      let best = 0;
      for (let k = 1; k < centroids.length; k += 1) {
        if (sq(p, centroids[k]) < sq(p, centroids[best])) best = k;
      }
      return best;
    });
    const updated = centroids.map((old, k) => {
      const members = points.filter((_, i) => labels[i] === k);
      if (members.length === 0) return old;
      return [
        members.reduce((a, m) => a + m[0], 0) / members.length,
        members.reduce((a, m) => a + m[1], 0) / members.length,
      ];
    });
    const wcss = points.reduce((a, p, i) => a + sq(p, updated[labels[i]]), 0);
    frames.push({centroids: updated, labels, wcss});
    const settled = updated.every((c, k) => close(c, centroids[k]));
    centroids = updated;
    if (settled) break;
  }
  return frames;
}

type Dataset = 'lecture' | 'blobs';

export default function KMeansLab() {
  const dark = useDarkViz();
  const [dataset, setDataset] = useState<Dataset>('lecture');
  const [preset, setPreset] = useState(0);
  const [index, setIndex] = useState(0);

  const points = dataset === 'lecture' ? LECTURE : BLOBS;
  const frames = useMemo(
    () =>
      dataset === 'lecture'
        ? lloyd(LECTURE, LECTURE_START)
        : lloyd(BLOBS, PRESETS[preset].rows.map((r) => BLOBS[r])),
    [dataset, preset],
  );
  const last = frames.length - 1;
  const frame = frames[Math.min(index, last)];

  const xMin = dataset === 'lecture' ? 0 : -2;
  const xMax = dataset === 'lecture' ? 13 : 7;
  const yMin = dataset === 'lecture' ? -1 : -3;
  const yMax = dataset === 'lecture' ? 1 : 8;
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const x = (v: number) => PAD.left + ((v - xMin) / (xMax - xMin)) * innerW;
  const y = (v: number) => PAD.top + innerH - ((v - yMin) / (yMax - yMin)) * innerH;

  const axisY = dataset === 'lecture' ? 0 : yMin;
  const neutral = dark ? '#848c99' : '#9aa0a6';
  const colorOf = (label: number | null) => (label === null ? neutral : seriesColor(label, dark));

  const reset = (next: Dataset, nextPreset: number) => {
    setDataset(next);
    setPreset(nextPreset);
    setIndex(0);
  };

  const status =
    index === 0
      ? 'iteration 0: starting centroids, nothing assigned yet'
      : `iteration ${index} of ${last}: WCSS ${frame.wcss!.toFixed(2)}${index === last ? ' (converged)' : ''}`;

  const rows = points.map((p, i) => [
    i,
    p[0].toFixed(2),
    dataset === 'lecture' ? '-' : p[1].toFixed(2),
    frame.labels ? frame.labels[i] + 1 : '-',
  ]);

  const ticks =
    dataset === 'lecture' ? [0, 2, 4, 6, 8, 10, 12] : [-2, 0, 2, 4, 6];

  return (
    <VizPanel
      title="k-means, step by step"
      hint="Step: every point joins its nearest centroid, then each centroid moves to the mean of its points. On the lecture data it converges to {2,3,4,5} and {10,11,12} with centroids 3.5 and 11 and WCSS 7. On the 2-D blobs, try the three starts: a poor start gets stuck at a much larger WCSS."
      legend={[
        {label: 'cluster 1', color: seriesColor(0, dark)},
        {label: 'cluster 2', color: seriesColor(1, dark)},
        ...(dataset === 'blobs' ? [{label: 'cluster 3', color: seriesColor(2, dark)}] : []),
        {label: 'unassigned', color: neutral},
      ]}
      table={{columns: ['point', 'x', 'y', 'cluster'], rows}}
      controls={
        <>
          <label className={s.control}>
            data
            <select
              className={s.select}
              value={dataset}
              onChange={(e) => reset(e.target.value as Dataset, 0)}>
              <option value="lecture">Lecture, 1-D (k = 2)</option>
              <option value="blobs">2-D blobs (k = 3)</option>
            </select>
          </label>
          {dataset === 'blobs' && (
            <label className={s.control}>
              start
              <select
                className={s.select}
                value={preset}
                onChange={(e) => reset('blobs', Number(e.target.value))}>
                {PRESETS.map((p, i) => (
                  <option key={p.label} value={i}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button type="button" className={s.button} onClick={() => setIndex(Math.min(index + 1, last))} disabled={index >= last}>
            Step
          </button>
          <button type="button" className={s.button} onClick={() => setIndex(last)} disabled={index >= last}>
            Run to the end
          </button>
          <button type="button" className={s.button} onClick={() => setIndex(0)} disabled={index === 0}>
            Reset
          </button>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`k-means on ${dataset === 'lecture' ? 'the seven lecture points' : 'sixty 2-D points'}, ${status}`}>
        <line className={s.axis} x1={PAD.left} y1={y(axisY)} x2={W - PAD.right} y2={y(axisY)} />
        {dataset === 'blobs' && (
          <line className={s.axis} x1={x(xMin)} y1={PAD.top} x2={x(xMin)} y2={PAD.top + innerH} />
        )}
        {ticks.map((t) => (
          <text key={t} className={s.tick} x={x(t)} y={H - 14} textAnchor="middle">
            {t}
          </text>
        ))}
        {frames.slice(0, Math.min(index, last) + 1).map((f, fi, arr) =>
          fi === 0
            ? null
            : f.centroids.map((c, k) => (
                <line
                  key={`${fi}-${k}`}
                  x1={x(arr[fi - 1].centroids[k][0])}
                  y1={y(arr[fi - 1].centroids[k][1])}
                  x2={x(c[0])}
                  y2={y(c[1])}
                  stroke={seriesColor(k, dark)}
                  strokeWidth={2}
                  strokeDasharray="4 3"
                />
              )),
        )}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={x(p[0])}
            cy={dataset === 'lecture' ? y(0) - 24 : y(p[1])}
            r={dataset === 'lecture' ? 10 : 4.5}
            fill={colorOf(frame.labels ? frame.labels[i] : null)}
            opacity={0.85}
            stroke="var(--surface-raised)"
            strokeWidth={1.5}
          />
        ))}
        {dataset === 'lecture' &&
          points.map((p, i) => (
            <text key={`l${i}`} className={s.dataLabel} x={x(p[0])} y={y(0) - 20} textAnchor="middle" fill="#fff">
              {p[0]}
            </text>
          ))}
        {frame.centroids.map((c, k) => (
          <g key={k}>
            <rect
              x={x(c[0]) - 8}
              y={y(c[1]) - 8}
              width={16}
              height={16}
              transform={`rotate(45 ${x(c[0])} ${y(c[1])})`}
              fill={seriesColor(k, dark)}
              stroke="var(--text-strong)"
              strokeWidth={2}
            />
            <text className={s.dataLabel} x={x(c[0])} y={y(c[1]) + (dataset === 'lecture' ? 30 : -14)} textAnchor="middle">
              {dataset === 'lecture' ? c[0].toFixed(c[0] % 1 === 0 ? 0 : 1) : `(${c[0].toFixed(2)}, ${c[1].toFixed(2)})`}
            </text>
          </g>
        ))}
      </svg>
      {dataset === 'blobs' && (
        <p className={s.hint} style={{padding: '0.4rem 0 0'}}>
          WCSS for k = 1 to 8 from scikit-learn with 10 restarts: {ELBOW.map((v) => v.toFixed(2)).join(', ')}. The knee is at
          k = 3.
        </p>
      )}
    </VizPanel>
  );
}
