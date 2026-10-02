import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;

export const CLOUD: number[][] = [[-0.19, -0.42], [1.34, -0.67], [-3.38, -1.29], [0.36, 0.67], [-0.34, -0.53], [-1.59, -1.12], [0.77, -0.02], [-0.75, -0.5], [-0.78, -0.82], [0.01, -0.52], [-1.49, -0.76], [-0.67, -0.15], [1.53, 1.36], [-3.1, -2.79], [3.35, 1.06], [-1.47, -0.79], [-2.03, -0.76], [-0.43, -0.08], [0.07, 0.56], [2.04, 0.94], [-0.88, 0.56], [1.59, 0.29], [0.71, 0.99], [0.06, 0.82], [2.91, 2.38], [-1.4, -1.67], [-0.58, 0.39], [-0.41, 0.36], [-4.14, -2.26], [0.68, -0.06], [-1.06, -0.74], [0.33, 0.13], [-0.84, -1.12], [3.25, 0.45], [-2.07, -1.17], [-2.58, -1.52], [1.15, 0.96], [0.46, -0.48], [1.06, 1.67], [-0.71, -0.17], [0.65, -0.03], [-1.5, -0.94], [1.33, 0.7], [1.5, 0.99], [-1.72, -1.5], [-2.27, -1.73], [-0.05, -0.52], [0.37, 0.24], [0.93, 0.12], [1.37, 0.31], [2.76, 1.46], [-0.92, 0.0], [-1.75, 0.44], [-0.43, -0.52], [-2.93, -2.34], [-1.41, -1.39], [-0.1, -1.23], [-1.5, -0.97], [1.22, 1.71], [-1.32, -0.75]];

export const WINE_RAW = [0.9981, 0.0017, 0.0001, 0.0001, 0, 0, 0, 0, 0, 0, 0, 0, 0];
export const WINE_SCALED = [0.362, 0.1921, 0.1112, 0.0707, 0.0656, 0.0494, 0.0424, 0.0268, 0.0222, 0.0193, 0.0174, 0.013, 0.008];

type Mode = 'variance' | 'rotate' | 'wine';

export function varianceShares(eigenvalues: number[]): number[] {
  const total = eigenvalues.reduce((a, b) => a + b, 0);
  return eigenvalues.map((v) => (total > 0 ? v / total : 0));
}

export function cloudStats() {
  const n = CLOUD.length;
  const mx = CLOUD.reduce((a, p) => a + p[0], 0) / n;
  const my = CLOUD.reduce((a, p) => a + p[1], 0) / n;
  const centred = CLOUD.map((p) => [p[0] - mx, p[1] - my]);
  const sxx = centred.reduce((a, p) => a + p[0] * p[0], 0) / (n - 1);
  const syy = centred.reduce((a, p) => a + p[1] * p[1], 0) / (n - 1);
  const sxy = centred.reduce((a, p) => a + p[0] * p[1], 0) / (n - 1);
  const mid = (sxx + syy) / 2;
  const radius = Math.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy);
  const first = mid + radius;
  const second = mid - radius;
  const angle = (0.5 * Math.atan2(2 * sxy, sxx - syy) * 180) / Math.PI;
  return {mx, my, centred, total: sxx + syy, first, second, angle};
}

export function varianceAlong(centred: number[][], degrees: number): number {
  const a = (degrees * Math.PI) / 180;
  const c = Math.cos(a);
  const sn = Math.sin(a);
  const t = centred.map((p) => p[0] * c + p[1] * sn);
  const mean = t.reduce((acc, v) => acc + v, 0) / t.length;
  return t.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (t.length - 1);
}

export default function PcaLab() {
  const dark = useDarkViz();
  const [mode, setMode] = useState<Mode>('variance');
  const [eig, setEig] = useState([6.2, 2.4, 1.0, 0.4]);
  const [keep, setKeep] = useState(2);
  const [angle, setAngle] = useState(0);
  const [scaled, setScaled] = useState(false);

  const stats = useMemo(() => cloudStats(), []);

  const keepColor = seriesColor(0, dark);
  const dropColor = dark ? '#848c99' : '#9aa0a6';
  const accent = seriesColor(1, dark);
  const good = DIVERGING[dark ? 'dark' : 'light'].positive;

  const shares = varianceShares(eig);
  const cumulative = shares.map((_, i) => shares.slice(0, i + 1).reduce((a, b) => a + b, 0));
  const retained = cumulative[keep - 1];

  const wine = scaled ? WINE_SCALED : WINE_RAW;
  const wineFirstTwo = wine[0] + wine[1];

  const along = varianceAlong(stats.centred, angle);
  const lost = stats.total - along;

  let headline = '';
  if (mode === 'variance') {
    headline = `keeping ${keep} of 4 components retains ${(retained * 100).toFixed(1)}% of the variance`;
  } else if (mode === 'rotate') {
    headline = `axis at ${angle.toFixed(1)} degrees: variance ${along.toFixed(3)} of ${stats.total.toFixed(3)} (${((along / stats.total) * 100).toFixed(1)}%)`;
  } else {
    headline = `${scaled ? 'standardised' : 'raw'} wine features: PC1 explains ${(wine[0] * 100).toFixed(2)}%, first two ${(wineFirstTwo * 100).toFixed(1)}%`;
  }

  const table =
    mode === 'variance'
      ? {
          columns: ['component', 'eigenvalue', 'share', 'cumulative'],
          rows: eig.map((v, i) => [
            `PC${i + 1}`,
            v.toFixed(1),
            `${(shares[i] * 100).toFixed(1)}%`,
            `${(cumulative[i] * 100).toFixed(1)}%`,
          ]),
        }
      : mode === 'rotate'
        ? {
            columns: ['quantity', 'value'],
            rows: [
              ['angle (degrees)', angle.toFixed(1)],
              ['variance along the axis', along.toFixed(3)],
              ['variance lost', lost.toFixed(3)],
              ['total variance', stats.total.toFixed(3)],
              ['eigenvalue 1 (PC1)', stats.first.toFixed(3)],
              ['eigenvalue 2 (PC2)', stats.second.toFixed(3)],
              ['PC1 angle (degrees)', stats.angle.toFixed(1)],
            ],
          }
        : {
            columns: ['component', 'raw', 'standardised'],
            rows: WINE_RAW.map((v, i) => [
              `PC${i + 1}`,
              `${(v * 100).toFixed(2)}%`,
              `${(WINE_SCALED[i] * 100).toFixed(2)}%`,
            ]),
          };

  const setEigAt = (i: number, v: number) => setEig(eig.map((e, j) => (j === i ? v : e)));

  const barChart = (values: number[], highlight: number, line: number[] | null, max: number, labels: string[]) => {
    const left = 44;
    const right = W - 20;
    const top = 28;
    const bottom = H - 48;
    const slot = (right - left) / values.length;
    const barW = Math.min(slot * 0.62, 56);
    return (
      <>
        <line className={s.axis} x1={left} y1={bottom} x2={right} y2={bottom} />
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line className={s.grid} x1={left} y1={bottom - t * (bottom - top)} x2={right} y2={bottom - t * (bottom - top)} />
            <text className={s.tick} x={left - 6} y={bottom - t * (bottom - top) + 3} textAnchor="end">
              {Math.round(t * max * 100)}%
            </text>
          </g>
        ))}
        {values.map((v, i) => {
          const h = (v / max) * (bottom - top);
          const cx = left + slot * (i + 0.5);
          return (
            <g key={i}>
              <rect
                x={cx - barW / 2}
                y={bottom - h}
                width={barW}
                height={Math.max(h, 0.5)}
                fill={i < highlight ? keepColor : dropColor}
                opacity={0.9}
              />
              {(values.length <= 6 || i < 3) && (
                <text className={s.dataLabel} x={cx} y={bottom - h - 5} textAnchor="middle">
                  {(v * 100).toFixed(values.length <= 6 ? 1 : 2)}%
                </text>
              )}
              <text className={s.tick} x={cx} y={bottom + 14} textAnchor="middle">
                {labels[i]}
              </text>
            </g>
          );
        })}
        {line && (
          <>
            <path
              d={line
                .map((v, i) => `${i ? 'L' : 'M'}${(left + slot * (i + 0.5)).toFixed(1)},${(bottom - (v / max) * (bottom - top)).toFixed(1)}`)
                .join(' ')}
              fill="none"
              stroke={accent}
              strokeWidth={2.4}
            />
            {line.map((v, i) => (
              <circle
                key={i}
                cx={left + slot * (i + 0.5)}
                cy={bottom - (v / max) * (bottom - top)}
                r={4}
                fill={accent}
                stroke="var(--surface-raised)"
                strokeWidth={2}
              />
            ))}
          </>
        )}
      </>
    );
  };

  const cx0 = W / 2;
  const cy0 = H / 2;
  const scale = 40;
  const ax = (v: number) => cx0 + v * scale;
  const ay = (v: number) => cy0 - v * scale;
  const rad = (angle * Math.PI) / 180;
  const dir = [Math.cos(rad), Math.sin(rad)];

  return (
    <VizPanel
      title="Principal component analysis"
      hint={
        mode === 'variance'
          ? 'Variance explained is each eigenvalue over the total. With the lecture values 6.2, 2.4, 1.0 and 0.4, the first two components keep 86%. Change an eigenvalue or the number kept.'
          : mode === 'rotate'
            ? 'Project every point onto an axis through the centre and measure the spread of the projections. The spread is largest at the first principal component, 30.9 degrees here, where it equals the first eigenvalue.'
            : 'PCA follows variance, and variance depends on units. Unscaled, one large-valued column (proline) is the first component; standardised, the variance is shared across many features.'
      }
      legend={
        mode === 'rotate'
          ? [
              {label: 'points and their projections', color: keepColor},
              {label: 'projection axis', color: accent},
            ]
          : [
              {label: mode === 'variance' ? 'components kept' : 'first component', color: keepColor},
              {label: 'dropped', color: dropColor},
              ...(mode === 'variance' ? [{label: 'cumulative', color: accent}] : []),
            ]
      }
      table={table}
      controls={
        <>
          <label className={s.control}>
            view
            <select className={s.select} value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              <option value="variance">Variance explained</option>
              <option value="rotate">Rotate the axis</option>
              <option value="wine">Scaling matters (wine)</option>
            </select>
          </label>
          {mode === 'variance' && (
            <>
              {eig.map((v, i) => (
                <label key={i} className={s.control}>
                  λ{i + 1}
                  <input
                    type="range"
                    min={0}
                    max={10}
                    step={0.1}
                    value={v}
                    onChange={(e) => setEigAt(i, Number(e.target.value))}
                  />
                  <span className={s.value}>{v.toFixed(1)}</span>
                </label>
              ))}
              <label className={s.control}>
                keep
                <input
                  type="range"
                  min={1}
                  max={4}
                  step={1}
                  value={keep}
                  onChange={(e) => setKeep(Number(e.target.value))}
                />
                <span className={s.value}>{keep}</span>
              </label>
            </>
          )}
          {mode === 'rotate' && (
            <>
              <label className={s.control}>
                angle
                <input
                  type="range"
                  min={0}
                  max={180}
                  step={0.1}
                  value={angle}
                  onChange={(e) => setAngle(Number(e.target.value))}
                />
                <span className={s.value}>{angle.toFixed(1)}°</span>
              </label>
              <button type="button" className={s.button} onClick={() => setAngle(Math.round(stats.angle * 10) / 10)}>
                Snap to PC1
              </button>
            </>
          )}
          {mode === 'wine' && (
            <button type="button" className={s.button} onClick={() => setScaled(!scaled)} aria-pressed={scaled}>
              {scaled ? 'Showing standardised, switch to raw' : 'Showing raw, switch to standardised'}
            </button>
          )}
          <span className={s.value} aria-live="polite">
            {headline}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={headline}>
        {mode === 'variance' && barChart(shares, keep, cumulative, 1, ['PC1', 'PC2', 'PC3', 'PC4'])}
        {mode === 'wine' &&
          barChart(
            wine,
            1,
            null,
            1,
            wine.map((_, i) => `${i + 1}`),
          )}
        {mode === 'rotate' && (
          <g>
            <line className={s.axis} x1={ax(-5.2)} y1={cy0} x2={ax(5.2)} y2={cy0} />
            <line className={s.axis} x1={cx0} y1={ay(-3.4)} x2={cx0} y2={ay(3.4)} />
            <line
              x1={ax(-5.6 * dir[0])}
              y1={ay(-5.6 * dir[1])}
              x2={ax(5.6 * dir[0])}
              y2={ay(5.6 * dir[1])}
              stroke={accent}
              strokeWidth={2.6}
            />
            {stats.centred.map((p, i) => {
              const t = p[0] * dir[0] + p[1] * dir[1];
              return (
                <g key={i}>
                  <line
                    x1={ax(p[0])}
                    y1={ay(p[1])}
                    x2={ax(t * dir[0])}
                    y2={ay(t * dir[1])}
                    stroke={dropColor}
                    strokeWidth={1}
                    opacity={0.7}
                  />
                  <circle cx={ax(p[0])} cy={ay(p[1])} r={3.4} fill={keepColor} opacity={0.85} />
                  <circle cx={ax(t * dir[0])} cy={ay(t * dir[1])} r={2} fill={accent} />
                </g>
              );
            })}
            <text className={s.dataLabel} x={14} y={22} textAnchor="start" fill={good}>
              variance along axis {along.toFixed(3)}
            </text>
            <text className={s.tick} x={14} y={38} textAnchor="start">
              total {stats.total.toFixed(3)}, lost {lost.toFixed(3)}
            </text>
          </g>
        )}
        {mode === 'variance' && (
          <text className={s.axisLabel} x={W / 2} y={H - 4} textAnchor="middle">
            share of total variance (λ divided by the sum of the λ values)
          </text>
        )}
        {mode === 'wine' && (
          <text className={s.axisLabel} x={W / 2} y={H - 4} textAnchor="middle">
            principal component (explained variance ratio)
          </text>
        )}
      </svg>
    </VizPanel>
  );
}
