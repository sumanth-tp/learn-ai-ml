import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const BASE = [2, 4, 5, 6, 7, 8, 9, 10, 12];
const W = 440;
const H = 190;
const LEFT = 26;
const RIGHT = W - 26;
const MAX_VALUE = 100;

function quantile(sorted: number[], q: number) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export default function ScalingOutlierLab() {
  const dark = useDarkViz();
  const [value, setValue] = useState(8);
  const [outlier, setOutlier] = useState(45);
  const [useOutlier, setUseOutlier] = useState(true);
  const [sampleSigma, setSampleSigma] = useState(false);

  const stats = useMemo(() => {
    const data = useOutlier ? [...BASE, outlier] : [...BASE];
    const sorted = [...data].sort((a, b) => a - b);
    const n = data.length;
    const mean = data.reduce((a, b) => a + b, 0) / n;
    const ss = data.reduce((a, b) => a + (b - mean) * (b - mean), 0);
    const sigma = Math.sqrt(ss / (sampleSigma ? n - 1 : n));
    const q1 = quantile(sorted, 0.25);
    const q3 = quantile(sorted, 0.75);
    const iqr = q3 - q1;
    return {
      data,
      sorted,
      mean,
      sigma,
      q1,
      q3,
      iqr,
      median: quantile(sorted, 0.5),
      lo: sorted[0],
      hi: sorted[n - 1],
      fenceLow: q1 - 1.5 * iqr,
      fenceHigh: q3 + 1.5 * iqr,
      sigmaLow: mean - 3 * sigma,
      sigmaHigh: mean + 3 * sigma,
    };
  }, [outlier, useOutlier, sampleSigma]);

  const z = (x: number) => (x - stats.mean) / stats.sigma;
  const minMax = (x: number) => (x - stats.lo) / (stats.hi - stats.lo);
  const robust = (x: number) => (x - stats.median) / stats.iqr;
  const byIqr = (x: number) => x < stats.fenceLow || x > stats.fenceHigh;
  const bySigma = (x: number) => x < stats.sigmaLow || x > stats.sigmaHigh;

  const px = (x: number) => LEFT + (Math.max(-5, Math.min(MAX_VALUE + 5, x)) / MAX_VALUE) * (RIGHT - LEFT);
  const fenceColor = seriesColor(0, dark);
  const sigmaColor = seriesColor(1, dark);
  const pointColor = seriesColor(2, dark);

  const seen = new Map<number, number>();
  const placed = stats.data.map((x) => {
    const count = seen.get(x) ?? 0;
    seen.set(x, count + 1);
    return {x, row: count};
  });

  const flaggedIqr = stats.sorted.filter(byIqr);
  const flaggedSigma = stats.sorted.filter(bySigma);
  const fmt = (v: number) => v.toFixed(3);
  const list = (v: number[]) => (v.length ? v.join(', ') : 'none');

  return (
    <VizPanel
      title="Scale a value, then ask which points are outliers"
      hint="The lecture's data set with one editable outlier. Drag the outlier up and watch the mean and standard deviation chase it, so the 3-sigma limits run away while the IQR fences barely move."
      legend={[
        {label: 'IQR fences, Q1 and Q3 -/+ 1.5 IQR', color: fenceColor},
        {label: 'mean -/+ 3 sigma', color: sigmaColor},
        {label: 'data points', color: pointColor},
      ]}
      table={{
        columns: ['value', 'standardised', 'min-max', 'robust', 'IQR rule', '3-sigma rule'],
        rows: stats.sorted.map((x) => [
          x,
          fmt(z(x)),
          fmt(minMax(x)),
          fmt(robust(x)),
          byIqr(x) ? 'outlier' : 'ok',
          bySigma(x) ? 'outlier' : 'ok',
        ]),
      }}
      controls={
        <>
          <label className={s.control}>
            value x
            <input
              type="range"
              min={0}
              max={MAX_VALUE}
              step={0.5}
              value={value}
              onChange={(e) => setValue(Number(e.target.value))}
            />
            <span className={s.value}>{value}</span>
          </label>
          <label className={s.control}>
            outlier value
            <input
              type="range"
              min={10}
              max={MAX_VALUE}
              step={1}
              value={outlier}
              disabled={!useOutlier}
              onChange={(e) => setOutlier(Number(e.target.value))}
            />
            <span className={s.value}>{outlier}</span>
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={useOutlier} onChange={(e) => setUseOutlier(e.target.checked)} />
            include outlier
          </label>
          <label className={s.control}>
            sigma
            <select
              className={s.select}
              value={sampleSigma ? 'sample' : 'population'}
              onChange={(e) => setSampleSigma(e.target.value === 'sample')}>
              <option value="population">population (n)</option>
              <option value="sample">sample (n - 1)</option>
            </select>
          </label>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Number line with the data points, IQR fences, 3-sigma limits and the chosen value">
        <rect
          x={px(Math.max(stats.fenceLow, 0))}
          y={22}
          width={Math.max(px(stats.fenceHigh) - px(Math.max(stats.fenceLow, 0)), 0)}
          height={92}
          fill={fenceColor}
          opacity={0.12}
        />
        <line x1={px(stats.fenceHigh)} y1={22} x2={px(stats.fenceHigh)} y2={114} stroke={fenceColor} strokeWidth={2} />
        <text
          className={s.dataLabel}
          x={Math.max(80, Math.min(px(stats.fenceHigh), W - 80))}
          y={16}
          textAnchor="middle"
          fill={fenceColor}>
          IQR upper fence {stats.fenceHigh.toFixed(1)}
        </text>
        <line className={s.axis} x1={LEFT} y1={114} x2={RIGHT} y2={114} />
        {[0, 20, 40, 60, 80, 100].map((t) => (
          <text key={t} className={s.tick} x={px(t)} y={130} textAnchor="middle">
            {t}
          </text>
        ))}
        <line x1={px(stats.sigmaHigh)} y1={140} x2={px(stats.sigmaHigh)} y2={162} stroke={sigmaColor} strokeWidth={2} />
        <line
          x1={px(Math.max(stats.sigmaLow, 0))}
          y1={151}
          x2={px(stats.sigmaHigh)}
          y2={151}
          stroke={sigmaColor}
          strokeWidth={2}
          strokeDasharray="5 4"
        />
        <text
          className={s.dataLabel}
          x={Math.max(80, Math.min(px(stats.sigmaHigh), W - 80))}
          y={178}
          textAnchor="middle"
          fill={sigmaColor}>
          3-sigma upper limit {stats.sigmaHigh.toFixed(1)}
        </text>
        <line
          x1={px(value)}
          y1={30}
          x2={px(value)}
          y2={114}
          stroke="var(--text-strong)"
          strokeWidth={1.5}
          strokeDasharray="3 3"
        />
        <text className={s.dataLabel} x={px(value)} y={40} textAnchor="middle" dx={px(value) > W - 60 ? -26 : 26}>
          x = {value}
        </text>
        {placed.map((p, i) => (
          <g key={i}>
            <circle cx={px(p.x)} cy={104 - p.row * 11} r={5} fill={pointColor} stroke="var(--surface-raised)" strokeWidth={1.5} />
            {byIqr(p.x) && (
              <circle cx={px(p.x)} cy={104 - p.row * 11} r={10} fill="none" stroke={fenceColor} strokeWidth={2} />
            )}
            {bySigma(p.x) && (
              <path
                d={`M${px(p.x) - 7},${97 - p.row * 11} l14,14 M${px(p.x) + 7},${97 - p.row * 11} l-14,14`}
                stroke={sigmaColor}
                strokeWidth={2}
              />
            )}
          </g>
        ))}
      </svg>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem', display: 'block'}}>
        <div>
          x = <code>{value}</code>: standardised <code>{fmt(z(value))}</code>, min-max <code>{fmt(minMax(value))}</code>,
          robust <code>{fmt(robust(value))}</code>
        </div>
        <div style={{marginTop: '0.25rem'}}>
          mean <code>{stats.mean.toFixed(2)}</code>, sigma <code>{stats.sigma.toFixed(2)}</code>, Q1{' '}
          <code>{stats.q1.toFixed(2)}</code>, Q3 <code>{stats.q3.toFixed(2)}</code>, IQR{' '}
          <code>{stats.iqr.toFixed(2)}</code>
        </div>
        <div style={{marginTop: '0.25rem'}}>
          IQR rule flags <code>{list(flaggedIqr)}</code>; 3-sigma rule flags <code>{list(flaggedSigma)}</code>
        </div>
      </div>
    </VizPanel>
  );
}
