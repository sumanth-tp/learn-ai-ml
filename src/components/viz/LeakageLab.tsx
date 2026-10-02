import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

function mulberry32(seed: number) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(next: () => number) {
  const u = Math.max(next(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
}

function noiseProblem(n: number, p: number, seed: number) {
  const next = mulberry32(seed);
  const X = new Float64Array(n * p);
  for (let i = 0; i < n * p; i += 1) X[i] = normal(next);
  const y: number[] = [];
  for (let i = 0; i < n; i += 1) y.push(next() < 0.5 ? 1 : -1);
  return {X, y};
}

function chooseFeatures(X: Float64Array, y: number[], p: number, rows: number, count: number) {
  const corr = new Float64Array(p);
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < p; j += 1) corr[j] += X[i * p + j] * y[i];
  }
  for (let j = 0; j < p; j += 1) corr[j] /= rows;
  const order = Array.from({length: p}, (_, j) => j).sort((a, b) => Math.abs(corr[b]) - Math.abs(corr[a]));
  const keep = order.slice(0, count);
  return {keep, signs: keep.map((j) => Math.sign(corr[j]))};
}

function accuracyByK(
  X: Float64Array,
  y: number[],
  p: number,
  from: number,
  to: number,
  chosen: {keep: number[]; signs: number[]},
) {
  const scores = new Float64Array(to - from);
  const out: number[] = [];
  for (let k = 0; k < chosen.keep.length; k += 1) {
    const j = chosen.keep[k];
    let right = 0;
    for (let i = from; i < to; i += 1) {
      scores[i - from] += chosen.signs[k] * X[i * p + j];
      if ((scores[i - from] > 0 ? 1 : -1) === y[i]) right += 1;
    }
    out.push(right / (to - from));
  }
  return out;
}

const SEEDS = [5, 6, 7, 8];
const MAX_K = 100;
const TABLE_K = [1, 5, 10, 20, 50, 100];
const W1 = 300;
const H1 = 250;
const W2 = 360;
const H2 = 250;

export default function LeakageLab() {
  const dark = useDarkViz();
  const [rows, setRows] = useState(100);
  const [features, setFeatures] = useState(1000);
  const [k, setK] = useState(20);
  const [seed, setSeed] = useState(5);

  const kShown = Math.min(k, features);
  const half = Math.floor(rows / 2);

  const result = useMemo(() => {
    const {X, y} = noiseProblem(rows, features, seed);
    const top = Math.min(MAX_K, features);
    const all = chooseFeatures(X, y, features, rows, top);
    const train = chooseFeatures(X, y, features, half, top);
    return {
      leaky: accuracyByK(X, y, features, half, rows, all),
      honest: accuracyByK(X, y, features, half, rows, train),
      honestTrain: accuracyByK(X, y, features, 0, half, train),
    };
  }, [rows, features, seed, half]);

  const leakyAt = result.leaky[kShown - 1];
  const honestAt = result.honest[kShown - 1];
  const trainAt = result.honestTrain[kShown - 1];

  const leakyColor = seriesColor(1, dark);
  const honestColor = seriesColor(0, dark);
  const trainColor = seriesColor(2, dark);

  const bars = [
    {label: 'leaky', sub: 'test half', value: leakyAt, color: leakyColor},
    {label: 'honest', sub: 'test half', value: honestAt, color: honestColor},
    {label: 'honest', sub: 'own rows', value: trainAt, color: trainColor},
  ];
  const by = (v: number) => 20 + (1 - v) * (H1 - 70);
  const cx = (k0: number) => 40 + ((k0 - 1) / Math.max(MAX_K - 1, 1)) * (W2 - 56);
  const cy = (v: number) => 16 + (1 - v) * (H2 - 50);
  const line = (values: number[]) =>
    values.map((v, i) => `${i ? 'L' : 'M'}${cx(i + 1).toFixed(1)},${cy(v).toFixed(1)}`).join(' ');

  return (
    <VizPanel
      title="Feature selection leakage on pure noise"
      hint="The labels are coin flips, so the honest score hovers around 0.5. Choosing the best features with every row, test rows included, manufactures skill out of nothing. The effect grows with more candidate features and fewer rows."
      legend={[
        {label: 'leaky: features chosen on all rows', color: leakyColor},
        {label: 'honest: features chosen on training half', color: honestColor},
        {label: 'honest, scored on the rows it was chosen from', color: trainColor},
      ]}
      table={{
        columns: ['features kept', 'leaky test accuracy', 'honest test accuracy', 'honest train accuracy'],
        rows: TABLE_K.filter((v) => v <= result.leaky.length).map((v) => [
          v,
          result.leaky[v - 1].toFixed(3),
          result.honest[v - 1].toFixed(3),
          result.honestTrain[v - 1].toFixed(3),
        ]),
      }}
      controls={
        <>
          <label className={s.control}>
            rows
            <input
              type="range"
              min={40}
              max={400}
              step={20}
              value={rows}
              onChange={(e) => setRows(Number(e.target.value))}
            />
            <span className={s.value}>{rows}</span>
          </label>
          <label className={s.control}>
            candidate features
            <input
              type="range"
              min={100}
              max={2000}
              step={100}
              value={features}
              onChange={(e) => setFeatures(Number(e.target.value))}
            />
            <span className={s.value}>{features}</span>
          </label>
          <label className={s.control}>
            features kept
            <input
              type="range"
              min={1}
              max={MAX_K}
              step={1}
              value={kShown}
              onChange={(e) => setK(Number(e.target.value))}
            />
            <span className={s.value}>{kShown}</span>
          </label>
          <label className={s.control}>
            data set
            <select className={s.select} value={seed} onChange={(e) => setSeed(Number(e.target.value))}>
              {SEEDS.map((v, i) => (
                <option key={v} value={v}>
                  {i + 1}
                </option>
              ))}
            </select>
          </label>
        </>
      }>
      <div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap'}}>
        <svg
          className={s.svg}
          style={{flex: '1 1 240px', minWidth: 0}}
          viewBox={`0 0 ${W1} ${H1}`}
          role="img"
          aria-label="Accuracy of the leaky and honest procedures on the test half">
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line className={s.grid} x1={34} y1={by(t)} x2={W1 - 10} y2={by(t)} />
              <text className={s.tick} x={28} y={by(t) + 3} textAnchor="end">
                {t}
              </text>
            </g>
          ))}
          <line
            x1={34}
            y1={by(0.5)}
            x2={W1 - 10}
            y2={by(0.5)}
            stroke="var(--text-faint)"
            strokeWidth={1.5}
            strokeDasharray="5 4"
          />
          {bars.map((bar, i) => {
            const x = 48 + i * 82;
            return (
              <g key={bar.label + bar.sub}>
                <rect x={x} y={by(bar.value)} width={60} height={by(0) - by(bar.value)} rx={4} fill={bar.color} />
                <text
                  className={s.dataLabel}
                  x={x + 30}
                  y={by(bar.value) - 5}
                  textAnchor="middle"
                  stroke="var(--surface-raised)"
                  strokeWidth={4}
                  paintOrder="stroke">
                  {bar.value.toFixed(3)}
                </text>
                <text className={s.tick} x={x + 30} y={H1 - 26} textAnchor="middle">
                  {bar.label}
                </text>
                <text className={s.tick} x={x + 30} y={H1 - 14} textAnchor="middle">
                  {bar.sub}
                </text>
              </g>
            );
          })}
        </svg>
        <svg
          className={s.svg}
          style={{flex: '1 1 280px', minWidth: 0}}
          viewBox={`0 0 ${W2} ${H2}`}
          role="img"
          aria-label="Test accuracy against the number of features kept, leaky and honest">
          {[0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line className={s.grid} x1={40} y1={cy(t)} x2={W2 - 16} y2={cy(t)} />
              <text className={s.tick} x={34} y={cy(t) + 3} textAnchor="end">
                {t}
              </text>
            </g>
          ))}
          <line
            x1={40}
            y1={cy(0.5)}
            x2={W2 - 16}
            y2={cy(0.5)}
            stroke="var(--text-faint)"
            strokeWidth={1.5}
            strokeDasharray="5 4"
          />
          <line
            x1={cx(kShown)}
            y1={16}
            x2={cx(kShown)}
            y2={H2 - 34}
            stroke="var(--border-strong)"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
          {[1, 25, 50, 75, 100].map((t) => (
            <text key={t} className={s.tick} x={cx(t)} y={H2 - 20} textAnchor="middle">
              {t}
            </text>
          ))}
          <text className={s.axisLabel} x={W2 / 2} y={H2 - 4} textAnchor="middle">
            features kept
          </text>
          <path d={line(result.leaky)} fill="none" stroke={leakyColor} strokeWidth={2.5} />
          <path d={line(result.honest)} fill="none" stroke={honestColor} strokeWidth={2.5} />
          <circle cx={cx(kShown)} cy={cy(leakyAt)} r={4.5} fill={leakyColor} stroke="var(--surface-raised)" strokeWidth={2} />
          <circle cx={cx(kShown)} cy={cy(honestAt)} r={4.5} fill={honestColor} stroke="var(--surface-raised)" strokeWidth={2} />
        </svg>
      </div>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem', display: 'block'}}>
        <div>
          {rows} rows, {features} noise features, best <code>{kShown}</code> kept: leaky test accuracy{' '}
          <code>{leakyAt.toFixed(3)}</code>, honest <code>{honestAt.toFixed(3)}</code>, honest on its own training
          half <code>{trainAt.toFixed(3)}</code>
        </div>
      </div>
    </VizPanel>
  );
}
