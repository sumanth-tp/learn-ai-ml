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

const truth = (x: number) => Math.cos(1.5 * Math.PI * x);

function sample(n: number, seed: number, noise: number) {
  const next = mulberry32(seed);
  const x: number[] = [];
  for (let i = 0; i < n; i += 1) x.push((i + next()) / n);
  const y = x.map((xi) => truth(xi) + noise * normal(next));
  return {x, y};
}

function vandermonde(x: number[], degree: number) {
  return x.map((xi) => {
    const t = 2 * xi - 1;
    const row: number[] = [];
    let p = 1;
    for (let d = 0; d <= degree; d += 1) {
      row.push(p);
      p *= t;
    }
    return row;
  });
}

function leastSquares(A: number[][], b: number[]) {
  const m = A.length;
  const n = A[0].length;
  const R = A.map((row) => row.slice());
  const y = b.slice();
  for (let k = 0; k < n; k += 1) {
    let norm = 0;
    for (let i = k; i < m; i += 1) norm += R[i][k] * R[i][k];
    norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const alpha = R[k][k] > 0 ? -norm : norm;
    const v = new Array<number>(m).fill(0);
    for (let i = k; i < m; i += 1) v[i] = R[i][k];
    v[k] -= alpha;
    let vv = 0;
    for (let i = k; i < m; i += 1) vv += v[i] * v[i];
    if (vv === 0) continue;
    for (let j = k; j < n; j += 1) {
      let dot = 0;
      for (let i = k; i < m; i += 1) dot += v[i] * R[i][j];
      const f = (2 * dot) / vv;
      for (let i = k; i < m; i += 1) R[i][j] -= f * v[i];
    }
    let dy = 0;
    for (let i = k; i < m; i += 1) dy += v[i] * y[i];
    const fy = (2 * dy) / vv;
    for (let i = k; i < m; i += 1) y[i] -= fy * v[i];
  }
  const w = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i -= 1) {
    let acc = y[i];
    for (let j = i + 1; j < n; j += 1) acc -= R[i][j] * w[j];
    w[i] = R[i][i] === 0 ? 0 : acc / R[i][i];
  }
  return w;
}

function polynomial(w: number[], x: number) {
  const t = 2 * x - 1;
  let acc = 0;
  for (let d = w.length - 1; d >= 0; d -= 1) acc = acc * t + w[d];
  return acc;
}

function mse(w: number[], x: number[], y: number[]) {
  let total = 0;
  for (let i = 0; i < x.length; i += 1) {
    const e = polynomial(w, x[i]) - y[i];
    total += e * e;
  }
  return total / x.length;
}

const SEEDS = [11, 21, 31, 41];
const W1 = 360;
const H1 = 270;
const W2 = 300;
const H2 = 270;
const Y_LIMIT = 2.2;

export default function BiasVarianceLab() {
  const dark = useDarkViz();
  const [degree, setDegree] = useState(4);
  const [noise, setNoise] = useState(0.3);
  const [rows, setRows] = useState(30);
  const [seed, setSeed] = useState(11);

  const maxDegree = Math.min(15, rows - 1);
  const shown = Math.min(degree, maxDegree);

  const model = useMemo(() => {
    const train = sample(rows, seed, noise);
    const test = sample(200, seed + 1, noise);
    const fits = [];
    for (let d = 1; d <= maxDegree; d += 1) {
      const w = leastSquares(vandermonde(train.x, d), train.y);
      fits.push({degree: d, w, train: mse(w, train.x, train.y), test: mse(w, test.x, test.y)});
    }
    return {train, fits};
  }, [rows, seed, noise, maxDegree]);

  const current = model.fits[shown - 1];
  const trainColor = seriesColor(0, dark);
  const testColor = seriesColor(1, dark);
  const floor = noise * noise;

  const px = (x: number) => 36 + x * (W1 - 50);
  const py = (y: number) => 14 + ((Y_LIMIT - Math.max(-Y_LIMIT, Math.min(Y_LIMIT, y))) / (2 * Y_LIMIT)) * (H1 - 44);

  const curve = useMemo(() => {
    const out: string[] = [];
    for (let i = 0; i <= 240; i += 1) {
      const x = i / 240;
      out.push(`${i ? 'L' : 'M'}${px(x).toFixed(1)},${py(polynomial(current.w, x)).toFixed(1)}`);
    }
    return out.join(' ');
  }, [current]);

  const truthPath = useMemo(() => {
    const out: string[] = [];
    for (let i = 0; i <= 120; i += 1) {
      const x = i / 120;
      out.push(`${i ? 'L' : 'M'}${px(x).toFixed(1)},${py(truth(x)).toFixed(1)}`);
    }
    return out.join(' ');
  }, []);

  const logLow = Math.log10(0.02);
  const logHigh = Math.log10(20);
  const qx = (d: number) => 40 + ((d - 1) / Math.max(maxDegree - 1, 1)) * (W2 - 56);
  const qy = (v: number) => {
    const c = Math.log10(Math.max(0.02, Math.min(20, v)));
    return 14 + ((logHigh - c) / (logHigh - logLow)) * (H2 - 44);
  };
  const line = (key: 'train' | 'test') =>
    model.fits.map((f, i) => `${i ? 'L' : 'M'}${qx(f.degree).toFixed(1)},${qy(f[key]).toFixed(1)}`).join(' ');

  const verdict =
    shown <= 2 && current.test > 1.8 * floor
      ? 'Underfitting: both errors are high, the curve is too stiff to follow the data.'
      : current.test > 2 * current.train + floor
        ? 'Overfitting: the training error is low but the test error is much higher, the curve is chasing noise.'
        : 'A good balance: the test error sits close to the noise floor.';

  return (
    <VizPanel
      title="Bias and variance: fit a polynomial of growing degree"
      hint="Raise the degree: training error only ever falls, but test error falls and then climbs again. The dashed line at the bottom is the noise floor, the error no model can beat."
      legend={[
        {label: 'training points and training error', color: trainColor},
        {label: 'fitted polynomial and test error', color: testColor},
        {label: 'true curve, noise floor', color: 'var(--text-faint)'},
      ]}
      table={{
        columns: ['degree', 'train MSE', 'test MSE'],
        rows: model.fits.map((f) => [f.degree, f.train.toFixed(4), f.test.toFixed(4)]),
      }}
      controls={
        <>
          <label className={s.control}>
            degree
            <input
              type="range"
              min={1}
              max={maxDegree}
              step={1}
              value={shown}
              onChange={(e) => setDegree(Number(e.target.value))}
            />
            <span className={s.value}>{shown}</span>
          </label>
          <label className={s.control}>
            noise sd
            <input
              type="range"
              min={0.05}
              max={0.6}
              step={0.05}
              value={noise}
              onChange={(e) => setNoise(Number(e.target.value))}
            />
            <span className={s.value}>{noise.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            training rows
            <input
              type="range"
              min={10}
              max={60}
              step={5}
              value={rows}
              onChange={(e) => setRows(Number(e.target.value))}
            />
            <span className={s.value}>{rows}</span>
          </label>
          <label className={s.control}>
            sample
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
          style={{flex: '1 1 300px', minWidth: 0}}
          viewBox={`0 0 ${W1} ${H1}`}
          role="img"
          aria-label={`Polynomial of degree ${shown} fitted to ${rows} noisy points of a cosine curve`}>
          <line className={s.axis} x1={px(0)} y1={py(0)} x2={px(1)} y2={py(0)} />
          {[0, 0.5, 1].map((t) => (
            <text key={t} className={s.tick} x={px(t)} y={H1 - 16} textAnchor="middle">
              {t}
            </text>
          ))}
          {[-2, -1, 0, 1, 2].map((t) => (
            <text key={t} className={s.tick} x={30} y={py(t) + 3} textAnchor="end">
              {t}
            </text>
          ))}
          <text className={s.axisLabel} x={W1 / 2} y={H1 - 2} textAnchor="middle">
            input x
          </text>
          <path d={truthPath} fill="none" stroke="var(--text-faint)" strokeWidth={1.5} strokeDasharray="5 4" />
          <path d={curve} fill="none" stroke={testColor} strokeWidth={2.5} />
          {model.train.x.map((x, i) => (
            <circle key={i} cx={px(x)} cy={py(model.train.y[i])} r={3.2} fill={trainColor} fillOpacity={0.85} />
          ))}
        </svg>
        <svg
          className={s.svg}
          style={{flex: '1 1 260px', minWidth: 0}}
          viewBox={`0 0 ${W2} ${H2}`}
          role="img"
          aria-label="Training and test error against polynomial degree on a log axis">
          {[0.02, 0.1, 1, 10].map((v) => (
            <g key={v}>
              <line className={s.grid} x1={40} y1={qy(v)} x2={W2 - 16} y2={qy(v)} />
              <text className={s.tick} x={34} y={qy(v) + 3} textAnchor="end">
                {v}
              </text>
            </g>
          ))}
          <line
            x1={40}
            y1={qy(floor)}
            x2={W2 - 16}
            y2={qy(floor)}
            stroke="var(--text-faint)"
            strokeWidth={1.5}
            strokeDasharray="5 4"
          />
          <line
            x1={qx(shown)}
            y1={14}
            x2={qx(shown)}
            y2={H2 - 30}
            stroke="var(--border-strong)"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
          {model.fits.map((f) => (
            <text key={f.degree} className={s.tick} x={qx(f.degree)} y={H2 - 16} textAnchor="middle">
              {f.degree % 2 === 1 || maxDegree < 9 ? f.degree : ''}
            </text>
          ))}
          <text className={s.axisLabel} x={W2 / 2} y={H2 - 2} textAnchor="middle">
            polynomial degree
          </text>
          <path d={line('train')} fill="none" stroke={trainColor} strokeWidth={2.5} />
          <path d={line('test')} fill="none" stroke={testColor} strokeWidth={2.5} />
          <circle cx={qx(shown)} cy={qy(current.train)} r={4.5} fill={trainColor} stroke="var(--surface-raised)" strokeWidth={2} />
          <circle cx={qx(shown)} cy={qy(current.test)} r={4.5} fill={testColor} stroke="var(--surface-raised)" strokeWidth={2} />
        </svg>
      </div>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem', display: 'block'}}>
        <div>
          degree <code>{shown}</code>: train MSE <code>{current.train.toFixed(4)}</code>, test MSE{' '}
          <code>{current.test.toFixed(4)}</code>, noise variance <code>{floor.toFixed(4)}</code>
        </div>
        <div style={{marginTop: '0.25rem'}}>{verdict}</div>
      </div>
    </VizPanel>
  );
}
