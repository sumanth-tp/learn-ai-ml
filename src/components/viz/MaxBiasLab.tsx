import {useCallback, useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 220;
const PAD = {top: 18, right: 16, bottom: 30, left: 52};

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number) {
  const u = Math.max(rng(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

type Result = {single: number[]; double: number[]};

/** Every action's true value is 0; only the estimates are noisy. */
function simulate(actions: number, samples: number, trials: number, seed: number): Result {
  const rng = mulberry(seed);
  const single: number[] = [];
  const double: number[] = [];
  const noise = 1 / Math.sqrt(samples);
  for (let t = 0; t < trials; t += 1) {
    const a: number[] = [];
    const b: number[] = [];
    for (let i = 0; i < actions; i += 1) {
      a.push(gaussian(rng) * noise);
      b.push(gaussian(rng) * noise);
    }
    single.push(Math.max(...a));
    let best = 0;
    for (let i = 1; i < actions; i += 1) if (a[i] > a[best]) best = i;
    double.push(b[best]);
  }
  return {single, double};
}

const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;

export default function MaxBiasLab() {
  const dark = useDarkViz();
  const [actions, setActions] = useState(10);
  const [samples, setSamples] = useState(5);
  const [seed, setSeed] = useState(1);

  const result = useMemo(() => simulate(actions, samples, 4000, seed), [actions, samples, seed]);
  const singleBias = mean(result.single);
  const doubleBias = mean(result.double);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const scale = Math.max(0.05, Math.abs(singleBias) * 1.4);
  const x = (v: number) => PAD.left + innerW / 2 + (v / scale) * (innerW / 2);

  const singleColor = seriesColor(1, dark);
  const doubleColor = seriesColor(2, dark);

  const bar = (value: number, row: number, color: string, label: string) => {
    const zero = x(0);
    const end = x(value);
    const yTop = PAD.top + row * (innerH / 2) + 18;
    return (
      <g key={label}>
        <text className={s.tick} x={PAD.left - 8} y={yTop + 12} textAnchor="end">
          {label}
        </text>
        <rect
          x={Math.min(zero, end)}
          y={yTop}
          width={Math.max(2, Math.abs(end - zero))}
          height={20}
          rx={4}
          fill={color}
        />
        <text className={s.dataLabel} x={end + (value >= 0 ? 8 : -8)} y={yTop + 14}
              textAnchor={value >= 0 ? 'start' : 'end'}>
          {value >= 0 ? '+' : ''}{value.toFixed(4)}
        </text>
      </g>
    );
  };

  return (
    <VizPanel
      title="Maximisation bias: one estimator versus two"
      hint="Every action here is worth exactly zero. Taking the max of noisy estimates is reliably optimistic; splitting selection from evaluation removes the bias. Fewer samples or more actions makes it worse — which is exactly the situation early in training."
      legend={[
        {label: 'single estimator: max of the estimates', color: singleColor},
        {label: 'double estimator: select with A, evaluate with B', color: doubleColor},
      ]}
      table={{
        columns: ['estimator', 'mean estimate', 'true value', 'bias'],
        rows: [
          ['single (max)', singleBias.toFixed(4), '0.0000', `+${singleBias.toFixed(4)}`],
          ['double', doubleBias.toFixed(4), '0.0000', doubleBias.toFixed(4)],
        ],
      }}
      controls={
        <>
          <label className={s.control}>
            actions
            <input type="range" min={2} max={30} step={1} value={actions}
                   onChange={(e) => setActions(Number(e.target.value))} />
            <span className={s.value}>{actions}</span>
          </label>
          <label className={s.control}>
            samples per action
            <input type="range" min={1} max={60} step={1} value={samples}
                   onChange={(e) => setSamples(Number(e.target.value))} />
            <span className={s.value}>{samples}</span>
          </label>
          <button type="button" className={s.button} onClick={() => setSeed((v) => v + 1)}>
            resample
          </button>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Bias of the single and double estimators, where the true value is zero">
        <line className={s.grid} x1={x(0)} y1={PAD.top} x2={x(0)} y2={PAD.top + innerH} />
        <text className={s.tick} x={x(0)} y={PAD.top - 5} textAnchor="middle">
          true value = 0
        </text>
        {bar(singleBias, 0, singleColor, 'single')}
        {bar(doubleBias, 1, doubleColor, 'double')}
      </svg>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0}}>
        <span>
          overestimate removed:{' '}
          <strong>
            {singleBias !== 0
              ? `${(100 * (1 - Math.abs(doubleBias) / Math.abs(singleBias))).toFixed(1)}%`
              : '—'}
          </strong>
        </span>
        <span>4,000 trials per setting</span>
      </div>
    </VizPanel>
  );
}
