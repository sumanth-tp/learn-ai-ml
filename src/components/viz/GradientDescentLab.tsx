import {useId, useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const X1 = [1, 2, 3];
const Y1 = [1, 2, 2];
const M1 = X1.length;
const CURVATURE = X1.reduce((a, v) => a + v * v, 0) / M1;
const BEST1 = X1.reduce((a, v, i) => a + v * Y1[i], 0) / X1.reduce((a, v) => a + v * v, 0);

const cost1 = (theta: number) =>
  X1.reduce((a, v, i) => a + (theta * v - Y1[i]) ** 2, 0) / (2 * M1);
const grad1 = (theta: number) =>
  X1.reduce((a, v, i) => a + (theta * v - Y1[i]) * v, 0) / M1;

type Step1 = {k: number; theta: number; gradient: number | null; cost: number};

function trail1(alpha: number, start: number, steps: number): Step1[] {
  const out: Step1[] = [{k: 0, theta: start, gradient: null, cost: cost1(start)}];
  let theta = start;
  for (let k = 1; k <= steps; k += 1) {
    const g = grad1(theta);
    theta -= alpha * g;
    out.push({k, theta, gradient: g, cost: cost1(theta)});
  }
  return out;
}

const AREA = [60, 75, 90, 105, 120, 135, 150, 165];
const AGE = [12, 3, 18, 6, 15, 2, 9, 20];
const PRICE = [180, 260, 235, 310, 300, 395, 380, 390];
const M2 = AREA.length;
const CAP = 20000;
const TOLERANCE = 1e-3;

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;

type Problem = {
  H: [number, number, number];
  b: [number, number];
  best: [number, number];
  lambdaMin: number;
  lambdaMax: number;
  vectors: [[number, number], [number, number]];
  gap0: number;
};

function problem2(standardise: boolean): Problem {
  const cols = [AREA, AGE].map((c) => {
    const mu = mean(c);
    const sd = Math.sqrt(mean(c.map((v) => (v - mu) ** 2)));
    return c.map((v) => (standardise ? (v - mu) / sd : v - mu));
  });
  const my = mean(PRICE);
  const yc = PRICE.map((v) => v - my);
  const h11 = cols[0].reduce((a, v) => a + v * v, 0) / M2;
  const h22 = cols[1].reduce((a, v) => a + v * v, 0) / M2;
  const h12 = cols[0].reduce((a, v, i) => a + v * cols[1][i], 0) / M2;
  const b1 = cols[0].reduce((a, v, i) => a + v * yc[i], 0) / M2;
  const b2 = cols[1].reduce((a, v, i) => a + v * yc[i], 0) / M2;
  const det = h11 * h22 - h12 * h12;
  const best: [number, number] = [(h22 * b1 - h12 * b2) / det, (h11 * b2 - h12 * b1) / det];
  const tr = h11 + h22;
  const root = Math.sqrt((tr * tr) / 4 - det);
  const lambdaMax = tr / 2 + root;
  const lambdaMin = tr / 2 - root;
  const vec = (lam: number): [number, number] => {
    const v: [number, number] = h12 !== 0 ? [h12, lam - h11] : lam === h11 ? [1, 0] : [0, 1];
    const n = Math.hypot(v[0], v[1]);
    return [v[0] / n, v[1] / n];
  };
  const p: Problem = {
    H: [h11, h12, h22],
    b: [b1, b2],
    best,
    lambdaMin,
    lambdaMax,
    vectors: [vec(lambdaMin), vec(lambdaMax)],
    gap0: 0,
  };
  p.gap0 = gap2(p, [0, 0]);
  return p;
}

function gap2(p: Problem, t: [number, number]) {
  const d0 = t[0] - p.best[0];
  const d1 = t[1] - p.best[1];
  return 0.5 * (p.H[0] * d0 * d0 + 2 * p.H[1] * d0 * d1 + p.H[2] * d1 * d1);
}

function walk2(p: Problem, alpha: number, steps: number) {
  const out: [number, number][] = [[0, 0]];
  let t: [number, number] = [0, 0];
  for (let k = 0; k < steps; k += 1) {
    const g0 = p.H[0] * t[0] + p.H[1] * t[1] - p.b[0];
    const g1 = p.H[1] * t[0] + p.H[2] * t[1] - p.b[1];
    t = [t[0] - alpha * g0, t[1] - alpha * g1];
    out.push(t);
  }
  return out;
}

function stepsToClose(p: Problem, alpha: number): number | null {
  let t: [number, number] = [0, 0];
  for (let k = 0; k <= CAP; k += 1) {
    if (gap2(p, t) <= TOLERANCE * p.gap0) return k;
    const g0 = p.H[0] * t[0] + p.H[1] * t[1] - p.b[0];
    const g1 = p.H[1] * t[0] + p.H[2] * t[1] - p.b[1];
    t = [t[0] - alpha * g0, t[1] - alpha * g1];
    if (!Number.isFinite(t[0]) || Math.abs(t[0]) > 1e9) return null;
  }
  return null;
}

const W = 420;
const H1 = 250;
const PAD = {top: 14, right: 14, bottom: 34, left: 44};

export default function GradientDescentLab() {
  const dark = useDarkViz();
  const clipId = `gd${useId().replace(/:/g, '')}`;
  const [mode, setMode] = useState<'one' | 'two'>('one');
  const [alpha, setAlpha] = useState(0.1);
  const [steps, setSteps] = useState(1);
  const [start, setStart] = useState(0);
  const [standardise, setStandardise] = useState(false);
  const [fraction, setFraction] = useState(1);
  const [steps2, setSteps2] = useState(10);

  const colPath = seriesColor(1, dark);
  const colCurve = seriesColor(0, dark);
  const colMin = seriesColor(2, dark);

  const one = useMemo(() => trail1(alpha, start, steps), [alpha, start, steps]);
  const prob = useMemo(() => problem2(standardise), [standardise]);
  const alpha2 = fraction / prob.lambdaMax;
  const path2 = useMemo(() => walk2(prob, alpha2, steps2), [prob, alpha2, steps2]);
  const needed = useMemo(() => stepsToClose(prob, alpha2), [prob, alpha2]);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H1 - PAD.top - PAD.bottom;

  const tMin = -1;
  const tMax = 3;
  const jMax = 8;
  const px = (t: number) => PAD.left + ((t - tMin) / (tMax - tMin)) * innerW;
  const py = (j: number) => PAD.top + innerH - (Math.min(j, jMax) / jMax) * innerH;
  const parabola = Array.from({length: 161}, (_, i) => tMin + ((tMax - tMin) * i) / 160)
    .filter((t) => cost1(t) <= jMax)
    .map((t, i) => `${i ? 'L' : 'M'}${px(t).toFixed(1)},${py(cost1(t)).toFixed(1)}`)
    .join(' ');
  const visible = one.filter((p) => p.theta >= tMin && p.theta <= tMax && p.cost <= jMax);
  const hidden = one.length - visible.length;
  const sx = (k: number) => PAD.left + (k / 40) * innerW;
  const costLine = one.map((p, i) => `${i ? 'L' : 'M'}${sx(p.k).toFixed(1)},${py(p.cost).toFixed(1)}`).join(' ');
  const last = one[one.length - 1];
  const inside = alpha < 2 / CURVATURE;

  const levels = [0.5, 0.2, 0.05, 0.01, 0.001];
  const hx = Math.sqrt((2 * prob.gap0 * prob.H[2]) / (prob.H[0] * prob.H[2] - prob.H[1] ** 2));
  const hy = Math.sqrt((2 * prob.gap0 * prob.H[0]) / (prob.H[0] * prob.H[2] - prob.H[1] ** 2));
  const x0 = prob.best[0] - hx * 1.08;
  const x1 = prob.best[0] + hx * 1.08;
  const y0 = prob.best[1] - hy * 1.08;
  const y1 = prob.best[1] + hy * 1.08;
  const H2 = 280;
  const scale2 = Math.min((W - 20) / (x1 - x0), (H2 - 20) / (y1 - y0));
  const ox = (W - scale2 * (x1 - x0)) / 2;
  const oy = (H2 - scale2 * (y1 - y0)) / 2;
  const qx = (t: number) => ox + (t - x0) * scale2;
  const qy = (t: number) => H2 - oy - (t - y0) * scale2;
  const ellipse = (level: number) => {
    const r = Math.sqrt(2 * level * prob.gap0);
    const a = r / Math.sqrt(prob.lambdaMin);
    const c = r / Math.sqrt(prob.lambdaMax);
    return Array.from({length: 121}, (_, i) => {
      const phi = (2 * Math.PI * i) / 120;
      const u = a * Math.cos(phi);
      const v = c * Math.sin(phi);
      const tx = prob.best[0] + u * prob.vectors[0][0] + v * prob.vectors[1][0];
      const ty = prob.best[1] + u * prob.vectors[0][1] + v * prob.vectors[1][1];
      return `${i ? 'L' : 'M'}${qx(tx).toFixed(1)},${qy(ty).toFixed(1)}`;
    }).join(' ');
  };
  const walked = path2.map((p, i) => `${i ? 'L' : 'M'}${qx(p[0]).toFixed(1)},${qy(p[1]).toFixed(1)}`).join(' ');
  const lastGap = gap2(prob, path2[path2.length - 1]) / prob.gap0;

  const table =
    mode === 'one'
      ? {
          columns: ['step', 'theta', 'gradient', 'cost J'],
          rows: one.map((p) => [
            p.k,
            p.theta.toFixed(4),
            p.gradient === null ? '' : p.gradient.toFixed(4),
            p.cost.toFixed(4),
          ]),
        }
      : {
          columns: ['step', 'theta 1', 'theta 2', 'share of gap left'],
          rows: path2
            .map((p, i) => [i, p[0].toFixed(4), p[1].toFixed(4), (gap2(prob, p) / prob.gap0).toFixed(5)])
            .slice(0, 41),
        };

  const controls =
    mode === 'one' ? (
      <>
        <label className={s.control}>
          mode
          <select className={s.select} value={mode} onChange={(e) => setMode(e.target.value as 'one' | 'two')}>
            <option value="one">One weight (lecture example)</option>
            <option value="two">Two weights, scaling</option>
          </select>
        </label>
        <label className={s.control}>
          learning rate
          <input type="range" min={0.01} max={0.5} step={0.01} value={alpha}
                 onChange={(e) => setAlpha(Number(e.target.value))} />
          <span className={s.value}>{alpha.toFixed(2)}</span>
        </label>
        <label className={s.control}>
          steps
          <input type="range" min={0} max={40} step={1} value={steps}
                 onChange={(e) => setSteps(Number(e.target.value))} />
          <span className={s.value}>{steps}</span>
        </label>
        <label className={s.control}>
          start theta
          <input type="range" min={-1} max={2} step={0.1} value={start}
                 onChange={(e) => setStart(Number(e.target.value))} />
          <span className={s.value}>{start.toFixed(1)}</span>
        </label>
      </>
    ) : (
      <>
        <label className={s.control}>
          mode
          <select className={s.select} value={mode} onChange={(e) => setMode(e.target.value as 'one' | 'two')}>
            <option value="one">One weight (lecture example)</option>
            <option value="two">Two weights, scaling</option>
          </select>
        </label>
        <label className={s.control}>
          <input type="checkbox" checked={standardise} onChange={(e) => setStandardise(e.target.checked)} />
          standardise features
        </label>
        <label className={s.control}>
          rate as share of 1/λmax
          <input type="range" min={0.1} max={1.9} step={0.1} value={fraction}
                 onChange={(e) => setFraction(Number(e.target.value))} />
          <span className={s.value}>{fraction.toFixed(1)}</span>
        </label>
        <label className={s.control}>
          steps
          <input type="range" min={0} max={100} step={1} value={steps2}
                 onChange={(e) => setSteps2(Number(e.target.value))} />
          <span className={s.value}>{steps2}</span>
        </label>
      </>
    );

  return (
    <VizPanel
      title={mode === 'one' ? 'Gradient descent, one step at a time' : 'Why feature scaling speeds up gradient descent'}
      hint={
        mode === 'one'
          ? 'Raise the learning rate past 0.43 and the path flies out of the bowl; drop it to 0.01 and 20 steps barely move. At 0.10 and one step you get the lecture’s numbers: gradient −3.667, theta 0.367.'
          : 'Each setting uses a rate relative to its own limit. Raw features give a long thin bowl and many steps; standardised features give a round bowl and a handful. The rate is a share of 1/λmax; a share of 2.0 would diverge, so the slider stops at 1.9, where the path zig-zags.'
      }
      legend={
        mode === 'one'
          ? [
              {label: 'cost J(theta)', color: colCurve},
              {label: 'gradient descent path', color: colPath},
              {label: 'minimum', color: colMin},
            ]
          : [
              {label: 'contours of the cost', color: colCurve},
              {label: 'gradient descent path', color: colPath},
              {label: 'minimum', color: colMin},
            ]
      }
      table={table}
      controls={controls}>
      {mode === 'one' ? (
        <>
          <div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap'}}>
            <svg className={s.svg} style={{flex: '1 1 260px', minWidth: 0}} viewBox={`0 0 ${W} ${H1}`} role="img"
                 aria-label="Cost curve with the gradient descent path">
              <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} />
              <line className={s.axis} x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + innerH} />
              {[-1, 0, 1, 2, 3].map((t) => (
                <text key={t} className={s.tick} x={px(t)} y={H1 - 16} textAnchor="middle">{t}</text>
              ))}
              {[0, 2, 4, 6, 8].map((j) => (
                <text key={j} className={s.tick} x={PAD.left - 6} y={py(j) + 3} textAnchor="end">{j}</text>
              ))}
              <text className={s.axisLabel} x={W / 2} y={H1 - 2} textAnchor="middle">theta</text>
              <text className={s.axisLabel} x={12} y={PAD.top + innerH / 2} textAnchor="middle"
                    transform={`rotate(-90 12 ${PAD.top + innerH / 2})`}>cost J</text>
              <path d={parabola} fill="none" stroke={colCurve} strokeWidth={2.5} />
              <line x1={px(BEST1)} y1={PAD.top} x2={px(BEST1)} y2={PAD.top + innerH} stroke={colMin}
                    strokeDasharray="3 3" />
              <path d={visible.map((p, i) => `${i ? 'L' : 'M'}${px(p.theta).toFixed(1)},${py(p.cost).toFixed(1)}`).join(' ')}
                    fill="none" stroke={colPath} strokeWidth={1.5} />
              {visible.map((p) => (
                <circle key={p.k} cx={px(p.theta)} cy={py(p.cost)} r={p.k === steps ? 5 : 3} fill={colPath}
                        stroke="var(--surface-raised)" strokeWidth={1.5} />
              ))}
            </svg>
            <svg className={s.svg} style={{flex: '1 1 260px', minWidth: 0}} viewBox={`0 0 ${W} ${H1}`} role="img"
                 aria-label="Cost against step number">
              <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} />
              <line className={s.axis} x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + innerH} />
              {[0, 10, 20, 30, 40].map((k) => (
                <text key={k} className={s.tick} x={sx(k)} y={H1 - 16} textAnchor="middle">{k}</text>
              ))}
              {[0, 2, 4, 6, 8].map((j) => (
                <text key={j} className={s.tick} x={PAD.left - 6} y={py(j) + 3} textAnchor="end">{j}</text>
              ))}
              <text className={s.axisLabel} x={W / 2} y={H1 - 2} textAnchor="middle">step</text>
              <line x1={PAD.left} y1={py(cost1(BEST1))} x2={W - PAD.right} y2={py(cost1(BEST1))} stroke={colMin}
                    strokeDasharray="3 3" />
              <path d={costLine} fill="none" stroke={colPath} strokeWidth={2} />
              <circle cx={sx(last.k)} cy={py(last.cost)} r={4.5} fill={colPath} stroke="var(--surface-raised)" strokeWidth={1.5} />
            </svg>
          </div>
          <p style={{margin: '0.5rem 0 0', fontSize: '0.82rem', color: 'var(--text-muted)'}}>
            step {last.k}: theta = <code>{last.theta.toFixed(4)}</code>
            {last.gradient !== null && (
              <>
                , gradient used = <code>{last.gradient.toFixed(3)}</code>
              </>
            )}
            , J = <code>{last.cost.toFixed(4)}</code>. Minimum at theta = <code>{BEST1.toFixed(4)}</code>, J ={' '}
            <code>{cost1(BEST1).toFixed(4)}</code>. Stability limit 2 / {CURVATURE.toFixed(3)} ={' '}
            <code>{(2 / CURVATURE).toFixed(4)}</code>: this rate is {inside ? 'inside' : 'outside'} it.
            {hidden > 0 ? ` ${hidden} point${hidden === 1 ? '' : 's'} fall off the chart.` : ''}
          </p>
        </>
      ) : (
        <>
          <svg className={s.svg} viewBox={`0 0 ${W} ${H2}`} role="img"
               aria-label="Contours of the two-weight cost with the gradient descent path">
            <defs>
              <clipPath id={clipId}>
                <rect x={0} y={0} width={W} height={H2} />
              </clipPath>
            </defs>
            <g clipPath={`url(#${clipId})`}>
              {levels.map((l) => (
                <path key={l} d={ellipse(l)} fill="none" stroke={colCurve} strokeOpacity={0.55} strokeWidth={1.5} />
              ))}
              <path d={walked} fill="none" stroke={colPath} strokeWidth={1.6} />
              {path2.slice(0, 40).map((p, i) => (
                <circle key={i} cx={qx(p[0])} cy={qy(p[1])} r={i === path2.length - 1 ? 5 : 2.5} fill={colPath} />
              ))}
              {path2.length > 40 && (
                <circle cx={qx(path2[path2.length - 1][0])} cy={qy(path2[path2.length - 1][1])} r={5} fill={colPath} />
              )}
            </g>
            <circle cx={qx(prob.best[0])} cy={qy(prob.best[1])} r={5} fill={colMin} stroke="var(--surface-raised)" strokeWidth={1.5} />
            <circle cx={qx(0)} cy={qy(0)} r={4} fill="none" stroke="var(--text-strong)" strokeWidth={1.5} />
            <text className={s.tick} x={qx(0) - 8} y={qy(0) + 3} textAnchor="end">start</text>
          </svg>
          <p style={{margin: '0.5rem 0 0', fontSize: '0.82rem', color: 'var(--text-muted)'}}>
            eigenvalues <code>{prob.lambdaMin.toFixed(2)}</code> and <code>{prob.lambdaMax.toFixed(2)}</code>, condition number{' '}
            <code>{(prob.lambdaMax / prob.lambdaMin).toFixed(2)}</code>, largest stable rate{' '}
            <code>{(2 / prob.lambdaMax).toFixed(4)}</code>, rate used <code>{alpha2.toFixed(4)}</code>. Steps to close 99.9%
            of the gap: <code>{needed === null ? 'does not converge' : needed}</code>. After {steps2} steps{' '}
            <code>{(lastGap * 100).toFixed(2)}%</code> of the gap remains.
          </p>
        </>
      )}
    </VizPanel>
  );
}
