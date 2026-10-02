import {useId, useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Sample = {x: number; y: number; label: 1 | -1};
type Fit = {w: [number, number]; b: number};

const BASE: Sample[] = [
  {x: 2, y: 2, label: 1},
  {x: 3, y: 1, label: 1},
  {x: 4, y: 3, label: 1},
  {x: 3, y: 4, label: 1},
  {x: 1, y: 1, label: -1},
  {x: 2, y: 0, label: -1},
  {x: 0, y: 0, label: -1},
  {x: 0, y: 1, label: -1},
];
const STRAY: Sample = {x: 1.2, y: 1.3, label: 1};

const LIFT_X = [-4, -3, -2.5, -1.5, -1, 0, 1, 1.5, 2.5, 3, 4];

const W = 640;
const H = 300;
const PAD = {top: 14, right: 16, bottom: 34, left: 40};

export function solveSvm(samples: Sample[], C: number): Fit {
  const n = samples.length;
  const K = samples.map((a) => samples.map((b) => a.x * b.x + a.y * b.y));
  const y = samples.map((p) => p.label as number);
  const alpha = new Array(n).fill(0);
  const G = new Array(n).fill(-1);
  let gmax = 0;
  let gmin = 0;
  for (let iter = 0; iter < 50000; iter += 1) {
    let i = -1;
    let j = -1;
    gmax = -Infinity;
    gmin = Infinity;
    for (let t = 0; t < n; t += 1) {
      const up = (y[t] === 1 && alpha[t] < C) || (y[t] === -1 && alpha[t] > 0);
      const low = (y[t] === 1 && alpha[t] > 0) || (y[t] === -1 && alpha[t] < C);
      const v = -y[t] * G[t];
      if (up && v > gmax) {
        gmax = v;
        i = t;
      }
      if (low && v < gmin) {
        gmin = v;
        j = t;
      }
    }
    if (i < 0 || j < 0 || gmax - gmin < 1e-10) break;
    const eta = Math.max(K[i][i] + K[j][j] - 2 * K[i][j], 1e-12);
    let t = (gmax - gmin) / eta;
    const loI = y[i] === 1 ? -alpha[i] : alpha[i] - C;
    const hiI = y[i] === 1 ? C - alpha[i] : alpha[i];
    const loJ = y[j] === 1 ? alpha[j] - C : -alpha[j];
    const hiJ = y[j] === 1 ? alpha[j] : C - alpha[j];
    t = Math.min(t, hiI, hiJ);
    t = Math.max(t, loI, loJ);
    const di = y[i] * t;
    const dj = -y[j] * t;
    alpha[i] += di;
    alpha[j] += dj;
    for (let m = 0; m < n; m += 1) G[m] += y[m] * y[i] * K[m][i] * di + y[m] * y[j] * K[m][j] * dj;
  }
  const w: [number, number] = [0, 0];
  samples.forEach((p, m) => {
    w[0] += alpha[m] * y[m] * p.x;
    w[1] += alpha[m] * y[m] * p.y;
  });
  return {w, b: (gmax + gmin) / 2};
}

export function rbf(distance: number, gamma: number) {
  return Math.exp(-gamma * distance * distance);
}

export default function SvmMarginLab() {
  const dark = useDarkViz();
  const clipId = useId();
  const [view, setView] = useState<'margin' | 'lift' | 'rbf'>('margin');
  const [angle, setAngle] = useState(45);
  const [offset, setOffset] = useState(2.12);
  const [stray, setStray] = useState(false);
  const [logC, setLogC] = useState(0);
  const [fit, setFit] = useState<Fit | null>(null);
  const [cut, setCut] = useState(4.25);
  const [gamma, setGamma] = useState(0.5);

  const plusColor = seriesColor(0, dark);
  const minusColor = seriesColor(1, dark);
  const warnColor = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;
  const strong = dark ? '#dfe3e9' : '#23262b';

  const C = Math.pow(10, logC);
  const samples = useMemo(() => (stray ? [...BASE, STRAY] : BASE), [stray]);

  const geometry = useMemo(() => {
    let nx: number;
    let ny: number;
    let c: number;
    let half: number;
    let scores: number[];
    let wrong: boolean[];
    let support: boolean[];
    if (fit) {
      const norm = Math.hypot(fit.w[0], fit.w[1]);
      nx = fit.w[0] / norm;
      ny = fit.w[1] / norm;
      c = -fit.b / norm;
      half = 1 / norm;
      scores = samples.map((p) => fit.w[0] * p.x + fit.w[1] * p.y + fit.b);
      wrong = samples.map((p, i) => p.label * scores[i] <= 0);
      support = samples.map((p, i) => p.label * scores[i] <= 1 + 1e-5);
    } else {
      const rad = (angle * Math.PI) / 180;
      nx = Math.cos(rad);
      ny = Math.sin(rad);
      c = offset;
      const dist = samples.map((p) => nx * p.x + ny * p.y - c);
      wrong = samples.map((p, i) => p.label * dist[i] <= 0);
      const good = dist.filter((_, i) => !wrong[i]).map(Math.abs);
      half = good.length ? Math.min(...good) : 0;
      scores = dist;
      support = samples.map((_, i) => !wrong[i] && Math.abs(Math.abs(dist[i]) - half) < 0.02);
    }
    return {nx, ny, c, half, scores, wrong, support};
  }, [fit, angle, offset, samples]);

  const slackCount = fit
    ? samples.filter((p) => p.label * (fit.w[0] * p.x + fit.w[1] * p.y + fit.b) < 1 - 1e-6).length
    : 0;
  const errors = geometry.wrong.filter(Boolean).length;
  const supportCount = geometry.support.filter(Boolean).length;
  const marginWidth = 2 * geometry.half;

  const lo = -1;
  const hi = 5;
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const px = (v: number) => PAD.left + ((v - lo) / (hi - lo)) * innerW;
  const py = (v: number) => PAD.top + innerH - ((v - lo) / (hi - lo)) * innerH;

  const linePath = (shift: number) => {
    const {nx, ny, c} = geometry;
    const cx = nx * (c + shift);
    const cy = ny * (c + shift);
    const tx = -ny;
    const ty = nx;
    return `M${px(cx - tx * 12)},${py(cy - ty * 12)} L${px(cx + tx * 12)},${py(cy + ty * 12)}`;
  };
  const band = () => {
    const {nx, ny, c, half} = geometry;
    const tx = -ny;
    const ty = nx;
    const corner = (shift: number, sign: number) => {
      const cx = nx * (c + shift) + sign * tx * 12;
      const cy = ny * (c + shift) + sign * ty * 12;
      return `${px(cx)},${py(cy)}`;
    };
    return [corner(-half, -1), corner(-half, 1), corner(half, 1), corner(half, -1)].join(' ');
  };

  const onFit = () => {
    const result = solveSvm(samples, C);
    const norm = Math.hypot(result.w[0], result.w[1]);
    const deg = ((Math.atan2(result.w[1], result.w[0]) * 180) / Math.PI + 360) % 360;
    setAngle(Math.round(deg));
    setOffset(Math.min(5, Math.max(0, Math.round((-result.b / norm) * 100) / 100)));
    setFit(result);
  };

  const liftErrors = LIFT_X.filter((x) => {
    const label = x * x <= 2.25 + 1e-9 ? 1 : -1;
    const predicted = x * x < cut ? 1 : -1;
    return label !== predicted;
  }).length;
  const cutX = Math.sqrt(cut);

  const select = (
    <label className={s.control}>
      view
      <select className={s.select} value={view} onChange={(e) => setView(e.target.value as typeof view)}>
        <option value="margin">margin and C</option>
        <option value="lift">kernel lift (1-D to 2-D)</option>
        <option value="rbf">RBF similarity</option>
      </select>
    </label>
  );

  const controls =
    view === 'margin' ? (
      <>
        {select}
        <label className={s.control}>
          angle
          <input type="range" min={0} max={359} step={1} value={angle}
                 onChange={(e) => { setFit(null); setAngle(Number(e.target.value)); }} />
          <span className={s.value}>{angle}°</span>
        </label>
        <label className={s.control}>
          offset
          <input type="range" min={0} max={6} step={0.01} value={offset}
                 onChange={(e) => { setFit(null); setOffset(Number(e.target.value)); }} />
          <span className={s.value}>{offset.toFixed(2)}</span>
        </label>
        <label className={s.control}>
          <input type="checkbox" checked={stray} onChange={(e) => { setFit(null); setStray(e.target.checked); }} />
          add the stray + at (1.2, 1.3)
        </label>
        <label className={s.control}>
          C
          <input type="range" min={-2} max={2} step={0.25} value={logC}
                 onChange={(e) => setLogC(Number(e.target.value))} />
          <span className={s.value}>{C >= 1 ? C.toFixed(C < 10 ? 2 : 0) : C.toFixed(3)}</span>
        </label>
        <button type="button" className={s.button} onClick={onFit}>fit for this C</button>
      </>
    ) : view === 'lift' ? (
      <>
        {select}
        <label className={s.control}>
          cut height x²
          <input type="range" min={0} max={16} step={0.05} value={cut}
                 onChange={(e) => setCut(Number(e.target.value))} />
          <span className={s.value}>{cut.toFixed(2)}</span>
        </label>
      </>
    ) : (
      <>
        {select}
        <label className={s.control}>
          gamma
          <input type="range" min={0.05} max={3} step={0.05} value={gamma}
                 onChange={(e) => setGamma(Number(e.target.value))} />
          <span className={s.value}>{gamma.toFixed(2)}</span>
        </label>
      </>
    );

  const table =
    view === 'margin'
      ? {
          columns: ['point', 'class', 'score', 'side', 'support'],
          rows: samples.map((p, i) => [
            `(${p.x}, ${p.y})`,
            p.label === 1 ? '+' : '-',
            geometry.scores[i].toFixed(2),
            geometry.wrong[i] ? 'wrong' : 'right',
            geometry.support[i] ? 'yes' : 'no',
          ]),
        }
      : view === 'lift'
        ? {
            columns: ['x', 'class', 'lifted (x, x²)', 'side of the cut'],
            rows: LIFT_X.map((x) => [
              x,
              x * x <= 2.25 + 1e-9 ? '+' : '-',
              `(${x}, ${(x * x).toFixed(2)})`,
              x * x < cut ? 'below' : 'above',
            ]),
          }
        : {
            columns: ['distance', 'similarity'],
            rows: [0, 0.5, 1, 1.5, 2, 3, 3.606, 4, 5].map((d) => [d, rbf(d, gamma).toFixed(4)]),
          };

  const hint =
    view === 'margin'
      ? 'Default: the line x₁ + x₂ = 3 (angle 45°, offset 2.12), margin 1.41 = √2 and no errors; the four ringed points are the support vectors. Drag the sliders to tilt it and watch the margin shrink or points go wrong. Press "fit for this C" for the exact SVM solution; tick the stray point and sweep C to see the margin trade against mistakes.'
      : view === 'lift'
        ? 'A line on the number line cannot put the + points in the middle and the - points outside. Lift each point to (x, x²) and a horizontal cut does it; mapped back, the cut becomes the two points ±√height. Default height 4.25 falls at ±2.062 with no errors.'
        : 'The RBF kernel scores similarity as exp(−γ d²): 1 for identical points, falling towards 0 with distance. Larger gamma shrinks each example’s reach. At gamma 0.5 the points (1, 2) and (3, −1), distance √13 ≈ 3.606, have similarity 0.0015.';

  const liftW = innerW;
  const lx = (v: number) => PAD.left + ((v + 4.5) / 9) * liftW;
  const ly = (v: number) => PAD.top + 150 - (v / 20) * 150;
  const rx = (v: number) => PAD.left + (v / 5) * innerW;
  const ry = (v: number) => PAD.top + innerH - v * innerH;

  return (
    <VizPanel
      title={view === 'margin' ? 'Tilt the boundary' : view === 'lift' ? 'The kernel idea: lift, cut, map back' : 'RBF similarity'}
      hint={hint}
      legend={
        view === 'rbf'
          ? [{label: 'similarity', color: plusColor}]
          : [
              {label: '+ class (circle)', color: plusColor},
              {label: '- class (square)', color: minusColor},
              ...(view === 'margin' ? [{label: 'mistake (dashed ring)', color: warnColor}] : []),
            ]
      }
      table={table}
      controls={controls}>
      {view === 'margin' && (
        <>
          <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
               aria-label="Two classes, a separating line and its margin band">
            <defs>
              <clipPath id={clipId}>
                <rect x={PAD.left} y={PAD.top} width={innerW} height={innerH} />
              </clipPath>
            </defs>
            {[-1, 0, 1, 2, 3, 4, 5].map((v) => (
              <g key={v}>
                <line className={s.grid} x1={px(v)} y1={PAD.top} x2={px(v)} y2={PAD.top + innerH} />
                <line className={s.grid} x1={PAD.left} y1={py(v)} x2={PAD.left + innerW} y2={py(v)} />
                <text className={s.tick} x={px(v)} y={H - 18} textAnchor="middle">{v}</text>
                <text className={s.tick} x={PAD.left - 6} y={py(v) + 3} textAnchor="end">{v}</text>
              </g>
            ))}
            <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 3} textAnchor="middle">x₁</text>
            <g clipPath={`url(#${clipId})`}>
              <polygon points={band()} fill={plusColor} opacity={0.1} />
              <path d={linePath(0)} stroke={strong} strokeWidth={2.2} fill="none" />
              <path d={linePath(geometry.half)} stroke="var(--text-faint)" strokeWidth={1.4} strokeDasharray="5 4" fill="none" />
              <path d={linePath(-geometry.half)} stroke="var(--text-faint)" strokeWidth={1.4} strokeDasharray="5 4" fill="none" />
            </g>
            {samples.map((p, i) => {
              const color = p.label === 1 ? plusColor : minusColor;
              return (
                <g key={`${p.x}-${p.y}`}>
                  {p.label === 1 ? (
                    <circle cx={px(p.x)} cy={py(p.y)} r={6} fill={color} />
                  ) : (
                    <rect x={px(p.x) - 6} y={py(p.y) - 6} width={12} height={12} fill={color} />
                  )}
                  {geometry.support[i] && (
                    <circle cx={px(p.x)} cy={py(p.y)} r={11} fill="none" stroke={strong} strokeWidth={2} />
                  )}
                  {geometry.wrong[i] && (
                    <circle cx={px(p.x)} cy={py(p.y)} r={14} fill="none" stroke={warnColor}
                            strokeWidth={2} strokeDasharray="3 3" />
                  )}
                </g>
              );
            })}
          </svg>
          <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem'}}>
            <span>
              {fit
                ? `exact fit for C = ${C.toFixed(C < 1 ? 3 : 2)}: w = (${fit.w[0].toFixed(3)}, ${fit.w[1].toFixed(3)}), b = ${fit.b.toFixed(3)}, margin 2/‖w‖ = `
                : 'hand-set line: margin = '}
              <strong>{marginWidth.toFixed(2)}</strong>
              {fit ? ` · support vectors ${supportCount} · inside or past the margin ${slackCount}` : ` · support points ${supportCount}`}
              {` · mistakes ${errors} of ${samples.length}`}
            </span>
          </div>
        </>
      )}
      {view === 'lift' && (
        <>
          <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
               aria-label="One-dimensional points lifted to a parabola and cut by a horizontal line">
            <line className={s.axis} x1={PAD.left} y1={ly(0)} x2={PAD.left + liftW} y2={ly(0)} />
            <line className={s.grid} x1={PAD.left} y1={ly(cut)} x2={PAD.left + liftW} y2={ly(cut)}
                  stroke={strong} strokeWidth={2} />
            <text className={s.dataLabel} x={PAD.left + liftW - 4} y={ly(cut) - 6} textAnchor="end">
              cut at x² = {cut.toFixed(2)}
            </text>
            <text className={s.axisLabel} x={PAD.left - 6} y={PAD.top + 8} textAnchor="end">x²</text>
            {LIFT_X.map((x) => {
              const plus = x * x <= 2.25 + 1e-9;
              const color = plus ? plusColor : minusColor;
              return plus ? (
                <circle key={x} cx={lx(x)} cy={ly(x * x)} r={6} fill={color} />
              ) : (
                <rect key={x} x={lx(x) - 6} y={ly(x * x) - 6} width={12} height={12} fill={color} />
              );
            })}
            <rect x={lx(-cutX)} y={ly(0) + 44} width={lx(cutX) - lx(-cutX)} height={42} fill={plusColor} opacity={0.14} />
            <line x1={PAD.left} y1={ly(0) + 65} x2={PAD.left + liftW} y2={ly(0) + 65} stroke="var(--border-strong)" />
            <line x1={lx(-cutX)} y1={ly(0) + 40} x2={lx(-cutX)} y2={ly(0) + 90} stroke={strong} strokeWidth={2} />
            <line x1={lx(cutX)} y1={ly(0) + 40} x2={lx(cutX)} y2={ly(0) + 90} stroke={strong} strokeWidth={2} />
            {LIFT_X.map((x) => {
              const plus = x * x <= 2.25 + 1e-9;
              const color = plus ? plusColor : minusColor;
              return plus ? (
                <circle key={`n${x}`} cx={lx(x)} cy={ly(0) + 65} r={5} fill={color} />
              ) : (
                <rect key={`n${x}`} x={lx(x) - 5} y={ly(0) + 60} width={10} height={10} fill={color} />
              );
            })}
            {[-4, -2, 0, 2, 4].map((v) => (
              <text key={v} className={s.tick} x={lx(v)} y={ly(0) + 104} textAnchor="middle">{v}</text>
            ))}
            <text className={s.axisLabel} x={PAD.left + liftW / 2} y={H - 3} textAnchor="middle">
              the same points back on the number line, shaded where the cut says +
            </text>
          </svg>
          <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem'}}>
            <span>
              cut at x² = {cut.toFixed(2)} falls on the line at x = ±<strong>{cutX.toFixed(3)}</strong> · mistakes{' '}
              <strong>{liftErrors}</strong> of {LIFT_X.length}
            </span>
          </div>
        </>
      )}
      {view === 'rbf' && (
        <>
          <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="RBF similarity against distance">
            {[0, 1, 2, 3, 4, 5].map((v) => (
              <g key={v}>
                <line className={s.grid} x1={rx(v)} y1={PAD.top} x2={rx(v)} y2={PAD.top + innerH} />
                <text className={s.tick} x={rx(v)} y={H - 18} textAnchor="middle">{v}</text>
              </g>
            ))}
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <g key={v}>
                <line className={s.grid} x1={PAD.left} y1={ry(v)} x2={PAD.left + innerW} y2={ry(v)} />
                <text className={s.tick} x={PAD.left - 6} y={ry(v) + 3} textAnchor="end">{v}</text>
              </g>
            ))}
            <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 3} textAnchor="middle">
              distance ‖x − z‖
            </text>
            <path
              d={Array.from({length: 101}, (_, i) => {
                const d = (i / 100) * 5;
                return `${i ? 'L' : 'M'}${rx(d).toFixed(1)},${ry(rbf(d, gamma)).toFixed(1)}`;
              }).join(' ')}
              fill="none"
              stroke={plusColor}
              strokeWidth={2.5}
            />
            <circle cx={rx(Math.sqrt(13))} cy={ry(rbf(Math.sqrt(13), gamma))} r={5} fill={strong} />
            <text className={s.dataLabel} x={rx(Math.sqrt(13)) + 8} y={ry(rbf(Math.sqrt(13), gamma)) - 8}>
              d = √13: {rbf(Math.sqrt(13), gamma).toFixed(4)}
            </text>
          </svg>
          <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem'}}>
            <span>
              gamma = {gamma.toFixed(2)}: similarity at distance √13 ≈ 3.606 is{' '}
              <strong>{rbf(Math.sqrt(13), gamma).toFixed(4)}</strong>
            </span>
          </div>
        </>
      )}
    </VizPanel>
  );
}
