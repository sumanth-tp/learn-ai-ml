import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 270;
const PAD = {left: 44, right: 24, top: 18, bottom: 40};
const SPAN = 12;

const EPSILONS = [0.01, 0.1, 0.5, 1, 5];

const GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10];

const density = (x: number, centre: number, eps: number) => (eps / 2) * Math.exp(-eps * Math.abs(x - centre));

export default function PrivacyBudgetLab() {
  const dark = useDarkViz();
  const [slider, setSlider] = useState(GRID.indexOf(0.5));
  const [probe, setProbe] = useState(3);
  const [releases, setReleases] = useState(1);
  const [budget, setBudget] = useState(1);

  const eps = GRID[slider];
  const scale = 1 / eps;
  const ratio = Math.exp(-Math.abs(probe) * eps) / Math.exp(-Math.abs(probe - 1) * eps);
  const spent = releases * eps;
  const over = spent > budget;

  const colA = seriesColor(0, dark);
  const colB = seriesColor(1, dark);
  const mid = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;
  const bad = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const peak = Math.max(eps / 2, 0.02);
  const xs = Array.from({length: 241}, (_, i) => -SPAN + (i / 240) * 2 * SPAN);
  const px = (x: number) => PAD.left + ((x + SPAN) / (2 * SPAN)) * innerW;
  const py = (d: number) => PAD.top + innerH - (d / (peak * 1.1)) * innerH;
  const path = (centre: number) =>
    xs.map((x, i) => `${i ? 'L' : 'M'}${px(x).toFixed(1)},${py(density(x, centre, eps)).toFixed(1)}`).join(' ');

  const table = EPSILONS.map((e) => [e.toString(), (1 / e).toFixed(1), (1 / e).toFixed(2)]);

  return (
    <VizPanel
      title="Laplace mechanism: epsilon, noise and the privacy budget"
      hint="Defaults are the chapter's counting query: epsilon 0.5 gives noise scale 2.0, and one person more or fewer changes the density at an output 3 above the true count by a factor of 0.607 (the bound is 1.649). Raise the number of releases to watch the budget run out."
      legend={[
        {label: 'released value, true count c', color: colA},
        {label: 'released value if one person is added (c + 1)', color: colB},
        {label: 'probe output', color: mid},
      ]}
      table={{columns: ['epsilon', 'noise scale', 'expected absolute error'], rows: table}}
      controls={
        <>
          <label className={s.control}>
            epsilon
            <input type="range" min={0} max={GRID.length - 1} step={1} value={slider} onChange={(e) => setSlider(Number(e.target.value))} />
            <span className={s.value}>{eps < 0.1 ? eps.toFixed(3) : eps.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            probe output
            <input type="range" min={-8} max={8} step={1} value={probe} onChange={(e) => setProbe(Number(e.target.value))} />
            <span className={s.value}>{probe >= 0 ? `+${probe}` : probe}</span>
          </label>
          <label className={s.control}>
            releases
            <input type="range" min={1} max={100} step={1} value={releases} onChange={(e) => setReleases(Number(e.target.value))} />
            <span className={s.value}>{releases}</span>
          </label>
          <label className={s.control}>
            budget
            <select className={s.select} value={budget} onChange={(e) => setBudget(Number(e.target.value))}>
              {[1, 2, 5, 10].map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </label>
          <span className={s.value} aria-live="polite">
            scale {scale.toFixed(1)}, ratio {ratio.toFixed(3)} (bound {Math.exp(eps).toFixed(3)}), spent {spent.toFixed(2)} of {budget}
            {over ? ' (over budget)' : ''}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Two Laplace densities one count apart at epsilon ${eps.toFixed(2)}`}>
        <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} />
        {[-10, -5, 0, 5, 10].map((t) => (
          <text key={t} className={s.tick} x={px(t)} y={H - 20} textAnchor="middle">{t >= 0 ? `+${t}` : t}</text>
        ))}
        <text className={s.axisLabel} x={W / 2} y={H - 4} textAnchor="middle">released value minus the true count</text>
        <path d={path(1)} fill="none" stroke={colB} strokeWidth={2.5} strokeDasharray="6 3" />
        <path d={path(0)} fill="none" stroke={colA} strokeWidth={2.5} />
        <line x1={px(probe)} y1={PAD.top} x2={px(probe)} y2={PAD.top + innerH} stroke={mid} strokeWidth={1.5} strokeDasharray="3 3" />
        <circle cx={px(probe)} cy={py(density(probe, 0, eps))} r={4.5} fill={colA} stroke="var(--surface-raised)" strokeWidth={2} />
        <circle cx={px(probe)} cy={py(density(probe, 1, eps))} r={4.5} fill={colB} stroke="var(--surface-raised)" strokeWidth={2} />
        <text className={s.dataLabel} x={W - PAD.right} y={PAD.top + 12} textAnchor="end" fill={over ? bad : undefined}>
          {releases} release{releases === 1 ? '' : 's'}: total epsilon {spent.toFixed(2)}
        </text>
      </svg>
    </VizPanel>
  );
}
