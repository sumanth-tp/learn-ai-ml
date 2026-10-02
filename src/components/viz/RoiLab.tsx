import {useMemo, useState} from 'react';

import {costPerTask, fmt, roiSummary} from './craftMath';
import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const PAD = {top: 20, right: 24, bottom: 40, left: 72};

export default function RoiLab() {
  const dark = useDarkViz();
  const [adoption, setAdoption] = useState(0.6);
  const [minutesSaved, setMinutesSaved] = useState(2.5);
  const [rework, setRework] = useState(0.25);
  const [realisation, setRealisation] = useState(0.6);
  const [buildCost, setBuildCost] = useState(90000);
  const [steps, setSteps] = useState(6);

  const r = useMemo(
    () => roiSummary({adoption, minutesSaved, rework, realisation, buildCost, steps}),
    [adoption, minutesSaved, rework, realisation, buildCost, steps],
  );

  const lo = Math.min(...r.cumulative, 0);
  const hi = Math.max(...r.cumulative, 0);
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const x = (m: number) => PAD.left + (m / 24) * innerW;
  const y = (v: number) => PAD.top + innerH - ((v - lo) / (hi - lo || 1)) * innerH;
  const line = r.cumulative.map((v, m) => `${m ? 'L' : 'M'}${x(m).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const color = seriesColor(0, dark);
  const zero = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;
  const payback = Number.isFinite(r.payback) && r.payback <= 24 ? r.payback : null;

  const rows = [0, 3, 6, 9, 12, 18, 24].map((m) => [m, fmt(r.cumulative[m])]);
  const status = `monthly benefit ${fmt(r.benefit)}, cost ${fmt(r.cost)}, net ${fmt(r.net)}; payback ${
    Number.isFinite(r.payback) ? r.payback.toFixed(1) + ' months' : 'never'
  }; 24-month NPV ${fmt(r.npv)}; ROI ${r.roi.toFixed(2)}; cost per task ${costPerTask(steps).toFixed(5)}`;

  return (
    <VizPanel
      title="ROI of an LLM feature: cumulative cash over 24 months"
      hint="Defaults reproduce the chapter's base case: monthly net 9,686, payback 9.3 months, 24-month NPV 120,810 and ROI 1.58. Then lower realisation to 0.3 or minutes saved to 1 and watch the NPV go negative: the point estimate hides how fragile it is."
      legend={[{label: 'cumulative net cash', color}]}
      table={{columns: ['month', 'cumulative cash'], rows}}
      controls={
        <>
          <label className={s.control}>
            adoption
            <input type="range" min={0.1} max={1} step={0.05} value={adoption} onChange={(e) => setAdoption(Number(e.target.value))} />
            <span className={s.value}>{adoption.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            minutes saved per useful task
            <input type="range" min={0.5} max={5} step={0.5} value={minutesSaved} onChange={(e) => setMinutesSaved(Number(e.target.value))} />
            <span className={s.value}>{minutesSaved.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            rework share
            <input type="range" min={0} max={0.6} step={0.05} value={rework} onChange={(e) => setRework(Number(e.target.value))} />
            <span className={s.value}>{rework.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            realisation
            <input type="range" min={0.2} max={1} step={0.05} value={realisation} onChange={(e) => setRealisation(Number(e.target.value))} />
            <span className={s.value}>{realisation.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            build cost
            <input type="range" min={30000} max={200000} step={5000} value={buildCost} onChange={(e) => setBuildCost(Number(e.target.value))} />
            <span className={s.value}>{fmt(buildCost)}</span>
          </label>
          <label className={s.control}>
            agent steps per task
            <input type="range" min={1} max={20} step={1} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
            <span className={s.value}>{steps}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Cumulative net cash by month. ${status}`}>
        <line x1={PAD.left} y1={y(0)} x2={W - PAD.right} y2={y(0)} stroke={zero} strokeWidth={1.5} strokeDasharray="4 3" />
        <line className={s.axis} x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + innerH} />
        {[0, 6, 12, 18, 24].map((m) => (
          <text key={m} className={s.tick} x={x(m)} y={H - 20} textAnchor="middle">
            {m}
          </text>
        ))}
        {[lo, 0, hi].map((v, i) => (
          <text key={i} className={s.tick} x={PAD.left - 6} y={y(v) + 3} textAnchor="end">
            {fmt(v)}
          </text>
        ))}
        <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 4} textAnchor="middle">
          month
        </text>
        <path d={line} fill="none" stroke={color} strokeWidth={2.5} />
        {payback !== null && (
          <g>
            <circle cx={x(payback)} cy={y(0)} r={5.5} fill={color} stroke="var(--surface-raised)" strokeWidth={1.5} />
            <text className={s.dataLabel} x={x(payback)} y={y(0) - 10} textAnchor="middle">
              payback {payback.toFixed(1)}
            </text>
          </g>
        )}
      </svg>
    </VizPanel>
  );
}
