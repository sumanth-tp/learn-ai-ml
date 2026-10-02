import {useMemo, useState} from 'react';

import {ESTIMATE_TASKS, fmt, percentile, simulateProject} from './craftMath';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const PAD = {top: 22, right: 24, bottom: 40, left: 44};
const BIN = 2;
const X_MIN = 30;
const X_MAX = 130;
const LIKELY = ESTIMATE_TASKS.reduce((a, t) => a + t.mode, 0);

export default function EstimateRangeLab() {
  const dark = useDarkViz();
  const [bad, setBad] = useState(0.35);
  const [slowdown, setSlowdown] = useState(1.5);
  const [shared, setShared] = useState(true);
  const [pct, setPct] = useState(85);

  const totals = useMemo(() => simulateProject(bad, slowdown, shared), [bad, slowdown, shared]);
  const mean = totals.reduce((a, v) => a + v, 0) / totals.length;
  const p50 = percentile(totals, 50);
  const pSel = percentile(totals, pct);
  const within = totals.filter((v) => v <= LIKELY).length / totals.length;

  const bins = useMemo(() => {
    const counts = new Array((X_MAX - X_MIN) / BIN).fill(0);
    totals.forEach((v) => {
      const i = Math.min(counts.length - 1, Math.max(0, Math.floor((v - X_MIN) / BIN)));
      counts[i] += 1;
    });
    return counts as number[];
  }, [totals]);

  const maxCount = Math.max(...bins);
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const x = (v: number) => PAD.left + ((v - X_MIN) / (X_MAX - X_MIN)) * innerW;
  const barW = innerW / bins.length;
  const color = seriesColor(0, dark);
  const accent = seriesColor(1, dark);

  const rows = [10, 25, 50, 70, 85, 95].map((q) => [`P${q}`, percentile(totals, q).toFixed(1)]);
  const status = `mean ${mean.toFixed(1)} days, P50 ${p50.toFixed(1)}, P${pct} ${pSel.toFixed(1)}; ${(within * 100).toFixed(1)}% of runs finish within the ${LIKELY} likely days`;

  return (
    <VizPanel
      title="How long will the project take? A simulated range"
      hint="Defaults reproduce the chapter: shared data risk at 35% and 1.5x gives a mean of 59.6 days, P50 58.1 and P85 70.9. Switch to independent and the mean stays at 59.5 but P85 falls to 69.0: the same expected slowdown, a thinner tail. Raise the slowdown to 2.5 and the shared tail grows fastest."
      legend={[
        {label: 'simulated projects', color},
        {label: `P${pct} and the sum of likely days`, color: accent},
      ]}
      table={{columns: ['percentile', 'days'], rows}}
      controls={
        <>
          <label className={s.control}>
            chance data is worse than assumed
            <input type="range" min={0} max={0.8} step={0.05} value={bad} onChange={(e) => setBad(Number(e.target.value))} />
            <span className={s.value}>{bad.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            slowdown if it is
            <input type="range" min={1} max={3} step={0.25} value={slowdown} onChange={(e) => setSlowdown(Number(e.target.value))} />
            <span className={s.value}>{slowdown.toFixed(2)}x</span>
          </label>
          <label className={s.control}>
            risk model
            <select className={s.select} value={shared ? 'shared' : 'independent'} onChange={(e) => setShared(e.target.value === 'shared')}>
              <option value="shared">shared: one data verdict for the project</option>
              <option value="independent">independent: each task rolls its own</option>
            </select>
          </label>
          <label className={s.control}>
            percentile to read
            <input type="range" min={50} max={95} step={5} value={pct} onChange={(e) => setPct(Number(e.target.value))} />
            <span className={s.value}>P{pct}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Histogram of simulated project length in days. ${status}`}>
        <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} />
        {[30, 40, 50, 60, 70, 80, 90, 100, 110, 120].map((t) => (
          <text key={t} className={s.tick} x={x(t)} y={H - 20} textAnchor="middle">
            {t}
          </text>
        ))}
        <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 4} textAnchor="middle">
          project length in working days
        </text>
        {bins.map((c, i) => {
          const h = (c / maxCount) * innerH;
          return <rect key={i} x={PAD.left + i * barW + 0.5} y={PAD.top + innerH - h} width={barW - 1} height={h} fill={color} opacity={0.8} />;
        })}
        <line x1={x(LIKELY)} y1={PAD.top} x2={x(LIKELY)} y2={PAD.top + innerH} stroke="var(--text-strong)" strokeWidth={1.5} strokeDasharray="3 3" />
        <text className={s.dataLabel} x={x(LIKELY)} y={PAD.top - 6} textAnchor="middle">
          sum of likely {LIKELY}
        </text>
        <line x1={x(pSel)} y1={PAD.top} x2={x(pSel)} y2={PAD.top + innerH} stroke={accent} strokeWidth={2.5} />
        <text className={s.dataLabel} x={x(pSel)} y={PAD.top - 6} textAnchor="middle">
          P{pct} {fmt(pSel, 1)}
        </text>
      </svg>
    </VizPanel>
  );
}
