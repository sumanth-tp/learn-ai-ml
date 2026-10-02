import {useMemo, useState} from 'react';

import {calibrationRun, type CalibrationMethod} from './evalMath';
import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 360;
const H = 310;
const BOX = 250;
const ORIGIN = {x: 52, y: 20};
const METHODS: {id: CalibrationMethod; label: string}[] = [
  {id: 'none', label: 'none'},
  {id: 'platt', label: 'Platt (sigmoid)'},
  {id: 'isotonic', label: 'isotonic'},
];

export default function CalibrationLab() {
  const dark = useDarkViz();
  const [slope, setSlope] = useState(0.6);
  const [shift, setShift] = useState(0);
  const [n, setN] = useState(500);
  const [method, setMethod] = useState<CalibrationMethod>('none');

  const runs = useMemo(
    () => ({
      none: calibrationRun(n, slope, shift, 'none'),
      platt: calibrationRun(n, slope, shift, 'platt'),
      isotonic: calibrationRun(n, slope, shift, 'isotonic'),
    }),
    [n, slope, shift],
  );
  const run = runs[method];

  const dotColor = seriesColor(0, dark);
  const mapColor = seriesColor(1, dark);
  const diag = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;

  const px = (v: number) => ORIGIN.x + v * BOX;
  const py = (v: number) => ORIGIN.y + (1 - v) * BOX;
  const mapPath = Array.from({length: 101}, (_, i) => {
    const q = (i + 0.5) / 101;
    return `${i ? 'L' : 'M'}${px(q).toFixed(1)},${py(run.map(q)).toFixed(1)}`;
  }).join(' ');
  const maxCount = Math.max(1, ...run.bins.map((b) => b.count));

  return (
    <VizPanel
      title="Reliability diagram and recalibration"
      hint="Dots on the diagonal mean the model's 70% really happens 70% of the time. With slope 0.6 the model is overconfident. Platt scaling fixes a smooth bend with few points; isotonic can fit any monotone shape but wobbles when n is small. Defaults match the chapter: Brier 0.2109 and ECE 0.0780 before calibration."
      legend={[
        {label: 'observed rate in a bin (size = cases)', color: dotColor},
        {label: 'calibrator mapping', color: mapColor},
        {label: 'perfect calibration', color: diag},
      ]}
      table={{
        columns: ['bin', 'cases', 'mean predicted', 'observed rate'],
        rows: run.bins.map((b, i) => [
          `${(i / 10).toFixed(1)} to ${((i + 1) / 10).toFixed(1)}`,
          b.count,
          b.count ? b.predicted.toFixed(3) : '-',
          b.count ? b.observed.toFixed(3) : '-',
        ]),
      }}
      controls={
        <>
          <label className={s.control}>
            true slope
            <input type="range" min={0.3} max={1.8} step={0.05} value={slope}
                   onChange={(e) => setSlope(Number(e.target.value))} />
            <span className={s.value}>{slope.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            shift
            <input type="range" min={-1.5} max={1.5} step={0.1} value={shift}
                   onChange={(e) => setShift(Number(e.target.value))} />
            <span className={s.value}>{shift.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            cases
            <input type="range" min={40} max={2000} step={20} value={n}
                   onChange={(e) => setN(Number(e.target.value))} />
            <span className={s.value}>{n}</span>
          </label>
          <label className={s.control}>
            calibration
            <select className={s.select} value={method} onChange={(e) => setMethod(e.target.value as CalibrationMethod)}>
              {METHODS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        </>
      }>
      <svg className={s.svg} style={{maxWidth: 360, margin: '0 auto'}} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Reliability diagram of predicted probability against observed rate">
        <rect x={ORIGIN.x} y={ORIGIN.y} width={BOX} height={BOX} fill="none" stroke="var(--border-strong)" />
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <text className={s.tick} x={px(t)} y={ORIGIN.y + BOX + 14} textAnchor="middle">{t}</text>
            <text className={s.tick} x={ORIGIN.x - 6} y={py(t) + 3} textAnchor="end">{t}</text>
          </g>
        ))}
        <text className={s.axisLabel} x={ORIGIN.x + BOX / 2} y={ORIGIN.y + BOX + 32} textAnchor="middle">
          predicted probability
        </text>
        <text className={s.axisLabel} x={ORIGIN.x - 34} y={ORIGIN.y + BOX / 2} textAnchor="middle"
              transform={`rotate(-90 ${ORIGIN.x - 34} ${ORIGIN.y + BOX / 2})`}>observed rate</text>
        <line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} stroke={diag} strokeDasharray="4 4" />
        <path d={mapPath} fill="none" stroke={mapColor} strokeWidth={1.5} opacity={0.7} />
        {run.bins.map((b, i) =>
          b.count ? (
            <circle key={i} cx={px(b.predicted)} cy={py(b.observed)} r={3 + 7 * Math.sqrt(b.count / maxCount)}
                    fill={dotColor} fillOpacity={0.85} stroke="var(--surface-raised)" strokeWidth={1.5} />
          ) : null,
        )}
      </svg>
      <div style={{marginTop: '0.6rem', fontSize: '0.8rem', color: 'var(--text-muted)'}}>
        <div style={{fontWeight: 600, marginBottom: '0.25rem'}}>scored on the test half ({run.yTest.length} cases)</div>
        {METHODS.map((m) => {
          const r = runs[m.id];
          return (
            <div key={m.id} style={{fontWeight: m.id === method ? 700 : 400, color: m.id === method ? 'var(--text-strong)' : undefined}}>
              {m.id === method ? '> ' : ''}{m.label}: Brier {r.brier.toFixed(4)}, ECE {r.ece.toFixed(4)}
            </div>
          );
        })}
      </div>
    </VizPanel>
  );
}
