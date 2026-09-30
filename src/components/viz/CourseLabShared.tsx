import {useId} from 'react';
import {seriesColor} from './palette';
import {useDarkViz, vizStyles as s} from './VizPanel';
import c from './CourseLab.module.css';

export function Slider({label, value, min, max, step = 1, onChange}: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void;
}) {
  const id = useId();
  return <label className={c.control} htmlFor={id}>{label}
    <input id={id} type="range" value={value} min={min} max={max} step={step}
      onChange={event => onChange(Number(event.target.value))} />
    <span className={s.value}>{step < 1 ? value.toFixed(2) : value}</span>
  </label>;
}

export function Score({value, label, detail}: {value: number; label: string; detail?: string}) {
  const dark = useDarkViz();
  return <>
    <div className={c.score} aria-live="polite" aria-atomic="true">
      <output aria-label={label}>{value.toFixed(4)}</output><strong>{label}</strong>
      {detail && <span className={c.badge}>{detail}</span>}
    </div>
    <div className={c.bar} aria-hidden="true"><span style={{width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: seriesColor(0, dark)}} /></div>
  </>;
}

type Series = {label: string; values: number[]; dashed?: boolean; step?: boolean};
export function LinePlot({series, xValues, xLabel, yLabel, yMax, marker}: {
  series: Series[]; xValues: number[]; xLabel: string; yLabel: string; yMax: number; marker?: number;
}) {
  const dark = useDarkViz();
  const w = 660, h = 260, left = 52, right = 18, top = 30, bottom = 44;
  const xMin = xValues[0], xMax = xValues[xValues.length - 1];
  const x = (v: number) => left + (v - xMin) / Math.max(1, xMax - xMin) * (w - left - right);
  const y = (v: number) => h - bottom - v / Math.max(0.001, yMax) * (h - top - bottom);
  const ticks = [...new Set([0, 1, 2, 3, 4].map(i => Math.round(i * (xValues.length - 1) / 4)))];
  return <div className={c.chart}>
    <svg className={s.svg} viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${yLabel} by ${xLabel}`}>
      <text className={s.axisLabel} x={left} y={15}>{yLabel}</text>
      {[0, 0.25, 0.5, 0.75, 1].map(f => <g key={f}>
        <line className={s.grid} x1={left} x2={w - right} y1={y(yMax * f)} y2={y(yMax * f)} />
        <text className={s.tick} x={left - 8} y={y(yMax * f) + 4} textAnchor="end">{Number((yMax * f).toFixed(2))}</text>
      </g>)}
      {ticks.map(i => <text key={i} className={s.tick} x={x(xValues[i])} y={h - bottom + 18} textAnchor="middle">{xValues[i]}</text>)}
      <text className={s.axisLabel} x={w / 2} y={h - 4} textAnchor="middle">{xLabel}</text>
      {marker !== undefined && <line className={s.axis} x1={x(marker)} x2={x(marker)} y1={top} y2={h - bottom} strokeDasharray="3 3" />}
      {series.map((line, index) => <path key={line.label} fill="none" stroke={seriesColor(index, dark)} strokeWidth={2.5}
        strokeDasharray={line.dashed ? '6 4' : undefined}
        d={line.values.map((v, i) => i === 0 ? `M${x(xValues[i])},${y(v)}` : line.step ? `H${x(xValues[i])}V${y(v)}` : `L${x(xValues[i])},${y(v)}`).join(' ')}>
        <title>{line.label}</title>
      </path>)}
    </svg>
    <div className={c.chips}>{series.map((line, i) => <span key={line.label} className={c.chip}>
      <span aria-hidden="true" style={{color: seriesColor(i, dark)}}>{line.dashed ? '┄ ' : '━ '}</span>{line.label}
    </span>)}</div>
  </div>;
}
