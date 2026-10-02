import {useState} from 'react';

import {confusionMetrics} from './evalMath';
import {sequentialColor, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Counts = {tp: number; fp: number; fn: number; tn: number};

const PRESETS: Record<string, {label: string; counts: Counts}> = {
  lecture: {label: 'Lecture example (40, 10, 5, 45)', counts: {tp: 40, fp: 10, fn: 5, tn: 45}},
  bank: {label: 'Question bank Q50 (40, 10, 20, 30)', counts: {tp: 40, fp: 10, fn: 20, tn: 30}},
  legit: {label: 'Always predicts legit (0, 0, 10, 990)', counts: {tp: 0, fp: 0, fn: 10, tn: 990}},
};

const FIELDS: {key: keyof Counts; label: string; max: number}[] = [
  {key: 'tp', label: 'TP', max: 1000},
  {key: 'fp', label: 'FP', max: 1000},
  {key: 'fn', label: 'FN', max: 1000},
  {key: 'tn', label: 'TN', max: 1000},
];

const FORMULAS: Record<string, string> = {
  accuracy: '(TP + TN) / all',
  precision: 'TP / (TP + FP)',
  recall: 'TP / (TP + FN)',
  f1: '2PR / (P + R)',
  specificity: 'TN / (TN + FP)',
};

const fmt = (v: number | null) => (v === null ? 'n/a' : v.toFixed(3));

const W = 640;
const H = 250;

export default function EvalConfusionMatrixLab() {
  const dark = useDarkViz();
  const [preset, setPreset] = useState('lecture');
  const [counts, setCounts] = useState<Counts>(PRESETS.lecture.counts);

  const metrics = confusionMetrics(counts);
  const total = counts.tp + counts.fp + counts.fn + counts.tn;
  const names = ['accuracy', 'precision', 'recall', 'f1', 'specificity'] as const;

  const cells = [
    {key: 'tn', label: 'TN', value: counts.tn, x: 0, y: 0},
    {key: 'fp', label: 'FP', value: counts.fp, x: 1, y: 0},
    {key: 'fn', label: 'FN', value: counts.fn, x: 0, y: 1},
    {key: 'tp', label: 'TP', value: counts.tp, x: 1, y: 1},
  ];
  const cell = 74;
  const gx = 74;
  const gy = 56;
  const barX = 360;
  const barW = 220;

  return (
    <VizPanel
      title="Confusion-matrix calculator"
      hint="Set the four counts and every metric recomputes. Try the third preset: 99% accuracy, zero recall."
      legend={[
        {label: 'cell share of all cases', color: sequentialColor(0.7, dark)},
        {label: 'metric value 0 to 1', color: seriesColor(0, dark)},
      ]}
      table={{
        columns: ['metric', 'formula', 'value'],
        rows: names.map((n) => [n, FORMULAS[n], fmt(metrics[n])]),
      }}
      controls={
        <>
          <label className={s.control}>
            preset
            <select
              className={s.select}
              value={preset}
              onChange={(e) => {
                setPreset(e.target.value);
                if (PRESETS[e.target.value]) setCounts(PRESETS[e.target.value].counts);
              }}>
              {Object.entries(PRESETS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.label}
                </option>
              ))}
              <option value="custom" disabled>
                Custom
              </option>
            </select>
          </label>
          {FIELDS.map((f) => (
            <label key={f.key} className={s.control}>
              {f.label}
              <input
                type="range"
                min={0}
                max={f.max}
                step={1}
                value={counts[f.key]}
                onChange={(e) => {
                  setPreset('custom');
                  setCounts({...counts, [f.key]: Number(e.target.value)});
                }}
              />
              <span className={s.value}>{counts[f.key]}</span>
            </label>
          ))}
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Confusion matrix and the metrics computed from it">
        <text className={s.axisLabel} x={gx + cell} y={18} textAnchor="middle">predicted</text>
        <text className={s.tick} x={gx + cell / 2} y={gy - 8} textAnchor="middle">negative</text>
        <text className={s.tick} x={gx + cell * 1.5} y={gy - 8} textAnchor="middle">positive</text>
        <text className={s.axisLabel} x={14} y={gy + cell} textAnchor="middle"
              transform={`rotate(-90 14 ${gy + cell})`}>actual</text>
        <text className={s.tick} x={gx - 6} y={gy + cell / 2 + 3} textAnchor="end">negative</text>
        <text className={s.tick} x={gx - 6} y={gy + cell * 1.5 + 3} textAnchor="end">positive</text>
        {cells.map((c) => (
          <g key={c.key}>
            <rect
              x={gx + c.x * cell}
              y={gy + c.y * cell}
              width={cell - 3}
              height={cell - 3}
              rx={6}
              fill={sequentialColor(total ? 0.15 + 0.85 * (c.value / total) : 0, dark)}
              stroke="var(--border-strong)"
            />
            <text className={s.dataLabel} x={gx + c.x * cell + cell / 2 - 1} y={gy + c.y * cell + 30}
                  textAnchor="middle">{c.label}</text>
            <text className={s.dataLabel} x={gx + c.x * cell + cell / 2 - 1} y={gy + c.y * cell + 50}
                  textAnchor="middle" style={{fontSize: 14}}>{c.value}</text>
          </g>
        ))}
        <text className={s.tick} x={gx + cell} y={gy + 2 * cell + 20} textAnchor="middle">{total} cases</text>
        {names.map((n, i) => {
          const v = metrics[n];
          const y = 40 + i * 40;
          return (
            <g key={n}>
              <text className={s.dataLabel} x={barX - 8} y={y + 12} textAnchor="end">{n}</text>
              <rect x={barX} y={y} width={barW} height={16} rx={8} fill="var(--border-subtle)" />
              {v !== null && (
                <rect x={barX} y={y} width={Math.max(2, barW * v)} height={16} rx={8} fill={seriesColor(0, dark)} />
              )}
              <text className={s.tick} x={barX + barW + 8} y={y + 12}>{fmt(v)}</text>
            </g>
          );
        })}
      </svg>
    </VizPanel>
  );
}
