import {useState} from 'react';

import {DIVERGING} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const BASE = -2.008;

type Row = {name: string; value: number; shap: number};
type Case = {label: string; rows: Row[]; output: number};

const CASES: Case[] = [
  {
    label: 'high risk',
    output: 1.758,
    rows: [
      {name: 'income', value: 65084, shap: -0.412},
      {name: 'debt_ratio', value: 0.512, shap: 1.358},
      {name: 'age', value: 24, shap: 0.871},
      {name: 'late_payments', value: 2, shap: 1.394},
      {name: 'utilisation', value: 0.665, shap: 0.526},
      {name: 'tenure_years', value: 2.6, shap: 0.029},
    ],
  },
  {
    label: 'borderline',
    output: -0.839,
    rows: [
      {name: 'income', value: 75658, shap: -0.588},
      {name: 'debt_ratio', value: 0.47, shap: 0.589},
      {name: 'age', value: 29, shap: 0.779},
      {name: 'late_payments', value: 1, shap: 0.309},
      {name: 'utilisation', value: 0.523, shap: 0.029},
      {name: 'tenure_years', value: 2.7, shap: 0.051},
    ],
  },
  {
    label: 'low risk',
    output: -3.56,
    rows: [
      {name: 'income', value: 33946, shap: 0.177},
      {name: 'debt_ratio', value: 0.199, shap: -0.214},
      {name: 'age', value: 66, shap: -0.633},
      {name: 'late_payments', value: 0, shap: -0.446},
      {name: 'utilisation', value: 0.476, shap: 0.001},
      {name: 'tenure_years', value: 8.8, shap: -0.438},
    ],
  },
];

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const signed = (v: number) => (v >= 0 ? '+' : '\u2212') + Math.abs(v).toFixed(3);

type Units = 'logodds' | 'probability';
type Order = 'size' | 'listed';

const W = 640;
const ROW = 34;
const LABEL_W = 190;
const PLOT_L = LABEL_W + 10;
const PLOT_R = W - 24;

export default function ShapWaterfallLab() {
  const dark = useDarkViz();
  const [caseName, setCaseName] = useState('high risk');
  const [units, setUnits] = useState<Units>('logodds');
  const [order, setOrder] = useState<Order>('size');

  const current = CASES.find((c) => c.label === caseName) ?? CASES[0];
  const rows =
    order === 'size' ? [...current.rows].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap)) : current.rows;

  const convert = (z: number) => (units === 'logodds' ? z : sigmoid(z));
  const steps: {name: string; label: string; from: number; to: number; delta: number}[] = [];
  let running = BASE;
  rows.forEach((r) => {
    const next = running + r.shap;
    steps.push({
      name: r.name,
      label: `${r.name} = ${r.value}`,
      from: convert(running),
      to: convert(next),
      delta: r.shap,
    });
    running = next;
  });
  const start = convert(BASE);
  const end = convert(current.output);

  const all = [start, end, ...steps.flatMap((st) => [st.from, st.to])];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const pad = span * 0.08;
  const x = (v: number) => PLOT_L + ((v - (lo - pad)) / (span + 2 * pad)) * (PLOT_R - PLOT_L);

  const up = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;
  const down = dark ? DIVERGING.dark.positive : DIVERGING.light.positive;
  const neutral = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;

  const height = (steps.length + 3) * ROW + 36;

  return (
    <VizPanel
      title="SHAP waterfall: from the base value to one prediction"
      hint="Each bar is one feature's SHAP value for this applicant. They add up exactly: base value plus every bar equals the model output. In probability units the order changes how big each step looks, because the sigmoid bends; the log-odds values do not depend on order."
      legend={[
        {label: 'raises predicted default risk', color: up},
        {label: 'lowers predicted default risk', color: down},
        {label: 'base value and model output', color: neutral},
      ]}
      table={{
        columns: ['feature', 'value', 'SHAP (log-odds)'],
        rows: [
          ['base value', '', BASE.toFixed(3)],
          ...current.rows.map((r) => [r.name, r.value, signed(r.shap)]),
          ['model output', '', signed(current.output)],
        ],
      }}
      controls={
        <>
          <label className={s.control}>
            applicant
            <select className={s.select} value={caseName} onChange={(e) => setCaseName(e.target.value)}>
              {CASES.map((c) => (
                <option key={c.label} value={c.label}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            units
            <select className={s.select} value={units} onChange={(e) => setUnits(e.target.value as Units)}>
              <option value="logodds">log-odds</option>
              <option value="probability">probability</option>
            </select>
          </label>
          <label className={s.control}>
            order
            <select className={s.select} value={order} onChange={(e) => setOrder(e.target.value as Order)}>
              <option value="size">by size</option>
              <option value="listed">as listed</option>
            </select>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${height}`} role="img"
           aria-label="SHAP waterfall for one applicant">
        <g>
          <text className={s.dataLabel} x={LABEL_W} y={ROW - 10} textAnchor="end">base value</text>
          <rect x={x(start) - 1.5} y={ROW - 24} width={3} height={22} fill={neutral} />
          <text className={s.tick} x={x(start) + 8} y={ROW - 8}>{start.toFixed(3)}</text>
        </g>
        {steps.map((st, i) => {
          const y = ROW + 8 + i * ROW;
          const x1 = x(Math.min(st.from, st.to));
          const x2 = x(Math.max(st.from, st.to));
          const color = st.delta >= 0 ? up : down;
          return (
            <g key={st.name}>
              <text className={s.dataLabel} x={LABEL_W} y={y + 14} textAnchor="end">{st.label}</text>
              <line x1={x(st.from)} y1={y - ROW + 22} x2={x(st.from)} y2={y} stroke="var(--border-strong)" strokeDasharray="2 3" />
              <rect x={x1} y={y} width={Math.max(2, x2 - x1)} height={20} rx={3} fill={color} />
              <text className={s.tick} x={x2 + 6} y={y + 14}>{signed(st.delta)}</text>
            </g>
          );
        })}
        {(() => {
          const y = ROW + 8 + steps.length * ROW;
          return (
            <g>
              <text className={s.dataLabel} x={LABEL_W} y={y + 14} textAnchor="end">model output</text>
              <line x1={x(end)} y1={y - ROW + 22} x2={x(end)} y2={y} stroke="var(--border-strong)" strokeDasharray="2 3" />
              <rect x={x(end) - 1.5} y={y} width={3} height={20} fill={neutral} />
              <text className={s.tick} x={x(end) + 8} y={y + 14}>
                {units === 'logodds' ? `${signed(current.output)} log-odds` : `${end.toFixed(3)} probability`}
              </text>
              {units === 'logodds' && (
                <text className={s.tick} x={PLOT_L} y={y + 40}>
                  = {sigmoid(current.output).toFixed(3)} probability
                </text>
              )}
            </g>
          );
        })()}
      </svg>
    </VizPanel>
  );
}
