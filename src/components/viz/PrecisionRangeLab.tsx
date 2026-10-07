import {useState} from 'react';

import {seriesColor} from './palette';
import {FORMATS, classify, formatNumber, roundTo, smallestNormal, smallestSubnormal} from './precisionMath';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const PRESETS: {label: string; value: string}[] = [
  {label: 'a tiny gradient, 2e-8', value: '2e-8'},
  {label: 'one tenth, 0.1', value: '0.1'},
  {label: 'one third, 0.3333333', value: '0.3333333'},
  {label: 'a large activation, 300', value: '300'},
  {label: 'a larger one, 500', value: '500'},
  {label: 'a very large one, 70000', value: '70000'},
  {label: 'a small weight update, 1e-5', value: '1e-5'},
];

const W = 640;
const H = 250;
const LEFT = 96;
const RIGHT = 16;
const LOG_MIN = -46;
const LOG_MAX = 40;

export default function PrecisionRangeLab() {
  const dark = useDarkViz();
  const [text, setText] = useState('2e-8');
  const [exponent, setExponent] = useState(0);

  const parsed = Number(text);
  const valid = text.trim() !== '' && Number.isFinite(parsed) && parsed !== 0;
  const value = valid ? Math.abs(parsed) : 2e-8;
  const scale = Math.pow(2, exponent);
  const x = (lg: number) => LEFT + ((lg - LOG_MIN) / (LOG_MAX - LOG_MIN)) * (W - LEFT - RIGHT);

  const rows = FORMATS.map((f, i) => {
    const scaled = value * scale;
    const stored = roundTo(scaled, f);
    const back = stored / scale;
    const error = Number.isFinite(back) ? Math.abs(back - value) / value : Infinity;
    return {f, i, scaled, stored, back, error, status: classify(scaled, stored, f)};
  });

  const status = valid
    ? `value ${formatNumber(value)} times 2^${exponent}: ${rows
        .map((r) => `${r.f.name} ${r.status}`)
        .join(', ')}`
    : 'enter a non-zero number such as 2e-8';

  return (
    <VizPanel
      title="Range and rounding of number formats"
      hint="The defaults are block 2, step 3: a gradient of 2e-8 in fp16 becomes zero at scale 1. Raise the loss scale to 10 (a factor of 1024) and it is stored as 2.0504e-05."
      legend={[
        {label: 'subnormal range (less precise)', color: seriesColor(2, dark)},
        {label: 'normal range', color: seriesColor(0, dark)},
      ]}
      table={{
        columns: ['format', 'value x scale', 'stored', 'after dividing by scale', 'relative error', 'status'],
        rows: rows.map((r) => [
          r.f.name,
          formatNumber(r.scaled),
          formatNumber(r.stored),
          formatNumber(r.back),
          Number.isFinite(r.error) ? r.error.toExponential(2) : 'n/a',
          r.status,
        ]),
      }}
      controls={
        <>
          <label className={s.control}>
            value
            <input
              className={s.select}
              type="text"
              inputMode="decimal"
              value={text}
              aria-invalid={!valid}
              onChange={(e) => setText(e.target.value)}
            />
          </label>
          <label className={s.control}>
            preset
            <select className={s.select} value="" onChange={(e) => e.target.value && setText(e.target.value)}>
              <option value="">choose a value</option>
              {PRESETS.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            loss scale = 2 to the power
            <input type="range" min={0} max={24} step={1} value={exponent} onChange={(e) => setExponent(Number(e.target.value))} />
            <span className={s.value}>{exponent}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Ranges of five number formats on a log axis. ${status}`}>
        {rows.map((r) => {
          const y = 18 + r.i * 38;
          const lo = Math.log10(smallestSubnormal(r.f));
          const mid = Math.log10(smallestNormal(r.f));
          const hi = Math.log10(r.f.maxFinite);
          return (
            <g key={r.f.name}>
              <text className={s.dataLabel} x={LEFT - 8} y={y + 16} textAnchor="end">
                {r.f.name}
              </text>
              <rect x={x(lo)} y={y} width={Math.max(1, x(mid) - x(lo))} height={22} fill={seriesColor(2, dark)} opacity={0.85} />
              <rect x={x(mid)} y={y} width={Math.max(1, x(hi) - x(mid))} height={22} fill={seriesColor(0, dark)} opacity={0.85} />
            </g>
          );
        })}
        {valid && (
          <>
            <line x1={x(Math.log10(value))} y1={6} x2={x(Math.log10(value))} y2={H - 38} stroke="var(--text-strong)" strokeWidth={2} />
            {exponent > 0 && (
              <line
                x1={x(Math.log10(value * scale))}
                y1={6}
                x2={x(Math.log10(value * scale))}
                y2={H - 38}
                stroke="#e34948"
                strokeWidth={2}
                strokeDasharray="5 3"
              />
            )}
          </>
        )}
        {[-40, -30, -20, -10, 0, 10, 20, 30, 38].map((t) => (
          <text key={t} className={s.tick} x={x(t)} y={H - 20} textAnchor="middle">
            1e{t}
          </text>
        ))}
        <text className={s.dataLabel} x={W / 2} y={H - 4} textAnchor="middle">
          solid line: the value. dashed red line: the value after the loss scale. Axis is log scale.
        </text>
      </svg>
    </VizPanel>
  );
}
