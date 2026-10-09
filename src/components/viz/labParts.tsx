import {vizStyles as s} from './VizPanel';

type SliderProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  digits?: number;
};

export function SliderControl({label, value, min, max, step, onChange, digits = 2}: SliderProps) {
  return (
    <label className={s.control}>
      {label}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span className={s.value}>{value.toFixed(digits)}</span>
    </label>
  );
}

export type Bar = {label: string; value: number; color: string};

type BarsProps = {
  bars: Bar[];
  min: number;
  max: number;
  caption: string;
  digits?: number;
};

const W = 640;
const LEFT = 190;
const RIGHT = 70;
const ROW = 36;

export function HorizontalBars({bars, min, max, caption, digits = 2}: BarsProps) {
  const height = bars.length * ROW + 34;
  const span = max - min || 1;
  const x = (v: number) => LEFT + ((Math.max(min, Math.min(max, v)) - min) / span) * (W - LEFT - RIGHT);
  const zero = x(0);
  return (
    <svg className={s.svg} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={caption}>
      {bars.map((bar, i) => {
        const y = 10 + i * ROW;
        const end = x(bar.value);
        return (
          <g key={bar.label}>
            <text className={s.dataLabel} x={LEFT - 10} y={y + 19} textAnchor="end">
              {bar.label}
            </text>
            <rect x={Math.min(zero, end)} y={y} width={Math.max(1, Math.abs(end - zero))} height={24} fill={bar.color} opacity={0.9} />
            <text className={s.dataLabel} x={Math.max(zero, end) + 6} y={y + 17}>
              {bar.value.toFixed(digits)}
            </text>
          </g>
        );
      })}
      <line x1={zero} y1={4} x2={zero} y2={height - 26} stroke="var(--text-strong)" strokeWidth={1.5} />
      <text className={s.tick} x={W / 2} y={height - 6} textAnchor="middle">
        {caption}
      </text>
    </svg>
  );
}
