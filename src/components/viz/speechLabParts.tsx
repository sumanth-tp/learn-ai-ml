import {vizStyles as s} from './VizPanel';

type SliderProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  digits?: number;
  unit?: string;
};

export function SpeechSlider({label, value, min, max, step, onChange, digits = 0, unit = ''}: SliderProps) {
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
      <span className={s.value}>
        {value.toFixed(digits)}
        {unit}
      </span>
    </label>
  );
}

type SelectProps = {
  label: string;
  value: number;
  options: number[];
  onChange: (value: number) => void;
  format?: (value: number) => string;
};

export function SpeechSelect({label, value, options, onChange, format}: SelectProps) {
  return (
    <label className={s.control}>
      {label}
      <select className={s.select} value={value} aria-label={label} onChange={(event) => onChange(Number(event.target.value))}>
        {options.map((option) => (
          <option key={option} value={option}>
            {format ? format(option) : option}
          </option>
        ))}
      </select>
    </label>
  );
}
