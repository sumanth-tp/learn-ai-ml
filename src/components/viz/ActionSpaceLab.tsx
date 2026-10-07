import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Box = [number, number, number, number];
const WIDGETS: Record<string, Box> = {
  name: [40, 120, 300, 32],
  amount: [40, 180, 300, 32],
  category: [40, 240, 200, 32],
  submit: [40, 320, 120, 40],
  close: [360, 16, 16, 16],
};
const TASK = ['name', 'amount', 'category', 'submit'];
const SIGMAS = [0, 4, 8, 12, 16, 24];
const SIMULATED: Record<number, [number, number]> = {
  0: [1.0, 0.239],
  4: [1.0, 0.239],
  8: [0.857, 0.207],
  12: [0.497, 0.12],
  16: [0.252, 0.063],
  24: [0.072, 0.017],
};
const SHIFT = 48;
const W = 420;
const H = 420;

function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1 - poly * Math.exp(-a * a));
}

const pHit = (widget: string, sigma: number) => {
  if (sigma === 0) return 1;
  const [, , w, h] = WIDGETS[widget];
  const side = (size: number) => erf(size / 2 / (sigma * Math.SQRT2));
  return side(w) * side(h);
};

function samples(count: number, seed: number): [number, number][] {
  let state = seed;
  const next = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return (state + 1) / 4294967297;
  };
  const out: [number, number][] = [];
  for (let i = 0; i < count; i++) {
    const r = Math.sqrt(-2 * Math.log(next()));
    const t = 2 * Math.PI * next();
    out.push([r * Math.cos(t), r * Math.sin(t)]);
  }
  return out;
}

export default function ActionSpaceLab() {
  const dark = useDarkViz();
  const [sigma, setSigma] = useState(12);
  const [shifted, setShifted] = useState(false);
  const [refs, setRefs] = useState(false);
  const [target, setTarget] = useState('submit');
  const noise = useMemo(() => samples(150, 42), []);

  const box = WIDGETS[target];
  const dy = shifted ? SHIFT : 0;
  const aimX = box[0] + box[2] / 2;
  const aimY = box[1] + box[3] / 2;
  const clicks = noise.map(([a, b]) => [aimX + a * sigma, aimY + b * sigma] as [number, number]);
  const hitsNow = (x: number, y: number) => x >= box[0] && x <= box[0] + box[2] && y >= box[1] + dy && y <= box[1] + box[3] + dy;
  const landed = refs ? clicks.length : clicks.filter(([x, y]) => hitsNow(x, y)).length;

  const analyticTask = TASK.reduce((acc, w) => acc * pHit(w, sigma), 1);
  const simulated = SIMULATED[sigma];
  const taskValue = refs ? 1 : shifted ? simulated[1] : simulated[0];
  const good = seriesColor(2, dark);
  const bad = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;
  const ink = dark ? '#e6e8ec' : '#1f2328';
  const faint = dark ? '#848c99' : '#9aa0a6';

  const rows = Object.keys(WIDGETS).map((name) => [name, `${WIDGETS[name][2]} x ${WIDGETS[name][3]}`, pHit(name, sigma).toFixed(3)]);

  return (
    <VizPanel
      title="A toy screen: where does the click land?"
      hint="The solid boxes are the widgets now. Dots are 150 clicks aimed at the centre of the target with the chosen grounding error (one standard deviation in pixels), green if they land inside. The defaults, error 12 px and no layout shift, reproduce the printed task success of 0.494 (analytic) and 0.497 (20,000 simulated runs). Turn on the layout shift, or switch to element references, and watch the number move."
      legend={[
        {label: 'click lands on the target', color: good},
        {label: 'click misses', color: bad},
      ]}
      table={{columns: ['widget', 'size (px)', 'P(click lands)'], rows}}
      controls={
        <>
          <label className={s.control}>
            grounding error (px)
            <select className={s.select} value={sigma} disabled={refs} onChange={(e) => setSigma(Number(e.target.value))}>
              {SIGMAS.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            action space
            <select className={s.select} value={refs ? 'refs' : 'pixels'} onChange={(e) => setRefs(e.target.value === 'refs')}>
              <option value="pixels">pixel coordinates</option>
              <option value="refs">element references</option>
            </select>
          </label>
          <label className={s.control}>
            layout
            <select className={s.select} value={shifted ? 'shift' : 'still'} onChange={(e) => setShifted(e.target.value === 'shift')}>
              <option value="still">stays still</option>
              <option value="shift">banner pushes it down 48 px (30% of steps)</option>
            </select>
          </label>
          <label className={s.control}>
            target
            <select className={s.select} value={target} onChange={(e) => setTarget(e.target.value)}>
              {Object.keys(WIDGETS).map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Toy form with five widgets and 150 clicks aimed at ${target}. ${landed} of 150 land on it.`}>
        {Object.entries(WIDGETS).map(([name, [x, y, w, h]]) => (
          <g key={name}>
            <rect x={x} y={y + (name === target ? dy : 0)} width={w} height={h} fill="none" stroke={name === target ? ink : faint} strokeWidth={name === target ? 2 : 1} />
            <text className={s.tick} x={x + 4} y={y + (name === target ? dy : 0) + h / 2 + 3}>
              {name}
            </text>
          </g>
        ))}
        {shifted && <rect x={box[0]} y={box[1]} width={box[2]} height={box[3]} fill="none" stroke={faint} strokeDasharray="4 3" />}
        {clicks.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={2.4} fill={refs || hitsNow(x, y) ? good : bad} opacity={refs ? 0.25 : 0.85} />
        ))}
        <circle cx={aimX} cy={aimY} r={3.5} fill="none" stroke={ink} strokeWidth={1.5} />
      </svg>
      <p className={s.hint} style={{padding: '0.4rem 0 0'}} aria-live="polite">
        {refs
          ? 'Element references name the widget, so there is no pixel error: every click lands (the dots are drawn faint). References have their own failure modes, such as a stale reference after the page re-renders.'
          : `${landed} of 150 clicks land on ${target}. P(click lands) = ${pHit(target, sigma).toFixed(3)}.`}{' '}
        Four-step task (name, amount, category, submit): analytic {analyticTask.toFixed(3)} with no shift, simulated {taskValue.toFixed(3)} for this setting.
      </p>
    </VizPanel>
  );
}
