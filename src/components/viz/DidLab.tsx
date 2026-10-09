import {useState} from 'react';

import {didTable} from './causalMath';
import {SliderControl} from './labParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const LEFT = 70;
const RIGHT = 150;
const TOP = 20;
const BOTTOM = 40;

export default function DidLab() {
  const dark = useDarkViz();
  const [commonChange, setCommonChange] = useState(4);
  const [groupGap, setGroupGap] = useState(6);
  const [trueEffect, setTrueEffect] = useState(3);
  const [extraTrend, setExtraTrend] = useState(0);
  const t = didTable({controlBefore: 50, commonChange, groupGap, trueEffect, extraTrend});

  const values = [t.controlBefore, t.controlAfter, t.treatedBefore, t.treatedAfter, t.treatedBefore + commonChange];
  const lo = Math.min(...values) - 2;
  const hi = Math.max(...values) + 2;
  const x = (i: number) => LEFT + i * (W - LEFT - RIGHT);
  const y = (v: number) => TOP + (1 - (v - lo) / (hi - lo)) * (H - TOP - BOTTOM);
  const blue = seriesColor(0, dark);
  const orange = seriesColor(1, dark);
  const counterfactual = t.treatedBefore + commonChange;

  const rows: (string | number)[][] = [
    ['Control, before', t.controlBefore.toFixed(2)],
    ['Control, after', t.controlAfter.toFixed(2)],
    ['Treated, before', t.treatedBefore.toFixed(2)],
    ['Treated, after', t.treatedAfter.toFixed(2)],
    ['Treated after minus control after (ignores the gap)', t.afterOnly.toFixed(2)],
    ['Treated after minus treated before (ignores the trend)', t.beforeAfterTreated.toFixed(2)],
    ['Difference-in-differences', t.did.toFixed(2)],
    ['Bias (estimate minus true effect)', (t.did - trueEffect).toFixed(2)],
  ];

  return (
    <div data-testid="did-lab">
      <VizPanel
        title="Difference-in-differences on a two-by-two table"
        hint="The defaults are the chapter's parallel-trends case: a common change of 4, a gap of 6 and an effect of 3 give a difference-in-differences of exactly 3. Add an extra trend to see the bias it creates."
        table={{columns: ['Quantity', 'Value'], rows}}
        legend={[
          {label: 'control stores', color: blue},
          {label: 'treated stores', color: orange},
          {label: 'treated if untreated (dashed)', color: orange},
        ]}
        controls={
          <>
            <SliderControl label="Change over time in both groups" value={commonChange} min={-5} max={10} step={0.5} onChange={setCommonChange} digits={1} />
            <SliderControl label="Treated minus control before" value={groupGap} min={-10} max={10} step={1} onChange={setGroupGap} digits={0} />
            <SliderControl label="True effect" value={trueEffect} min={0} max={6} step={0.5} onChange={setTrueEffect} digits={1} />
            <SliderControl label="Extra trend in treated group" value={extraTrend} min={-3} max={3} step={0.5} onChange={setExtraTrend} digits={1} />
            <span className={s.value} aria-live="polite" data-testid="did-summary">
              difference-in-differences {t.did.toFixed(2)} against true {trueEffect.toFixed(1)}
            </span>
          </>
        }>
        <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Control goes from ${t.controlBefore} to ${t.controlAfter}; treated goes from ${t.treatedBefore} to ${t.treatedAfter}; estimate ${t.did.toFixed(2)}`}>
          <line x1={x(0)} y1={y(t.controlBefore)} x2={x(1)} y2={y(t.controlAfter)} stroke={blue} strokeWidth={3} />
          <line x1={x(0)} y1={y(t.treatedBefore)} x2={x(1)} y2={y(counterfactual)} stroke={orange} strokeWidth={2} strokeDasharray="6 4" />
          <line x1={x(0)} y1={y(t.treatedBefore)} x2={x(1)} y2={y(t.treatedAfter)} stroke={orange} strokeWidth={3} />
          {[[0, t.controlBefore, blue], [1, t.controlAfter, blue], [0, t.treatedBefore, orange], [1, t.treatedAfter, orange]].map(([i, v, c], k) => (
            <circle key={k} cx={x(i as number)} cy={y(v as number)} r={5} fill={c as string} />
          ))}
          <text className={s.dataLabel} x={x(1) + 12} y={y(t.controlAfter) + 4}>control {t.controlAfter.toFixed(1)}</text>
          <text className={s.dataLabel} x={x(1) + 12} y={y(t.treatedAfter) + 4}>treated {t.treatedAfter.toFixed(1)}</text>
          <text className={s.tick} x={x(0)} y={H - 14} textAnchor="middle">before</text>
          <text className={s.tick} x={x(1)} y={H - 14} textAnchor="middle">after</text>
        </svg>
      </VizPanel>
    </div>
  );
}
