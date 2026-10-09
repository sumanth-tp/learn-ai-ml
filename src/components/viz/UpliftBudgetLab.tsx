import {useState} from 'react';

import {UPLIFT_SEGMENTS, upliftProfit, upliftSummary} from './causalMath';
import {SliderControl} from './labParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 280;
const LEFT = 60;
const RIGHT = 20;
const TOP = 16;
const BOTTOM = 40;

export default function UpliftBudgetLab() {
  const dark = useDarkViz();
  const [cost, setCost] = useState(0.5);
  const [share, setShare] = useState(0.2);
  const r = upliftSummary(cost, share);

  const steps = Array.from({length: 101}, (_, i) => i / 100);
  const model = steps.map((f) => upliftProfit(cost, f));
  const lottery = steps.map((f) => f * (r.averageEffect - cost));
  const all = [...model, ...lottery, 0];
  const lo = Math.min(...all) - 0.02;
  const hi = Math.max(...all) + 0.02;
  const x = (f: number) => LEFT + f * (W - LEFT - RIGHT);
  const y = (v: number) => TOP + (1 - (v - lo) / (hi - lo)) * (H - TOP - BOTTOM);
  const path = (values: number[]) => values.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(steps[i]).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const blue = seriesColor(0, dark);
  const orange = seriesColor(1, dark);

  const rows: (string | number)[][] = [
    ...UPLIFT_SEGMENTS.map((seg) => [`${seg.name} (${(seg.share * 100).toFixed(1)}% of customers)`, `effect ${seg.effect.toFixed(3)}`]),
    ['Profit per customer, model-ranked, at this budget', r.byModel.toFixed(3)],
    ['Profit per customer, random targeting, at this budget', r.byLottery.toFixed(3)],
    ['Profit per customer, everyone treated', r.everyone.toFixed(3)],
    ['Best share to treat', r.bestShare.toFixed(3)],
    ['Best profit per customer', r.bestProfit.toFixed(3)],
  ];

  return (
    <div data-testid="uplift-lab">
      <VizPanel
        title="How many customers should get the coupon?"
        hint="The defaults are block 1: coupon cost 0.5 and the top 20 per cent by uplift. A perfect ranking earns 0.352 per customer, random targeting loses 0.021, and treating everyone loses 0.104."
        table={{columns: ['Quantity', 'Value'], rows}}
        legend={[
          {label: 'ranked by true uplift', color: blue},
          {label: 'random targeting', color: orange},
        ]}
        controls={
          <>
            <SliderControl label="Coupon cost" value={cost} min={0} max={3} step={0.1} onChange={setCost} digits={1} />
            <SliderControl label="Share of customers who get a coupon" value={share} min={0} max={1} step={0.01} onChange={setShare} />
            <span className={s.value} aria-live="polite" data-testid="uplift-summary">
              model-ranked {r.byModel.toFixed(3)}, random {r.byLottery.toFixed(3)}, best share {r.bestShare.toFixed(3)}
            </span>
          </>
        }>
        <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Profit per customer against the share treated. Model-ranked ${r.byModel.toFixed(3)}, random ${r.byLottery.toFixed(3)} at the chosen share.`}>
          <line x1={LEFT} y1={y(0)} x2={W - RIGHT} y2={y(0)} stroke="var(--text-strong)" strokeWidth={1} strokeDasharray="4 3" />
          <path d={path(lottery)} fill="none" stroke={orange} strokeWidth={2.5} />
          <path d={path(model)} fill="none" stroke={blue} strokeWidth={3} />
          <line x1={x(share)} y1={TOP} x2={x(share)} y2={H - BOTTOM} stroke="var(--text-strong)" strokeWidth={1.5} />
          <circle cx={x(share)} cy={y(r.byModel)} r={5} fill={blue} />
          <circle cx={x(r.bestShare)} cy={y(r.bestProfit)} r={5} fill="none" stroke={blue} strokeWidth={2} />
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <text key={f} className={s.tick} x={x(f)} y={H - 20} textAnchor="middle">{`${f * 100}%`}</text>
          ))}
          <text className={s.axisLabel} x={W / 2} y={H - 4} textAnchor="middle">share of customers who get a coupon (solid line: chosen share; ring: best share)</text>
        </svg>
      </VizPanel>
    </div>
  );
}
