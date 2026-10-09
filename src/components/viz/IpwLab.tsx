import {useState} from 'react';

import {ipwEstimate} from './causalMath';
import {HorizontalBars, SliderControl} from './labParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function IpwLab() {
  const dark = useDarkViz();
  const [shareEngaged, setShareEngaged] = useState(0.5);
  const [notifyEngaged, setNotifyEngaged] = useState(0.8);
  const [notifyOther, setNotifyOther] = useState(0.2);
  const [engagementEffect, setEngagementEffect] = useState(4);
  const [trim, setTrim] = useState(0);
  const trueEffect = 2;
  const r = ipwEstimate({shareEngaged, notifyEngaged, notifyOther, engagementEffect, trueEffect, trim});

  const rows: (string | number)[][] = [
    ['Notified users', r.countTreated.toFixed(0)],
    ['Not notified users', r.countControl.toFixed(0)],
    ['Naive difference', r.naive.toFixed(3)],
    ['Weighted mean, notified', r.ipwTreated.toFixed(3)],
    ['Weighted mean, not notified', r.ipwControl.toFixed(3)],
    ['Weighted effect', r.ipwEffect.toFixed(3)],
    ['Effective sample size, notified', r.essTreated.toFixed(1)],
    ['Effective sample size, not notified', r.essControl.toFixed(1)],
    ['Largest weight', r.maxWeight.toFixed(2)],
  ];

  return (
    <div data-testid="ipw-lab">
      <VizPanel
        title="Weighting a population of 1,000 users until it looks randomised"
        hint="The defaults are the chapter's worked example: weights 1.25 and 5, a weighted effect of exactly 2.0 and an effective sample size of 320 out of 500 notified users."
        table={{columns: ['Quantity', 'Value'], rows}}
        legend={[
          {label: 'true effect', color: seriesColor(2, dark)},
          {label: 'naive difference', color: seriesColor(1, dark)},
          {label: 'weighted effect', color: seriesColor(0, dark)},
        ]}
        controls={
          <>
            <SliderControl label="Share of users who are engaged" value={shareEngaged} min={0.1} max={0.9} step={0.05} onChange={setShareEngaged} />
            <SliderControl label="Notified, among engaged" value={notifyEngaged} min={0.05} max={0.95} step={0.05} onChange={setNotifyEngaged} />
            <SliderControl label="Notified, among others" value={notifyOther} min={0.05} max={0.95} step={0.05} onChange={setNotifyOther} />
            <SliderControl label="Spend gained from engagement" value={engagementEffect} min={0} max={8} step={0.5} onChange={setEngagementEffect} digits={1} />
            <SliderControl label="Trim propensities to" value={trim} min={0} max={0.45} step={0.05} onChange={setTrim} />
            <span className={s.value} aria-live="polite" data-testid="ipw-summary">
              weighted effect {r.ipwEffect.toFixed(3)}, effective sample {r.essTreated.toFixed(0)} notified
            </span>
          </>
        }>
        <HorizontalBars
          caption="average extra spend per user (true effect is 2)"
          min={0}
          max={14}
          bars={[
            {label: 'true effect', value: trueEffect, color: seriesColor(2, dark)},
            {label: 'naive difference', value: r.naive, color: seriesColor(1, dark)},
            {label: 'weighted effect', value: r.ipwEffect, color: seriesColor(0, dark)},
          ]}
        />
      </VizPanel>
    </div>
  );
}
