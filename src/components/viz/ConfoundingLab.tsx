import {useState} from 'react';

import {confoundingGap} from './causalMath';
import {HorizontalBars, SliderControl} from './labParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function ConfoundingLab() {
  const dark = useDarkViz();
  const [shareEngaged, setShareEngaged] = useState(0.5);
  const [notifyEngaged, setNotifyEngaged] = useState(0.8);
  const [notifyOther, setNotifyOther] = useState(0.2);
  const [engagementEffect, setEngagementEffect] = useState(4);
  const [trueEffect, setTrueEffect] = useState(2);
  const r = confoundingGap({shareEngaged, notifyEngaged, notifyOther, engagementEffect, trueEffect});

  const rows: (string | number)[][] = [
    ['Share of notified users who are engaged', r.engagedAmongTreated.toFixed(3)],
    ['Share of un-notified users who are engaged', r.engagedAmongUntreated.toFixed(3)],
    ['True effect of the notification', trueEffect.toFixed(2)],
    ['Naive difference in means', r.naive.toFixed(3)],
    ['Bias (naive minus true)', r.bias.toFixed(3)],
    ['Adjusted for engagement', r.adjusted.toFixed(3)],
  ];

  return (
    <div data-testid="confounding-lab">
      <VizPanel
        title="How much does confounding inflate a naive comparison?"
        hint="The defaults are the chapter's worked example: half the users are engaged, 80 per cent of them are notified against 20 per cent of the rest, engagement is worth 4 and the notification 2. The naive gap is 4.4."
        table={{columns: ['Quantity', 'Value'], rows}}
        legend={[
          {label: 'true effect', color: seriesColor(2, dark)},
          {label: 'naive comparison', color: seriesColor(1, dark)},
          {label: 'adjusted for engagement', color: seriesColor(0, dark)},
        ]}
        controls={
          <>
            <SliderControl label="Share of users who are engaged" value={shareEngaged} min={0.1} max={0.9} step={0.05} onChange={setShareEngaged} />
            <SliderControl label="Notified, among engaged" value={notifyEngaged} min={0.05} max={0.95} step={0.05} onChange={setNotifyEngaged} />
            <SliderControl label="Notified, among others" value={notifyOther} min={0.05} max={0.95} step={0.05} onChange={setNotifyOther} />
            <SliderControl label="Spend gained from engagement" value={engagementEffect} min={0} max={8} step={0.5} onChange={setEngagementEffect} digits={1} />
            <SliderControl label="True effect of notification" value={trueEffect} min={0} max={5} step={0.5} onChange={setTrueEffect} digits={1} />
            <span className={s.value} aria-live="polite" data-testid="confounding-naive">
              naive {r.naive.toFixed(3)} against true {trueEffect.toFixed(2)}
            </span>
          </>
        }>
        <HorizontalBars
          caption="average extra spend per user"
          min={0}
          max={14}
          bars={[
            {label: 'true effect', value: trueEffect, color: seriesColor(2, dark)},
            {label: 'naive comparison', value: r.naive, color: seriesColor(1, dark)},
            {label: 'adjusted for engagement', value: r.adjusted, color: seriesColor(0, dark)},
          ]}
        />
      </VizPanel>
    </div>
  );
}
