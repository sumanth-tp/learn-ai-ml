import {useState} from 'react';

import {waldEstimate} from './causalMath';
import {HorizontalBars, SliderControl} from './labParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function WaldIvLab() {
  const dark = useDarkViz();
  const [withNudge, setWithNudge] = useState(0.756);
  const [without, setWithout] = useState(0.368);
  const [trueEffect, setTrueEffect] = useState(2);
  const [direct, setDirect] = useState(0);
  const r = waldEstimate({takeUpWithNudge: withNudge, takeUpWithout: without, trueEffect, directEffect: direct});
  const wald = Number.isFinite(r.wald) ? r.wald : 0;

  const rows: (string | number)[][] = [
    ['First stage (take-up gap)', r.firstStage.toFixed(3)],
    ['Reduced form (earnings gap by nudge)', r.reducedForm.toFixed(3)],
    ['Wald estimate', Number.isFinite(r.wald) ? r.wald.toFixed(3) : 'undefined'],
    ['Bias (estimate minus true effect)', Number.isFinite(r.wald) ? r.bias.toFixed(3) : 'undefined'],
    ['Noise multiplier (1 / first stage)', Number.isFinite(r.noiseMultiplier) ? r.noiseMultiplier.toFixed(2) : 'undefined'],
  ];

  return (
    <div data-testid="wald-lab">
      <VizPanel
        title="The Wald estimator: a reduced form divided by a first stage"
        hint="The defaults are block 2: take-up of 0.756 with the nudge and 0.368 without, a true effect of 2 and no direct path from nudge to earnings. The Wald estimate is exactly 2. Open a direct path to see the bias."
        table={{columns: ['Quantity', 'Value'], rows}}
        legend={[
          {label: 'true effect', color: seriesColor(2, dark)},
          {label: 'earnings gap by nudge (reduced form)', color: seriesColor(1, dark)},
          {label: 'Wald estimate', color: seriesColor(0, dark)},
        ]}
        controls={
          <>
            <SliderControl label="Take-up with the nudge" value={withNudge} min={0} max={1} step={0.01} onChange={setWithNudge} />
            <SliderControl label="Take-up without the nudge" value={without} min={0} max={1} step={0.01} onChange={setWithout} />
            <SliderControl label="True effect of take-up" value={trueEffect} min={0} max={5} step={0.5} onChange={setTrueEffect} digits={1} />
            <SliderControl label="Direct effect of nudge on earnings" value={direct} min={0} max={1} step={0.1} onChange={setDirect} digits={1} />
            <span className={s.value} aria-live="polite" data-testid="wald-summary">
              first stage {r.firstStage.toFixed(3)}, Wald {Number.isFinite(r.wald) ? r.wald.toFixed(3) : 'undefined'}
            </span>
          </>
        }>
        <HorizontalBars
          caption="earnings per person"
          min={0}
          max={8}
          bars={[
            {label: 'true effect', value: trueEffect, color: seriesColor(2, dark)},
            {label: 'reduced form', value: r.reducedForm, color: seriesColor(1, dark)},
            {label: 'Wald estimate', value: Math.max(0, wald), color: seriesColor(0, dark)},
          ]}
        />
      </VizPanel>
    </div>
  );
}
