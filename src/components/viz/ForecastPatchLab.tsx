import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const context = [10, 12, 11, 13, 10, 12];

export default function ForecastPatchLab() {
  const dark = useDarkViz();
  const [width, setWidth] = useState(2);
  const [horizon, setHorizon] = useState(2);
  const contextPatches = Array.from({length: Math.ceil(context.length / width)}, (_, index) => context.slice(index * width, (index + 1) * width));
  const futurePatches = Math.ceil(horizon / width);
  const promotions = Array.from({length: horizon}, (_, index) => index % 2);
  const rows: [string, string, string][] = contextPatches.map((values, index) => [`Context patch ${index + 1}`, values.join(', '), 'Observed']);
  rows.push(['Future target', `${horizon} hidden values`, 'Unavailable']);
  rows.push(['Future promotion flags', promotions.join(', '), 'Known by schedule']);

  return <VizPanel title="Patch observed history, mask the horizon"
    hint="This illustrates an input contract, not a run of TimesFM-3 or Chronos-2. Future targets remain hidden; only genuinely known future covariates may be supplied."
    table={{columns: ['Part', 'Values', 'Visibility'], rows}}
    controls={<div className={s.controls}>
      <label className={s.control}>Patch width: {width}
        <input type="range" min="1" max="3" step="1" value={width} aria-label="Forecast patch width" onChange={event => setWidth(Number(event.target.value))} />
      </label>
      <label className={s.control}>Forecast horizon: {horizon}
        <input type="range" min="1" max="4" step="1" value={horizon} aria-label="Forecast horizon" onChange={event => setHorizon(Number(event.target.value))} />
      </label>
    </div>}>
    <div style={{display: 'grid', gap: '0.7rem'}}>
      <div style={{display: 'flex', gap: '0.3rem', flexWrap: 'wrap'}}>
        {contextPatches.map((values, index) => <span key={index} style={{padding: '0.5rem', color: '#fff', background: seriesColor(0, dark), borderRadius: '0.25rem'}}>Past {index + 1}: [{values.join(', ')}]</span>)}
        {Array.from({length: futurePatches}, (_, index) => <span key={index} style={{padding: '0.5rem', color: '#fff', background: seriesColor(1, dark), borderRadius: '0.25rem'}}>Future {index + 1}: masked</span>)}
      </div>
      <output>{contextPatches.length} observed patches; {horizon} future target values hidden. Known future promotion flags [{promotions.join(', ')}] are a separate input.</output>
    </div>
  </VizPanel>;
}
