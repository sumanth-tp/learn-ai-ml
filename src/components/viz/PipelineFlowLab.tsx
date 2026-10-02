import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function PipelineFlowLab() {
  const dark = useDarkViz();
  const [arrival, setArrival] = useState(100);
  const [tenths, setTenths] = useState(2);
  const seconds = tenths / 10;
  const inflight = arrival * seconds;

  return (
    <VizPanel
      title="Little's Law: average work in flight"
      hint="The relation uses stable long-run averages. It does not tell you whether an overloaded queue will keep growing."
      table={{columns: ['quantity', 'value', 'unit'], rows: [
        ['arrival rate', String(arrival), 'events per second'],
        ['mean time', seconds.toFixed(1), 'seconds'],
        ['mean in flight', inflight.toFixed(1), 'events'],
      ]}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            arrival rate: {arrival} events/s
            <input type="range" min="20" max="200" value={arrival} aria-label="Arrival rate"
              onChange={(event) => setArrival(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            mean time: {seconds.toFixed(1)} s
            <input type="range" min="1" max="10" value={tenths} aria-label="Mean time in pipeline"
              onChange={(event) => setTenths(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.7rem'}}>
        <strong>{arrival} events/s × {seconds.toFixed(1)} s = {inflight.toFixed(1)} events in flight</strong>
        <div style={{height: '1.5rem', borderRadius: '0.25rem', background: 'var(--ifm-color-emphasis-200)', overflow: 'hidden'}}
          role="img" aria-label={`${inflight.toFixed(1)} average events in flight`}>
          <div style={{height: '100%', width: `${Math.min(100, inflight / 200 * 100)}%`, background: seriesColor(0, dark)}} />
        </div>
        <output>Average work in flight: {inflight.toFixed(1)}</output>
      </div>
    </VizPanel>
  );
}
