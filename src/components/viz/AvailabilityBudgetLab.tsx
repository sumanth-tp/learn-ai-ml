import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function AvailabilityBudgetLab() {
  const dark = useDarkViz();
  const [basisPoints, setBasisPoints] = useState(9990);
  const [days, setDays] = useState(365);
  const target = basisPoints / 100;
  const hours = days * 24;
  const unavailable = hours * (1 - basisPoints / 10000);
  const minutes = unavailable * 60;

  return (
    <VizPanel
      title="What an availability target permits"
      hint="This is an idealised complete-outage equivalent. A real SLA defines eligible requests, windows, exclusions and remedies."
      table={{columns: ['quantity', 'value'], rows: [
        ['target availability', `${target.toFixed(2)}%`],
        ['measurement period', `${days} days`],
        ['period hours', hours.toFixed(0)],
        ['unavailable hours', unavailable.toFixed(3)],
        ['unavailable minutes', minutes.toFixed(1)],
      ]}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            target availability: {target.toFixed(2)}%
            <input type="range" min="9900" max="9999" value={basisPoints} aria-label="Availability target"
              onChange={(event) => setBasisPoints(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            period: {days} days
            <input type="range" min="7" max="365" value={days} aria-label="Measurement period days"
              onChange={(event) => setDays(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.7rem'}}>
        <strong>{unavailable.toFixed(3)} hours of equivalent complete downtime</strong>
        <div style={{height: '1.5rem', borderRadius: '0.25rem', background: seriesColor(2, dark), overflow: 'hidden'}}
          role="img" aria-label={`${target.toFixed(2)} per cent available time`}>
          <div style={{height: '100%', width: `${target}%`, background: seriesColor(0, dark)}} />
        </div>
        <output>{minutes.toFixed(1)} minutes across {days} days</output>
      </div>
    </VizPanel>
  );
}
