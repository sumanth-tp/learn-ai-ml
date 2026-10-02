import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function RepresentationGapLab() {
  const dark = useDarkViz();
  const [training, setTraining] = useState(10);
  const [deployment, setDeployment] = useState(40);
  const gap = Math.abs(deployment - training);
  const rows: [string, string][] = [
    ['Training rural share', `${training}%`],
    ['Deployment rural share', `${deployment}%`],
    ['Absolute gap', `${gap} percentage points`],
  ];

  return (
    <VizPanel
      title="Representation gap"
      hint="A sample-share gap warrants investigation of coverage and model errors by group. It does not establish the cause of an accuracy change."
      table={{columns: ['Measure', 'Value'], rows}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            Training rural share: {training}%
            <input type="range" min="0" max="100" step="1" value={training} aria-label="Training rural share"
              onChange={(event) => setTraining(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Deployment rural share: {deployment}%
            <input type="range" min="0" max="100" step="1" value={deployment} aria-label="Deployment rural share"
              onChange={(event) => setDeployment(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.9rem'}}>
        {([['Training', training, 0], ['Deployment', deployment, 1]] as const).map(([label, value, colour]) => (
          <div key={label}>
            <div>{label}: {value}% rural</div>
            <div style={{height: '1.5rem', borderRadius: '0.3rem', background: dark ? '#293142' : '#e7eaf0', overflow: 'hidden'}}>
              <div style={{height: '100%', width: `${value}%`, background: seriesColor(colour, dark)}} />
            </div>
          </div>
        ))}
        <output>{gap} percentage-point gap</output>
      </div>
    </VizPanel>
  );
}
