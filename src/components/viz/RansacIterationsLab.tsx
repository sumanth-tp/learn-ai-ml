import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function RansacIterationsLab() {
  const dark = useDarkViz();
  const [inliers, setInliers] = useState(0.5);
  const [sample, setSample] = useState(2);
  const [target, setTarget] = useState(0.99);
  const oneSampleSuccess = inliers ** sample;
  const continuous = Math.log(1 - target) / Math.log(1 - oneSampleSuccess);
  const iterations = Math.ceil(continuous);
  const achieved = 1 - (1 - oneSampleSuccess) ** iterations;
  const rows: [string, string][] = [
    ['Inlier fraction', inliers.toFixed(2)],
    ['Minimal sample size', String(sample)],
    ['Target success', `${(target * 100).toFixed(1)}%`],
    ['One-sample success', oneSampleSuccess.toFixed(4)],
    ['Continuous bound', continuous.toFixed(3)],
    ['Whole iterations', String(iterations)],
    ['Achieved success', `${(achieved * 100).toFixed(3)}%`],
  ];

  return (
    <VizPanel
      title="How many RANSAC trials?"
      hint="The formula assumes independent uniform samples and a known inlier fraction. Geometry may have degenerate samples or an incorrect inlier threshold."
      table={{columns: ['Measure', 'Value'], rows}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            Inlier fraction: {inliers.toFixed(2)}
            <input type="range" min="0.1" max="0.9" step="0.05" value={inliers} aria-label="RANSAC inlier fraction"
              onChange={event => setInliers(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Minimal sample size: {sample}
            <input type="range" min="2" max="5" step="1" value={sample} aria-label="RANSAC minimal sample size"
              onChange={event => setSample(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Target success: {(target * 100).toFixed(1)}%
            <input type="range" min="0.8" max="0.999" step="0.001" value={target} aria-label="RANSAC target success probability"
              onChange={event => setTarget(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.8rem'}}>
        <strong>{iterations} whole trials reach {(achieved * 100).toFixed(3)}% success under the model</strong>
        <div style={{height: '1.4rem', borderRadius: '0.25rem', background: dark ? '#293142' : '#e7eaf0', overflow: 'hidden'}} role="img" aria-label={`Achieved probability ${(achieved * 100).toFixed(3)} percent`}>
          <div style={{height: '100%', width: `${achieved * 100}%`, background: seriesColor(0, dark)}} />
        </div>
        <output>Continuous result {continuous.toFixed(3)} must be rounded up.</output>
      </div>
    </VizPanel>
  );
}
