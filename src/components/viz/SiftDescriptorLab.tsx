import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function SiftDescriptorLab() {
  const dark = useDarkViz();
  const [side, setSide] = useState(4);
  const [bins, setBins] = useState(8);
  const cells = side * side;
  const dimensions = cells * bins;
  const rows: [string, string][] = [
    ['Cells per side', String(side)],
    ['Total cells', String(cells)],
    ['Orientation bins per cell', String(bins)],
    ['Descriptor dimensions', String(dimensions)],
  ];

  return (
    <VizPanel
      title="Build a local orientation descriptor"
      hint="This grid explains descriptor length. It does not extract SIFT keypoints or reproduce their weighted orientation histograms."
      table={{columns: ['Measure', 'Value'], rows}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            Cells per side: {side}
            <input type="range" min="2" max="6" step="1" value={side} aria-label="SIFT cells per side"
              onChange={event => setSide(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Orientation bins: {bins}
            <input type="range" min="4" max="12" step="1" value={bins} aria-label="SIFT orientation bins"
              onChange={event => setBins(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', justifyItems: 'center', gap: '0.65rem'}}>
        <div style={{display: 'grid', gridTemplateColumns: `repeat(${side}, 1fr)`, gap: '0.25rem', width: '12rem'}} role="img" aria-label={`${side} by ${side} spatial cell grid`}>
          {Array.from({length: cells}, (_, index) => (
            <div key={index} style={{aspectRatio: '1', borderRadius: '0.2rem', background: seriesColor(index % 3, dark), opacity: 0.5 + (index % 3) * 0.2}} />
          ))}
        </div>
        <output>{side} × {side} cells × {bins} bins = {dimensions} dimensions</output>
      </div>
    </VizPanel>
  );
}
