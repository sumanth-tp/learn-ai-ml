import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function BoxIouNmsLab() {
  const dark = useDarkViz();
  const [start, setStart] = useState(5);
  const [threshold, setThreshold] = useState(0.5);
  const intersection = Math.max(0, 10 - start) * 10;
  const union = 200 - intersection;
  const iou = intersection / union;
  const keepBoth = iou <= threshold;
  const rows: [string, string][] = [
    ['Box A', '[0, 0, 10, 10]'], ['Box B', `[${start}, 0, ${start + 10}, 10]`],
    ['Intersection area', String(intersection)], ['Union area', String(union)],
    ['IoU', iou.toFixed(3)], ['NMS threshold', threshold.toFixed(2)],
    ['Decision', keepBoth ? 'Keep both' : 'Keep A; suppress B'],
  ];

  return <VizPanel title="Box overlap and suppression"
    hint="A and B represent duplicate candidates for the same class. A has the higher score; NMS suppresses B only when IoU exceeds the selected threshold."
    table={{columns: ['Measure', 'Value'], rows}}
    controls={<div className={s.controls}>
      <label className={s.control}>Box B horizontal start: {start}
        <input type="range" min="0" max="15" step="1" value={start} aria-label="Box B horizontal start" onChange={event => setStart(Number(event.target.value))} />
      </label>
      <label className={s.control}>NMS threshold: {threshold.toFixed(2)}
        <input type="range" min="0.1" max="0.9" step="0.05" value={threshold} aria-label="NMS threshold" onChange={event => setThreshold(Number(event.target.value))} />
      </label>
    </div>}>
    <div style={{display: 'grid', gap: '0.6rem'}}>
      <svg width="360" height="180" viewBox="0 0 290 145" role="img" aria-label={`Box A from zero to ten and box B from ${start} to ${start + 10}; intersection area ${intersection}`} style={{maxWidth: '100%'}}>
        <rect x="20" y="20" width="250" height="100" fill={dark ? '#202838' : '#f2f4f8'} />
        <rect x="30" y="30" width="100" height="80" fill="none" stroke={seriesColor(0, dark)} strokeWidth="4" />
        <rect x={30 + start * 10} y="30" width="100" height="80" fill="none" stroke={seriesColor(1, dark)} strokeWidth="4" />
        <text x="35" y="48" fill={dark ? '#fff' : '#111'}>A</text>
        <text x={35 + start * 10} y="100" fill={dark ? '#fff' : '#111'}>B</text>
      </svg>
      <output>Intersection {intersection}; union {union}; IoU {iou.toFixed(3)}. {keepBoth ? 'Keep both boxes.' : 'Suppress B.'}</output>
    </div>
  </VizPanel>;
}
