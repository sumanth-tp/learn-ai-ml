import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function TrackingAssociationLab() {
  const dark = useDarkViz();
  const [intersection, setIntersection] = useState(30);
  const [union, setUnion] = useState(70);
  const [threshold, setThreshold] = useState(0.3);
  const iou = intersection / union;
  const associated = iou >= threshold;
  const rows: [string, string][] = [
    ['Intersection area', String(intersection)], ['Union area', String(union)],
    ['IoU', iou.toFixed(3)], ['Association threshold', threshold.toFixed(2)],
    ['Pairwise decision', associated ? 'Eligible association' : 'Reject association'],
  ];

  return <VizPanel title="Should two frame boxes be linked?"
    hint="IoU is one pairwise association cue. A real multi-object tracker also resolves competing matches, missed detections, motion and identity switches."
    table={{columns: ['Measure', 'Value'], rows}}
    controls={<div className={s.controls}>
      <label className={s.control}>Intersection area: {intersection}
        <input type="range" min="0" max="70" step="1" value={intersection} aria-label="Tracking intersection area" onChange={event => setIntersection(Math.min(Number(event.target.value), union))} />
      </label>
      <label className={s.control}>Union area: {union}
        <input type="range" min="70" max="140" step="1" value={union} aria-label="Tracking union area" onChange={event => setUnion(Number(event.target.value))} />
      </label>
      <label className={s.control}>Association threshold: {threshold.toFixed(2)}
        <input type="range" min="0.1" max="0.9" step="0.05" value={threshold} aria-label="Tracking association threshold" onChange={event => setThreshold(Number(event.target.value))} />
      </label>
    </div>}>
    <div style={{display: 'grid', gap: '0.6rem'}}>
      <div>IoU {iou.toFixed(3)} versus threshold {threshold.toFixed(2)}</div>
      <div style={{height: '1.3rem', background: dark ? '#293142' : '#e7eaf0', borderRadius: '0.25rem', overflow: 'hidden', position: 'relative'}} role="img" aria-label={`IoU ${iou.toFixed(3)}, threshold ${threshold.toFixed(2)}`}>
        <div style={{height: '100%', width: `${iou * 100}%`, background: seriesColor(0, dark)}} />
        <span style={{position: 'absolute', left: `${threshold * 100}%`, top: 0, bottom: 0, borderLeft: `3px solid ${seriesColor(1, dark)}`}} />
      </div>
      <output>{associated ? 'Eligible to link this pair.' : 'Reject this pair.'}</output>
    </div>
  </VizPanel>;
}
