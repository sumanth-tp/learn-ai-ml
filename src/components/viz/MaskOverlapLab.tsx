import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function MaskOverlapLab() {
  const dark = useDarkViz();
  const [predicted, setPredicted] = useState(100);
  const [groundTruth, setGroundTruth] = useState(100);
  const [requestedOverlap, setOverlap] = useState(50);
  const overlap = Math.min(requestedOverlap, predicted, groundTruth);
  const union = predicted + groundTruth - overlap;
  const iou = union ? overlap / union : null;
  const dice = predicted + groundTruth ? 2 * overlap / (predicted + groundTruth) : null;
  const format = (value: number | null) => value === null ? 'undefined' : value.toFixed(3);
  const rows: [string, string][] = [
    ['Predicted pixels', String(predicted)], ['Ground-truth pixels', String(groundTruth)],
    ['Intersection pixels', String(overlap)], ['Union pixels', String(union)],
    ['IoU', format(iou)], ['Dice', format(dice)],
  ];

  return <VizPanel title="Measure mask overlap"
    hint="These counts do not specify mask shapes. The bars show counts, not a literal geometric overlay. For two empty masks, this lab marks IoU and Dice undefined."
    table={{columns: ['Measure', 'Value'], rows}}
    controls={<div className={s.controls}>{([
      ['Predicted mask pixels', predicted, setPredicted],
      ['Ground-truth mask pixels', groundTruth, setGroundTruth],
      ['Overlapping pixels', requestedOverlap, setOverlap],
    ] as const).map(([label, value, setter]) => <label className={s.control} key={label}>{label}: {value}
      <input type="range" min="0" max={label === 'Overlapping pixels' ? Math.min(predicted, groundTruth) : 200} step="1"
        value={label === 'Overlapping pixels' ? overlap : value} aria-label={label}
        onChange={event => setter(Number(event.target.value))} />
    </label>)}</div>}>
    <div style={{display: 'grid', gap: '0.7rem'}}>
      {([['Predicted', predicted], ['Ground truth', groundTruth], ['Intersection', overlap]] as const).map(([label, count], index) => <div key={label}>
        <div>{label}: {count} pixels</div>
        <div style={{height: '1rem', background: dark ? '#293142' : '#e7eaf0', borderRadius: '0.25rem', overflow: 'hidden'}}>
          <div style={{height: '100%', width: `${count / 200 * 100}%`, background: seriesColor(index, dark)}} />
        </div>
      </div>)}
      <output>Union {union}; IoU {format(iou)}; Dice {format(dice)}</output>
    </div>
  </VizPanel>;
}
