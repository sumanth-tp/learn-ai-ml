import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function PerspectiveProjectionLab() {
  const dark = useDarkViz();
  const [focal, setFocal] = useState(2);
  const [depth, setDepth] = useState(2);
  const projected = focal / depth;
  const rows: [string, string, string, string][] = [
    ['Near', `(1, 1, ${depth})`, projected.toFixed(2), projected.toFixed(2)],
    ['Far', `(2, 2, ${2 * depth})`, projected.toFixed(2), projected.toFixed(2)],
  ];

  return (
    <VizPanel
      title="Two depths, one projected pixel"
      hint="Both 3D points lie on the same ray. A single pinhole view cannot distinguish their depths without more evidence."
      table={{columns: ['Point', '3D coordinates', 'Image x', 'Image y'], rows}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            Focal length: {focal.toFixed(1)}
            <input type="range" min="1" max="5" step="0.5" value={focal} aria-label="Focal length"
              onChange={event => setFocal(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Near point depth: {depth.toFixed(1)}
            <input type="range" min="1" max="5" step="0.5" value={depth} aria-label="Near point depth"
              onChange={event => setDepth(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.8rem'}}>
        <strong>Near and far point both project to ({projected.toFixed(2)}, {projected.toFixed(2)})</strong>
        <div style={{position: 'relative', height: '5rem', borderRadius: '0.4rem', border: `1px solid ${seriesColor(0, dark)}`, background: dark ? '#1c2433' : '#eef5ff'}} role="img" aria-label={`Both points overlap at image coordinate ${projected.toFixed(2)}, ${projected.toFixed(2)}`}>
          <div style={{position: 'absolute', left: `${Math.min(90, projected * 16 + 8)}%`, top: '50%', width: '1rem', height: '1rem', borderRadius: '50%', transform: 'translate(-50%, -50%)', background: seriesColor(1, dark)}} />
        </div>
        <output>Different depths can share this image measurement.</output>
      </div>
    </VizPanel>
  );
}
