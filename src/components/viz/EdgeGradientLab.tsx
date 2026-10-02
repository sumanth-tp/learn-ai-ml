import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function EdgeGradientLab() {
  const dark = useDarkViz();
  const [gx, setGx] = useState(4);
  const [gy, setGy] = useState(3);
  const magnitude = Math.hypot(gx, gy);
  const angle = magnitude === 0 ? undefined : Math.atan2(gy, gx) * 180 / Math.PI;
  const rows: [string, string][] = [
    ['Horizontal gradient Gx', String(gx)],
    ['Vertical gradient Gy', String(gy)],
    ['Magnitude', magnitude.toFixed(2)],
    ['Orientation', angle === undefined ? 'undefined at zero gradient' : `${angle.toFixed(2)}°`],
  ];

  return (
    <VizPanel
      title="Gradient strength and direction"
      hint="The vector points across the local intensity change. It does not identify which object caused the edge."
      table={{columns: ['Measure', 'Value'], rows}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            Horizontal gradient: {gx}
            <input type="range" min="-10" max="10" step="1" value={gx} aria-label="Horizontal gradient Gx"
              onChange={event => setGx(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Vertical gradient: {gy}
            <input type="range" min="-10" max="10" step="1" value={gy} aria-label="Vertical gradient Gy"
              onChange={event => setGy(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', justifyItems: 'center', gap: '0.5rem'}}>
        <svg viewBox="0 0 240 180" width="240" height="180" role="img" aria-label={`Gradient vector ${gx}, ${gy}`}>
          <line x1="20" y1="90" x2="220" y2="90" stroke={seriesColor(2, dark)} strokeWidth="1" />
          <line x1="120" y1="10" x2="120" y2="170" stroke={seriesColor(2, dark)} strokeWidth="1" />
          <line x1="120" y1="90" x2={120 + gx * 8} y2={90 - gy * 8} stroke={seriesColor(0, dark)} strokeWidth="4" />
          <circle cx={120 + gx * 8} cy={90 - gy * 8} r="5" fill={seriesColor(0, dark)} />
        </svg>
        <output>Magnitude {magnitude.toFixed(2)} · orientation {angle === undefined ? 'undefined' : `${angle.toFixed(2)}°`}</output>
      </div>
    </VizPanel>
  );
}
