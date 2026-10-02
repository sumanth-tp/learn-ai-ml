import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function HoughVoteLab() {
  const dark = useDarkViz();
  const [x, setX] = useState(2);
  const [y, setY] = useState(2);
  const [angle, setAngle] = useState(45);
  const radians = angle * Math.PI / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  const rho = x * cosine + y * sine;
  const rows: [string, string][] = [
    ['Point', `(${x}, ${y})`],
    ['Normal angle', `${angle}°`],
    ['Cosine', cosine.toFixed(3)],
    ['Sine', sine.toFixed(3)],
    ['Rho', rho.toFixed(3)],
  ];

  return (
    <VizPanel
      title="One Hough line vote"
      hint="One edge point votes for many parameter cells. A line needs consistent votes from several points."
      table={{columns: ['Measure', 'Value'], rows}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            Point x: {x}
            <input type="range" min="-5" max="5" step="1" value={x} aria-label="Hough point x"
              onChange={event => setX(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Point y: {y}
            <input type="range" min="-5" max="5" step="1" value={y} aria-label="Hough point y"
              onChange={event => setY(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Normal angle: {angle}°
            <input type="range" min="0" max="180" step="15" value={angle} aria-label="Hough normal angle"
              onChange={event => setAngle(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', justifyItems: 'center', gap: '0.5rem'}}>
        <svg viewBox="0 0 240 180" width="240" height="180" role="img" aria-label={`Point ${x}, ${y} casts rho vote ${rho.toFixed(3)}`}>
          <line x1="120" y1="90" x2={120 + cosine * 70} y2={90 - sine * 70} stroke={seriesColor(0, dark)} strokeWidth="3" />
          <circle cx={120 + x * 10} cy={90 - y * 10} r="6" fill={seriesColor(1, dark)} />
          <circle cx="120" cy="90" r="3" fill={seriesColor(2, dark)} />
        </svg>
        <output>ρ = {x} × {cosine.toFixed(3)} + {y} × {sine.toFixed(3)} = {rho.toFixed(3)}</output>
      </div>
    </VizPanel>
  );
}
