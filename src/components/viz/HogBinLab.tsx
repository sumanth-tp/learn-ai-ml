import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function HogBinLab() {
  const dark = useDarkViz();
  const [angle, setAngle] = useState(50);
  const [magnitude, setMagnitude] = useState(5);
  const bin = Math.floor(angle / 20);
  const weights = Array.from({length: 9}, (_, index) => index === bin ? magnitude : 0);
  const rows: [string, string][] = [
    ['Unsigned orientation', `${angle}°`], ['Magnitude', String(magnitude)],
    ['Selected bin', `${bin} (${bin * 20}–${(bin + 1) * 20}°)`],
    ...weights.map((weight, index): [string, string] => [`Bin ${index} · ${index * 20}–${(index + 1) * 20}°`, String(weight)]),
  ];

  return <VizPanel title="Put one gradient into nine HoG bins"
    hint="This is hard assignment for one synthetic gradient. Full HoG may interpolate votes between bins and normalises groups of cells."
    table={{columns: ['Measure', 'Value'], rows}}
    controls={<div className={s.controls}>
      <label className={s.control}>Unsigned gradient orientation: {angle}°
        <input type="range" min="0" max="179" step="1" value={angle} aria-label="Unsigned gradient orientation" onChange={event => setAngle(Number(event.target.value))} />
      </label>
      <label className={s.control}>Gradient magnitude: {magnitude}
        <input type="range" min="0" max="10" step="1" value={magnitude} aria-label="Gradient magnitude for HoG" onChange={event => setMagnitude(Number(event.target.value))} />
      </label>
    </div>}>
    <div style={{display: 'grid', gridTemplateColumns: 'repeat(9, minmax(0, 1fr))', gap: '0.25rem'}} role="img" aria-label={`Gradient at ${angle} degrees with magnitude ${magnitude} votes in bin ${bin}`}>
      {weights.map((weight, index) => <div key={index} style={{textAlign: 'center'}}>
        <div style={{height: '4rem', display: 'flex', alignItems: 'end', background: dark ? '#293142' : '#e7eaf0', borderRadius: '0.2rem', overflow: 'hidden'}}>
          <div style={{width: '100%', height: `${weight * 10}%`, background: seriesColor(index, dark)}} />
        </div>
        <small>{index}</small>
      </div>)}
      <output style={{gridColumn: '1 / -1'}}>Bin {bin} covers {bin * 20}–{(bin + 1) * 20}° and receives weight {magnitude}.</output>
    </div>
  </VizPanel>;
}
