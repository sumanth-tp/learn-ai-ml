import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function PixelClusterLab() {
  const dark = useDarkViz();
  const [pixel, setPixel] = useState(120);
  const [first, setFirst] = useState(50);
  const [second, setSecond] = useState(200);
  const firstDistance = Math.abs(pixel - first);
  const secondDistance = Math.abs(pixel - second);
  const tie = firstDistance === secondDistance;
  const assigned = firstDistance <= secondDistance ? 1 : 2;
  const rows: [string, string][] = [
    ['Pixel intensity', String(pixel)], ['Centre 1', String(first)], ['Centre 2', String(second)],
    ['Distance to centre 1', String(firstDistance)], ['Distance to centre 2', String(secondDistance)],
    ['Assignment', tie ? 'Tie; centre 1 by convention' : `Centre ${assigned}`],
  ];

  return <VizPanel title="Assign one pixel to a colour cluster"
    hint="This is one assignment step. Full k-means also recomputes centres from assigned pixels and repeats. Spatial neighbours are ignored here."
    table={{columns: ['Measure', 'Value'], rows}}
    controls={<div className={s.controls}>{([
      ['Pixel intensity', pixel, setPixel], ['Centre 1', first, setFirst], ['Centre 2', second, setSecond],
    ] as const).map(([label, value, setter]) => <label className={s.control} key={label}>{label}: {value}
      <input type="range" min="0" max="255" step="1" value={value} aria-label={label}
        onChange={event => setter(Number(event.target.value))} />
    </label>)}</div>}>
    <div style={{display: 'grid', gap: '0.7rem'}}>
      <div style={{position: 'relative', height: '2rem', margin: '1.2rem 0', background: dark ? '#293142' : '#e7eaf0', borderRadius: '0.25rem'}} role="img" aria-label={`Intensity ruler: centre one at ${first}, pixel at ${pixel}, centre two at ${second}`}>
        {[[first, 'C1', seriesColor(0, dark)], [second, 'C2', seriesColor(1, dark)], [pixel, 'P', seriesColor(2, dark)]].map(([value, label, colour], index) =>
          <span key={index} style={{position: 'absolute', left: `${Number(value) / 255 * 100}%`, top: index === 2 ? '0.1rem' : '0.8rem', transform: 'translateX(-50%)', color: String(colour), fontWeight: 700}}>{label}</span>)}
      </div>
      <output>|{pixel} − {first}| = {firstDistance}; |{pixel} − {second}| = {secondDistance}. {tie ? 'Tie: centre 1 by convention.' : `Assign centre ${assigned}.`}</output>
    </div>
  </VizPanel>;
}
