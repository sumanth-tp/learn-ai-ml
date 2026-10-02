import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

function variableBytes(value: number) {
  const bytes: number[] = [];
  let remaining = value;
  do {
    bytes.unshift(remaining % 128);
    remaining = Math.floor(remaining / 128);
  } while (remaining > 0);
  bytes[bytes.length - 1] += 128;
  return bytes;
}

export default function GapEncodingLab() {
  const dark = useDarkViz();
  const [first, setFirst] = useState(5);
  const [secondGap, setSecondGap] = useState(125);
  const [thirdGap, setThirdGap] = useState(2);
  const ids = [first, first + secondGap, first + secondGap + thirdGap];
  const gaps = [first, secondGap, thirdGap];
  const encoded = gaps.map(variableBytes);
  const compressed = encoded.reduce((total, bytes) => total + bytes.length, 0);
  const hex = (bytes: number[]) => bytes.map((byte) => `0x${byte.toString(16).toUpperCase().padStart(2, '0')}`).join(' ');
  const controls = [
    {name: 'first document ID', value: first, min: 1, max: 30, set: setFirst},
    {name: 'second gap', value: secondGap, min: 1, max: 255, set: setSecondGap},
    {name: 'third gap', value: thirdGap, min: 1, max: 30, set: setThirdGap},
  ];

  return (
    <VizPanel title="Gap and variable-byte encoding"
      hint="The first gap equals the first ID; each later gap is the distance from the previous ID. Values below 128 fit in one variable byte under this convention."
      table={{columns: ['posting', 'absolute ID', 'gap', 'encoded bytes'], rows: ids.map((id, index) => [index + 1, id, gaps[index], hex(encoded[index])])}}
      controls={<div className={s.controls}>{controls.map((control) => (
        <label className={s.control} key={control.name}>{control.name}: {control.value}
          <input type="range" min={control.min} max={control.max} value={control.value}
            onChange={(event) => control.set(Number(event.target.value))} aria-label={control.name} />
        </label>
      ))}</div>}>
      <div style={{display: 'grid', gap: '0.6rem'}}>
        <div>postings: <strong>{ids.join(' · ')}</strong></div>
        <div>gaps: <strong>{gaps.join(' · ')}</strong></div>
        <div>bytes: <code>{encoded.map(hex).join(' · ')}</code></div>
        <div style={{display: 'grid', gap: '0.4rem', marginTop: '0.3rem'}}>
          <div>compressed: <strong>{compressed} bytes</strong>
            <div style={{height: '1.1rem', width: `${compressed / 12 * 100}%`, minWidth: '1rem', background: seriesColor(0, dark), borderRadius: '0.25rem'}} />
          </div>
          <div>three 32-bit integers: <strong>12 bytes</strong>
            <div style={{height: '1.1rem', width: '100%', background: seriesColor(1, dark), borderRadius: '0.25rem'}} />
          </div>
        </div>
      </div>
    </VizPanel>
  );
}
