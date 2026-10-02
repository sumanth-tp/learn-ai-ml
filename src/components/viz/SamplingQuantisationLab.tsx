import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function SamplingQuantisationLab() {
  const dark = useDarkViz();
  const [side, setSide] = useState(512);
  const [channels, setChannels] = useState(3);
  const [bits, setBits] = useState(8);
  const levels = 2 ** bits;
  const bytes = Math.ceil(side * side * channels * bits / 8);
  const kib = bytes / 1024;
  const rows: [string, string][] = [
    ['Image', `${side} × ${side} pixels`],
    ['Channels', String(channels)],
    ['Bit depth', `${bits} bits per channel`],
    ['Levels per channel', String(levels)],
    ['Packed minimum', `${bytes.toLocaleString()} bytes`],
  ];

  return (
    <VizPanel
      title="Sample space, quantise values"
      hint="The storage figure is a packed-bit minimum before file headers and compression. Real arrays may pad each sample to a byte or word."
      table={{columns: ['Measure', 'Value'], rows}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            Image side: {side} pixels
            <input type="range" min="64" max="1024" step="64" value={side} aria-label="Image side in pixels"
              onChange={event => setSide(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Channels: {channels}
            <select aria-label="Number of image channels" value={channels} onChange={event => setChannels(Number(event.target.value))}>
              <option value="1">1, grayscale</option>
              <option value="3">3, RGB</option>
            </select>
          </label>
          <label className={s.control}>
            Bits per channel: {bits}
            <input type="range" min="1" max="16" step="1" value={bits} aria-label="Bits per channel"
              onChange={event => setBits(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.8rem'}}>
        <strong>{levels.toLocaleString()} levels per channel · {bytes.toLocaleString()} packed bytes ({kib.toFixed(1)} KiB)</strong>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: '0.3rem', maxWidth: '16rem'}} role="img" aria-label="Illustrative spatial pixel grid">
          {Array.from({length: 32}, (_, index) => (
            <span key={index} style={{aspectRatio: '1', borderRadius: '0.2rem', background: seriesColor(index % channels, dark), opacity: 0.3 + 0.6 * ((index % 8) / 7)}} />
          ))}
        </div>
        <output>More samples resolve finer spatial variation; more levels represent finer intensity steps.</output>
      </div>
    </VizPanel>
  );
}
