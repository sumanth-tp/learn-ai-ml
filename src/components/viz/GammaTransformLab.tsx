import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function GammaTransformLab() {
  const dark = useDarkViz();
  const [input, setInput] = useState(0.25);
  const [gamma, setGamma] = useState(0.5);
  const output = input ** gamma;
  const samples = [0, 0.25, 0.5, 0.75, 1].map(value => [value.toFixed(2), (value ** gamma).toFixed(3)] as [string, string]);
  const rows: [string, string][] = [['Input', input.toFixed(2)], ['Gamma', gamma.toFixed(1)], ['Output', output.toFixed(3)], ...samples.map(([x, y]) => [`Curve at ${x}`, y] as [string, string])];

  return (
    <VizPanel
      title="Gamma remaps intensity"
      hint="Values are normalised to 0–1. Gamma below 1 brightens intermediate values; gamma above 1 darkens them."
      table={{columns: ['Measure', 'Value'], rows}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            Input intensity: {input.toFixed(2)}
            <input type="range" min="0" max="1" step="0.01" value={input} aria-label="Input intensity"
              onChange={event => setInput(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            Gamma: {gamma.toFixed(1)}
            <input type="range" min="0.2" max="3" step="0.1" value={gamma} aria-label="Gamma"
              onChange={event => setGamma(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.65rem'}}>
        {([['Input', input, 0], ['Output', output, 1]] as const).map(([label, value, colour]) => (
          <div key={label}>
            <div>{label}: {value.toFixed(3)}</div>
            <div style={{height: '1.3rem', borderRadius: '0.25rem', background: dark ? '#293142' : '#e7eaf0', overflow: 'hidden'}}>
              <div style={{height: '100%', width: `${value * 100}%`, background: seriesColor(colour, dark)}} />
            </div>
          </div>
        ))}
        <output>{input.toFixed(2)} raised to {gamma.toFixed(1)} is {output.toFixed(3)}</output>
      </div>
    </VizPanel>
  );
}
