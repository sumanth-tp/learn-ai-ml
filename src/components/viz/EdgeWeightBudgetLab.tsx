import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const models = {mobile: {name: 'MobileNet V1 1.0-224', millions: 4.2}, vgg: {name: 'VGG-16', millions: 138}};

export default function EdgeWeightBudgetLab() {
  const dark = useDarkViz();
  const [model, setModel] = useState<'mobile' | 'vgg'>('mobile');
  const [bits, setBits] = useState(8);
  const selected = models[model];
  const fullMegabytes = selected.millions * 32 / 8;
  const selectedMegabytes = selected.millions * bits / 8;
  const ratio = fullMegabytes / selectedMegabytes;
  const parameterRatio = models.vgg.millions / models.mobile.millions;
  const rows: [string, string][] = [
    ['Paper model example', selected.name], ['Rounded parameters', `${selected.millions} million`],
    ['Selected precision', `${bits} bits`], ['32-bit raw weights', `${fullMegabytes.toFixed(1)} decimal MB`],
    ['Selected raw weights', `${selectedMegabytes.toFixed(1)} decimal MB`],
    ['Idealised storage reduction', `${ratio.toFixed(1)}×`],
    ['VGG-16 / MobileNet parameters', `${parameterRatio.toFixed(3)}×`],
  ];

  return <VizPanel title="Estimate a raw weight budget"
    hint="This calculation covers densely stored parameter values only. Quantisation scales, model packaging, activations and hardware operations can change real size, memory use and latency."
    table={{columns: ['Measure', 'Value'], rows}}
    controls={<div className={s.controls}>
      <label className={s.control}>Paper model example
        <select value={model} aria-label="Paper model example" onChange={event => setModel(event.target.value as 'mobile' | 'vgg')}>
          <option value="mobile">MobileNet V1 1.0-224</option>
          <option value="vgg">VGG-16</option>
        </select>
      </label>
      <label className={s.control}>Stored weight precision
        <select value={bits} aria-label="Stored weight precision" onChange={event => setBits(Number(event.target.value))}>
          {[32, 16, 8, 4].map(value => <option value={value} key={value}>{value} bits</option>)}
        </select>
      </label>
    </div>}>
    <div style={{display: 'grid', gap: '0.7rem'}}>
      {([['32-bit baseline', fullMegabytes, 0], [`${bits}-bit raw weights`, selectedMegabytes, 1]] as const).map(([label, size, index]) => <div key={label}>
        <div>{label}: {size.toFixed(1)} decimal MB</div>
        <div style={{height: '1.1rem', background: dark ? '#293142' : '#e7eaf0', borderRadius: '0.25rem', overflow: 'hidden'}}>
          <div style={{height: '100%', width: `${size / fullMegabytes * 100}%`, background: seriesColor(index, dark)}} />
        </div>
      </div>)}
      <output>{ratio.toFixed(1)}× idealised raw weight saving; VGG-16 has {parameterRatio.toFixed(3)}× the rounded parameters of MobileNet in the original comparison.</output>
    </div>
  </VizPanel>;
}
