import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const NOISE_AUC = {allRows: 0.946, insideFolds: 0.526};

export default function SplitGateLab() {
  const dark = useDarkViz();
  const [examples, setExamples] = useState(10000);
  const [trainPct, setTrainPct] = useState(70);
  const [valPct, setValPct] = useState(15);
  const [folds, setFolds] = useState(5);
  const [selectOnAll, setSelectOnAll] = useState(false);
  const train = Math.floor((examples * trainPct) / 100);
  const validation = Math.floor((examples * valPct) / 100);
  const test = examples - train - validation;
  const heldOut = Math.floor(examples / folds);
  const auc = selectOnAll ? NOISE_AUC.allRows : NOISE_AUC.insideFolds;
  const bars = [
    {label: 'train', value: train},
    {label: 'validation', value: validation},
    {label: 'test', value: test},
  ];

  return (
    <VizPanel
      title="Split counts and the cost of fitting before splitting"
      hint="Counts follow the chapter's 10,000 example split. The two AUC values come from the noise-feature experiment in this chapter and do not change with the sliders."
      table={{columns: ['quantity', 'value'], rows: [
        ['train', String(train)],
        ['validation', String(validation)],
        ['test', String(test)],
        [`held out per fold (${folds} folds)`, String(heldOut)],
        ['AUC on pure noise', auc.toFixed(3)],
      ]}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            examples: {examples}
            <input type="range" min="1000" max="100000" step="1000" value={examples} aria-label="Examples"
              onChange={(event) => setExamples(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            train share: {trainPct}%
            <input type="range" min="40" max="90" value={trainPct} aria-label="Train share"
              onChange={(event) => {
                const next = Number(event.target.value);
                setTrainPct(next);
                setValPct((current) => Math.min(current, 95 - next));
              }} />
          </label>
          <label className={s.control}>
            validation share: {valPct}%
            <input type="range" min="5" max={Math.max(5, 95 - trainPct)} value={valPct} aria-label="Validation share"
              onChange={(event) => setValPct(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            cross-validation folds: {folds}
            <input type="range" min="2" max="10" value={folds} aria-label="Folds"
              onChange={(event) => setFolds(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={selectOnAll} aria-label="Select features on all rows"
              onChange={(event) => setSelectOnAll(event.target.checked)} />
            {' '}select features on all rows before cross-validating
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.6rem'}}>
        {bars.map((bar, index) => (
          <div key={bar.label} style={{display: 'grid', gridTemplateColumns: '6rem minmax(0, 1fr) 5rem', gap: '0.6rem', alignItems: 'center'}}>
            <span>{bar.label}</span>
            <div style={{height: '1.1rem', borderRadius: '0.25rem', background: 'var(--ifm-color-emphasis-200)', overflow: 'hidden'}}>
              <div style={{height: '100%', width: `${(bar.value / examples) * 100}%`, background: seriesColor(index, dark)}} />
            </div>
            <output>{bar.value}</output>
          </div>
        ))}
        <strong>{heldOut} examples held out per fold; AUC on pure noise {auc.toFixed(3)}</strong>
        <output>{selectOnAll ? 'Chance is 0.5. Selecting on all rows lets the labels leak into the features.' : 'Selection inside each fold stays near chance, as it should on noise.'}</output>
      </div>
    </VizPanel>
  );
}
