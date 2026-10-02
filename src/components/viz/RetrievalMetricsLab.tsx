import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function RetrievalMetricsLab() {
  const dark = useDarkViz();
  const [relevantShown, setRelevantShown] = useState(40);
  const [totalRelevant, setTotalRelevant] = useState(60);
  const shown = 50;
  const precision = relevantShown / shown;
  const recall = relevantShown / totalRelevant;
  const f1 = precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall);
  const measures = [
    {name: 'precision', value: precision, detail: `${relevantShown} / ${shown}`},
    {name: 'recall', value: recall, detail: `${relevantShown} / ${totalRelevant}`},
    {name: 'F1', value: f1, detail: 'harmonic mean'},
  ];

  return (
    <VizPanel
      title="Retrieval quality: what did the search find?"
      hint="Precision asks how clean the result set is. Recall asks how much relevant material it found. F1 falls when either one is low."
      table={{columns: ['measure', 'calculation', 'value'], rows: measures.map((m) => [m.name, m.detail, m.value.toFixed(3)])}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            relevant among 50 shown: {relevantShown}
            <input type="range" min="0" max={Math.min(50, totalRelevant)} value={relevantShown}
              onChange={(event) => setRelevantShown(Number(event.target.value))} aria-label="Relevant documents among 50 shown" />
          </label>
          <label className={s.control}>
            relevant in collection: {totalRelevant}
            <input type="range" min="1" max="100" value={totalRelevant}
              onChange={(event) => {
                const next = Number(event.target.value);
                setTotalRelevant(next);
                setRelevantShown((current) => Math.min(current, next));
              }} aria-label="Total relevant documents in the collection" />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.8rem'}}>
        {measures.map((measure, index) => (
          <div key={measure.name} style={{display: 'grid', gridTemplateColumns: '5.5rem minmax(0, 1fr) 3.5rem', gap: '0.6rem', alignItems: 'center'}}>
            <strong>{measure.name}</strong>
            <div style={{height: '1.25rem', borderRadius: '0.25rem', background: 'var(--ifm-color-emphasis-200)', overflow: 'hidden'}}>
              <div style={{height: '100%', width: `${measure.value * 100}%`, background: seriesColor(index, dark)}} />
            </div>
            <output>{measure.value.toFixed(3)}</output>
          </div>
        ))}
      </div>
    </VizPanel>
  );
}
