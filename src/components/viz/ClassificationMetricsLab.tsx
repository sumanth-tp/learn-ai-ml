import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function ClassificationMetricsLab() {
  const dark = useDarkViz();
  const [firstLogit, setFirstLogit] = useState(2);
  const [tp, setTp] = useState(40);
  const [fp, setFp] = useState(10);
  const [fn, setFn] = useState(20);
  const logits = [firstLogit, 1, 0];
  const shifted = logits.map(value => Math.exp(value - Math.max(...logits)));
  const sum = shifted.reduce((total, value) => total + value, 0);
  const probabilities = shifted.map(value => value / sum);
  const precision = tp + fp ? tp / (tp + fp) : null;
  const recall = tp + fn ? tp / (tp + fn) : null;
  const f1 = 2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : null;
  const formatted = (value: number | null) => value === null ? 'undefined' : value.toFixed(3);
  const rows: [string, string][] = [
    ['Logits', `[${logits.join(', ')}]`],
    ...probabilities.map((value, index): [string, string] => [`Class ${index + 1} probability`, value.toFixed(3)]),
    ['True positives', String(tp)], ['False positives', String(fp)], ['False negatives', String(fn)],
    ['Precision', formatted(precision)], ['Recall', formatted(recall)], ['F1', formatted(f1)],
  ];

  return (
    <VizPanel title="From logits to a classification decision"
      hint="Softmax describes one prediction. Precision, recall and F1 summarise labelled outcomes at a chosen decision threshold; they do not follow from the logits alone."
      table={{columns: ['Measure', 'Value'], rows}}
      controls={<div className={s.controls}>
        {([
          ['First logit', firstLogit, setFirstLogit, -2, 4],
          ['True positives', tp, setTp, 0, 100],
          ['False positives', fp, setFp, 0, 100],
          ['False negatives', fn, setFn, 0, 100],
        ] as const).map(([label, value, setter, min, max]) =>
          <label className={s.control} key={label}>{label}: {value}
            <input type="range" min={min} max={max} step="1" value={value} aria-label={label}
              onChange={event => setter(Number(event.target.value))} />
          </label>)}
      </div>}>
      <div style={{display: 'grid', gap: '0.6rem'}}>
        {probabilities.map((value, index) => <div key={index}>
          <div>Class {index + 1}: {value.toFixed(3)}</div>
          <div style={{height: '1rem', background: dark ? '#293142' : '#e7eaf0', borderRadius: '0.25rem', overflow: 'hidden'}}>
            <div style={{height: '100%', width: `${value * 100}%`, background: seriesColor(index, dark)}} />
          </div>
        </div>)}
        <output>Precision {formatted(precision)} · recall {formatted(recall)} · F1 {formatted(f1)}</output>
      </div>
    </VizPanel>
  );
}
