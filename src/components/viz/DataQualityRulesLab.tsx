import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function DataQualityRulesLab() {
  const dark = useDarkViz();
  const [nulls, setNulls] = useState(50);
  const [threshold, setThreshold] = useState(99);
  const rows = 1000;
  const complete = rows - nulls;
  const percentage = complete / rows * 100;
  const passed = percentage >= threshold;

  return (
    <VizPanel
      title="A completeness rule is a decision"
      hint="The 1,000-row example measures one required field. Other fields and quality dimensions need separate checks."
      table={{columns: ['measure', 'value'], rows: [
        ['rows', String(rows)],
        ['non-null', String(complete)],
        ['null', String(nulls)],
        ['completeness', `${percentage.toFixed(1)}%`],
        ['required threshold', `${threshold}%`],
        ['result', passed ? 'pass' : 'fail'],
      ]}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            null values: {nulls}
            <input type="range" min="0" max="150" value={nulls} aria-label="Null values"
              onChange={(event) => setNulls(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            required completeness: {threshold}%
            <input type="range" min="90" max="100" value={threshold} aria-label="Required completeness percentage"
              onChange={(event) => setThreshold(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.8rem'}}>
        <strong>{percentage.toFixed(1)}% complete; rule {passed ? 'passes' : 'fails'} at {threshold}%</strong>
        <div style={{height: '1.5rem', borderRadius: '0.25rem', background: seriesColor(2, dark), overflow: 'hidden'}}
          role="img" aria-label={`${complete} complete values and ${nulls} null values`}>
          <div style={{height: '100%', width: `${percentage}%`, background: seriesColor(0, dark)}} />
        </div>
        <output>{complete} present · {nulls} missing</output>
      </div>
    </VizPanel>
  );
}
