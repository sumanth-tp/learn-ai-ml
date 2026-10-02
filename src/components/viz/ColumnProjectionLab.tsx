import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function ColumnProjectionLab() {
  const dark = useDarkViz();
  const [columns, setColumns] = useState(50);
  const [selected, setSelected] = useState(2);
  const fraction = selected / columns;
  const parquetGb = 2;
  const idealGb = parquetGb * fraction;

  return (
    <VizPanel
      title="How much can a column projection avoid?"
      hint="The estimate assumes equal-sized columns and ignores metadata and row-group overhead. A real scan depends on layout and filters."
      table={{columns: ['layout', 'stored GB', 'fraction selected', 'ideal GB read'], rows: [
        ['row scan', '10.00', '1.000', '10.000'],
        ['Parquet projection', parquetGb.toFixed(2), fraction.toFixed(3), idealGb.toFixed(3)],
      ]}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            columns in the dataset: {columns}
            <input type="range" min="10" max="100" value={columns} aria-label="Total columns"
              onChange={(event) => {
                const next = Number(event.target.value);
                setColumns(next);
                setSelected((current) => Math.min(current, next));
              }} />
          </label>
          <label className={s.control}>
            columns in the query: {selected}
            <input type="range" min="1" max={columns} value={selected} aria-label="Selected columns"
              onChange={(event) => setSelected(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.85rem'}}>
        {[
          {label: 'Row scan', value: 10, detail: '10.000 GB'},
          {label: 'Ideal Parquet projection', value: idealGb, detail: `${idealGb.toFixed(3)} GB`},
        ].map((item, index) => (
          <div key={item.label} style={{display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 6rem', gap: '0.6rem', alignItems: 'center'}}>
            <div>
              <strong>{item.label}</strong>
              <div style={{height: '1.25rem', borderRadius: '0.25rem', background: 'var(--ifm-color-emphasis-200)', overflow: 'hidden'}}>
                <div style={{height: '100%', width: `${item.value * 10}%`, background: seriesColor(index, dark)}} />
              </div>
            </div>
            <output>{item.detail}</output>
          </div>
        ))}
      </div>
    </VizPanel>
  );
}
