import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const FILES: Record<string, number[]> = {
  'a.parquet': [10, 20, 30],
  'b.parquet': [40, 50],
  'a_fixed.parquet': [10, 25, 30],
  'c_half_written.parquet': [999],
};

const SNAPSHOTS: Record<number, string[]> = {
  1: ['a.parquet'],
  2: ['a.parquet', 'b.parquet'],
  3: ['a_fixed.parquet', 'b.parquet'],
};

export default function SnapshotTimeTravelLab() {
  const dark = useDarkViz();
  const [version, setVersion] = useState(3);
  const [listing, setListing] = useState(false);
  const files = listing ? Object.keys(FILES) : SNAPSHOTS[version];
  const amounts = files.flatMap((name) => FILES[name]);
  const rows = amounts.length;
  const total = amounts.reduce((sum, value) => sum + value, 0);

  return (
    <VizPanel
      title="Read a table by manifest or by directory listing"
      hint="The manifest names the files of one committed snapshot. A directory listing also picks up superseded and half-written files."
      table={{columns: ['file', 'rows', 'sum of amount', 'read'], rows: Object.keys(FILES).map((name) => [
        name,
        String(FILES[name].length),
        String(FILES[name].reduce((sum, value) => sum + value, 0)),
        files.includes(name) ? 'yes' : 'no',
      ])}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            snapshot version: {version}
            <input type="range" min="1" max="3" value={version} aria-label="Snapshot version"
              disabled={listing} onChange={(event) => setVersion(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={listing} aria-label="Read the directory listing"
              onChange={(event) => setListing(event.target.checked)} />
            {' '}read the directory listing instead of the manifest
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.7rem'}}>
        <strong>{rows} rows, total amount {total}</strong>
        <div style={{display: 'flex', gap: '0.4rem', flexWrap: 'wrap'}} role="img"
          aria-label={`files read: ${files.join(', ')}`}>
          {Object.keys(FILES).map((name, index) => (
            <span key={name} style={{
              padding: '0.3rem 0.6rem', borderRadius: '0.25rem', fontSize: '0.85rem',
              background: files.includes(name) ? seriesColor(index, dark) : 'var(--ifm-color-emphasis-200)',
              color: files.includes(name) ? '#fff' : 'inherit',
            }}>{name}</span>
          ))}
        </div>
        <output>{listing ? 'Not a committed state: it double counts the corrected file and includes an unfinished write.' : `Snapshot ${version} is a consistent committed state.`}</output>
      </div>
    </VizPanel>
  );
}
