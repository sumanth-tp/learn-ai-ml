import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function PartitionSkewLab() {
  const dark = useDarkViz();
  const [gib, setGib] = useState(10);
  const [targetMib, setTargetMib] = useState(128);
  const [largestShare, setLargestShare] = useState(50);
  const idealPartitions = Math.ceil(gib * 1024 / targetMib);
  const largestGib = gib * largestShare / 100;

  return (
    <VizPanel
      title="Partitions, parallelism and a heavy key"
      hint="The size division is an ideal count. Real file boundaries and shuffle plans change task counts; tasks do not all run concurrently."
      table={{columns: ['measure', 'value'], rows: [
        ['dataset size', `${gib} GiB`],
        ['target partition size', `${targetMib} MiB`],
        ['ideal size-based partitions', String(idealPartitions)],
        ['largest key share', `${largestShare}%`],
        ['largest key mass', `${largestGib.toFixed(1)} GiB`],
      ]}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            dataset size: {gib} GiB
            <input type="range" min="1" max="20" value={gib} aria-label="Dataset size in GiB"
              onChange={(event) => setGib(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            target partition: {targetMib} MiB
            <input type="range" min="64" max="256" step="64" value={targetMib} aria-label="Target partition size in MiB"
              onChange={(event) => setTargetMib(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            largest key share: {largestShare}%
            <input type="range" min="10" max="80" value={largestShare} aria-label="Largest key percentage"
              onChange={(event) => setLargestShare(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.75rem'}}>
        <strong>{gib} GiB / {targetMib} MiB ≈ {idealPartitions} partitions</strong>
        <div style={{height: '1.5rem', borderRadius: '0.25rem', background: 'var(--ifm-color-emphasis-200)', overflow: 'hidden'}}
          role="img" aria-label={`Largest key holds ${largestShare} per cent of data`}>
          <div style={{height: '100%', width: `${largestShare}%`, background: seriesColor(0, dark)}} />
        </div>
        <output>One key contains {largestGib.toFixed(1)} GiB and may create a straggler after grouping.</output>
      </div>
    </VizPanel>
  );
}
