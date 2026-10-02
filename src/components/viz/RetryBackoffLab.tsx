import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function RetryBackoffLab() {
  const dark = useDarkViz();
  const [base, setBase] = useState(2);
  const [count, setCount] = useState(3);
  const delays = Array.from({length: count}, (_, index) => base * 2 ** index);
  const total = delays.reduce((sum, delay) => sum + delay, 0);

  return (
    <VizPanel
      title="Exponential retry waits"
      hint="These are scheduled waits before retries. Real systems may add jitter, a maximum delay, queue time and task runtime."
      table={{columns: ['retry', 'delay seconds', 'cumulative wait seconds'], rows: delays.map((delay, index) => [
        String(index + 1), String(delay), String(delays.slice(0, index + 1).reduce((sum, item) => sum + item, 0)),
      ])}}
      controls={
        <div className={s.controls}>
          <label className={s.control}>
            base delay: {base} s
            <input type="range" min="1" max="5" value={base} aria-label="Base retry delay"
              onChange={(event) => setBase(Number(event.target.value))} />
          </label>
          <label className={s.control}>
            retry count: {count}
            <input type="range" min="1" max="5" value={count} aria-label="Retry count"
              onChange={(event) => setCount(Number(event.target.value))} />
          </label>
        </div>
      }>
      <div style={{display: 'grid', gap: '0.6rem'}}>
        <strong>Waits {delays.join(', ')} s; total {total} s</strong>
        {delays.map((delay, index) => (
          <div key={index} style={{display: 'grid', gridTemplateColumns: '3.5rem minmax(0, 1fr) 3rem', gap: '0.5rem', alignItems: 'center'}}>
            <span>retry {index + 1}</span>
            <div style={{height: '1rem', borderRadius: '0.25rem', background: 'var(--ifm-color-emphasis-200)', overflow: 'hidden'}}>
              <div style={{height: '100%', width: `${delay / 80 * 100}%`, background: seriesColor(index, dark)}} />
            </div>
            <output>{delay} s</output>
          </div>
        ))}
      </div>
    </VizPanel>
  );
}
