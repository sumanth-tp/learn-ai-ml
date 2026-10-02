import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function CrawlerRateLab() {
  const dark = useDarkViz();
  const [hosts, setHosts] = useState(500);
  const [delay, setDelay] = useState(1);
  const perHost = 1 / delay;
  const total = hosts * perHost;
  const table = {columns: ['quantity', 'value', 'interpretation'], rows: [
    ['active hosts', hosts, 'independent host queues'],
    ['delay per host', `${delay} s`, 'minimum time between requests to one host'],
    ['per-host rate', `${perHost.toFixed(2)} pages/s`, 'one page per delay interval'],
    ['aggregate rate', `${total.toFixed(0)} pages/s`, 'idealised throughput before latency and errors'],
  ]};

  return <VizPanel title="Scale a polite crawler across hosts"
    hint="More hosts can raise aggregate throughput while keeping a fixed minimum delay for each host. This ignores network and processing bottlenecks."
    table={table}
    controls={<div className={s.controls}>
      <label className={s.control}>parallel hosts: {hosts}
        <input type="range" min="100" max="1000" step="100" value={hosts} aria-label="Parallel hosts"
          onChange={(event) => setHosts(Number(event.target.value))} />
      </label>
      <label className={s.control}>minimum host delay: {delay} s
        <input type="range" min="1" max="5" step="1" value={delay} aria-label="Minimum per-host delay"
          onChange={(event) => setDelay(Number(event.target.value))} />
      </label>
    </div>}>
    <div style={{display: 'grid', gap: '0.7rem'}}>
      <div>one host: <strong>{perHost.toFixed(2)} pages/s</strong></div>
      <div style={{height: '1.5rem', background: 'var(--ifm-color-emphasis-200)', borderRadius: '0.25rem'}}>
        <div style={{height: '100%', width: `${total / 1000 * 100}%`, background: seriesColor(0, dark), borderRadius: '0.25rem'}} />
      </div>
      <div aria-live="polite">{hosts} hosts × {perHost.toFixed(2)} pages/s = <strong>{total.toFixed(0)} pages/s</strong></div>
    </div>
  </VizPanel>;
}
