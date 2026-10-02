import {useMemo, useState} from 'react';

import {simulate} from './latencyBudgetModel';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 270;
const PAD = {top: 14, right: 16, bottom: 40, left: 44};
const BIN = 2;
const MAX_MS = 160;
const TIMEOUTS = [0, 20, 30, 40, 60];

export default function LatencyBudgetLab() {
  const dark = useDarkViz();
  const [parallel, setParallel] = useState(true);
  const [lookups, setLookups] = useState(8);
  const [lookupP99, setLookupP99] = useState(25);
  const [hedge, setHedge] = useState(false);
  const [timeout, setTimeoutMs] = useState(0);
  const [slo, setSlo] = useState(100);

  const r = useMemo(
    () =>
      simulate({
        lookups,
        lookupP50: 4,
        lookupP99,
        parallel,
        hedge,
        timeout: timeout === 0 ? null : timeout,
      }),
    [lookups, lookupP99, parallel, hedge, timeout],
  );

  const bins = useMemo(() => {
    const counts = new Array(MAX_MS / BIN + 1).fill(0);
    for (const t of r.totals) counts[Math.min(counts.length - 1, Math.floor(t / BIN))] += 1;
    return counts as number[];
  }, [r]);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const peak = Math.max(...bins);
  const x = (ms: number) => PAD.left + (ms / MAX_MS) * innerW;
  const barW = innerW / bins.length;
  const over = r.over(slo);
  const bar = seriesColor(0, dark);
  const accent = seriesColor(1, dark);

  const rows = [
    ['p50 (ms)', r.p50.toFixed(1)],
    ['p95 (ms)', r.p95.toFixed(1)],
    ['p99 (ms)', r.p99.toFixed(1)],
    ['p99.9 (ms)', r.p999.toFixed(1)],
    [`share over ${slo} ms`, `${(over * 100).toFixed(2)}%`],
    ['degraded by timeout', `${(r.degraded * 100).toFixed(2)}%`],
    ['extra lookup calls from hedging', `${(r.extra * 100).toFixed(2)}%`],
  ];

  return (
    <VizPanel
      title="Latency budget, 20,000 simulated decisions"
      hint="Network, model and rules stages are fixed. Change the feature fetch: series against parallel, more lookups, a slower tail, a hedged second call after the single-lookup p95, a timeout that falls back to default features. Defaults give p99 57.9 ms and 0.07% over 100 ms; switch to series for 104.2 ms and 1.39%."
      legend={[
        {label: 'requests per 2 ms bin', color: bar},
        {label: 'service level limit', color: accent},
      ]}
      table={{columns: ['measure', 'value'], rows}}
      controls={
        <>
          <label className={s.control}>
            lookups
            <select className={s.select} value={parallel ? 'parallel' : 'series'} onChange={(e) => setParallel(e.target.value === 'parallel')}>
              <option value="parallel">in parallel</option>
              <option value="series">in series</option>
            </select>
          </label>
          <label className={s.control}>
            number of lookups
            <input type="range" min={1} max={20} step={1} value={lookups} onChange={(e) => setLookups(Number(e.target.value))} />
            <span className={s.value}>{lookups}</span>
          </label>
          <label className={s.control}>
            lookup p99 (ms)
            <input type="range" min={10} max={80} step={5} value={lookupP99} onChange={(e) => setLookupP99(Number(e.target.value))} />
            <span className={s.value}>{lookupP99}</span>
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={hedge} onChange={(e) => setHedge(e.target.checked)} />
            hedge slow lookups
          </label>
          <label className={s.control}>
            fetch timeout
            <select className={s.select} value={timeout} onChange={(e) => setTimeoutMs(Number(e.target.value))}>
              {TIMEOUTS.map((t) => (
                <option key={t} value={t}>
                  {t === 0 ? 'none' : `${t} ms`}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            limit (ms)
            <input type="range" min={50} max={150} step={5} value={slo} onChange={(e) => setSlo(Number(e.target.value))} />
            <span className={s.value}>{slo}</span>
          </label>
          <span className={s.value} aria-live="polite">
            p50 {r.p50.toFixed(1)}, p99 {r.p99.toFixed(1)} ms, {(over * 100).toFixed(2)}% over {slo} ms
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Latency histogram: p50 ${r.p50.toFixed(1)} ms, p99 ${r.p99.toFixed(1)} ms, ${(over * 100).toFixed(2)} percent over ${slo} ms`}>
        <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} />
        {bins.map((c, i) => {
          const h = (c / peak) * innerH;
          return <rect key={i} x={PAD.left + i * barW + 0.5} y={PAD.top + innerH - h} width={Math.max(1, barW - 1)} height={h} fill={bar} opacity={0.85} />;
        })}
        {[0, 20, 40, 60, 80, 100, 120, 140].map((t) => (
          <text key={t} className={s.tick} x={x(t)} y={H - 22} textAnchor="middle">
            {t}
          </text>
        ))}
        <line x1={x(slo)} y1={PAD.top} x2={x(slo)} y2={PAD.top + innerH} stroke={accent} strokeWidth={2} strokeDasharray="5 3" />
        <line x1={x(Math.min(r.p99, MAX_MS))} y1={PAD.top} x2={x(Math.min(r.p99, MAX_MS))} y2={PAD.top + innerH} stroke="var(--text-strong)" strokeWidth={1.5} />
        <text className={s.dataLabel} x={x(Math.min(r.p99, MAX_MS)) + 4} y={PAD.top + 12}>
          p99 {r.p99.toFixed(1)}
        </text>
        <text className={s.axisLabel} x={W / 2} y={H - 4} textAnchor="middle">
          end-to-end latency (ms)
        </text>
      </svg>
    </VizPanel>
  );
}
