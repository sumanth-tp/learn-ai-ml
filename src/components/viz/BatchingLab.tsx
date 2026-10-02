import {useMemo, useState} from 'react';

import {BatchingResult, makeTrace, runContinuous, runStatic} from './inferenceMath';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 330;
const SHOWN = 40;
const POLICIES = [
  'static batching',
  'continuous',
  'continuous, chunked prefill 512',
  'continuous, chunked prefill 256',
  'continuous, shortest prompt first',
];
const SHORT = ['static', 'continuous', 'continuous + chunked 512', 'continuous + chunked 256', 'continuous + shortest first'];
const BATCHES = [16, 32, 64];

function run(policy: number, rate: number, maxBatch: number): BatchingResult {
  const trace = makeTrace(600, rate);
  if (policy === 0) return runStatic(trace, maxBatch);
  if (policy === 1) return runContinuous(trace, maxBatch);
  if (policy === 2) return runContinuous(trace, maxBatch, 512);
  if (policy === 3) return runContinuous(trace, maxBatch, 256);
  return runContinuous(trace, maxBatch, null, true);
}

const fixed = (v: number, d: number) => v.toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d});

export default function BatchingLab() {
  const dark = useDarkViz();
  const [policy, setPolicy] = useState(1);
  const [rate, setRate] = useState(30);
  const [maxBatch, setMaxBatch] = useState(64);

  const results = useMemo(() => POLICIES.map((_, p) => run(p, rate, maxBatch)), [rate, maxBatch]);
  const trace = useMemo(() => makeTrace(SHOWN, rate), [rate]);
  const chosen = results[policy];

  const wait = seriesColor(1, dark);
  const decode = seriesColor(0, dark);
  const muted = dark ? '#848c99' : '#9aa0a6';

  const first = trace.map((r) => r.id);
  const horizon = Math.max(...first.map((id) => chosen.finish[id]));
  const left = {x: 40, w: 340};
  const rowH = 7.2;
  const tx = (v: number) => left.x + (v / horizon) * left.w;
  const maxTokens = Math.max(...results.map((r) => r.tokensPerS));

  const rows = POLICIES.map((name, p) => {
    const r = results[p];
    return [
      name,
      Math.round(r.tokensPerS).toLocaleString('en-US'),
      fixed(r.latencyMeanS, 2),
      fixed(r.latencyP99S, 2),
      fixed(r.ttftMeanMs, 1),
      fixed(r.ttftP99Ms, 1),
      fixed(r.gapP99Ms, 2),
    ];
  });

  return (
    <VizPanel
      title="Static against continuous batching on one request trace"
      hint="Same 600 requests, same step-time model, five policies. Static batching holds every request until the longest in its batch is done, so latency explodes once the rate passes its capacity. Raise the rate to 60 or more to see queueing under continuous batching, and compare chunked prefill's effect on the inter-token gap at 58. Defaults match the chapter: 4,324 tokens per second, mean latency 0.91 s, p99 inter-token gap 10.88 ms."
      legend={[
        {label: 'waiting plus prefill, to first token', color: wait},
        {label: 'decoding, first token to finish', color: decode},
        {label: 'tokens per second, other policies', color: muted},
      ]}
      table={{
        columns: ['policy', 'tokens/s', 'latency mean s', 'latency p99 s', 'TTFT mean ms', 'TTFT p99 ms', 'gap p99 ms'],
        rows,
      }}
      controls={
        <>
          <label className={s.control}>
            policy
            <select className={s.select} value={policy} onChange={(e) => setPolicy(Number(e.target.value))}>
              {POLICIES.map((p, i) => (
                <option key={p} value={i}>{p}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            requests per second
            <input type="range" min={5} max={70} step={5} value={rate}
                   onChange={(e) => setRate(Number(e.target.value))} />
            <span className={s.value}>{rate}</span>
          </label>
          <label className={s.control}>
            max batch
            <select className={s.select} value={maxBatch} onChange={(e) => setMaxBatch(Number(e.target.value))}>
              {BATCHES.map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`${POLICIES[policy]} at ${rate} requests per second: ${Math.round(chosen.tokensPerS)} tokens per second, mean latency ${chosen.latencyMeanS.toFixed(2)} seconds, p99 time to first token ${chosen.ttftP99Ms.toFixed(1)} milliseconds.`}>
        <text className={s.axisLabel} x={left.x} y={14}>first 40 requests, arrival to finish</text>
        {trace.map((r, k) => {
          const y = 22 + k * rowH;
          const a = tx(r.arrival);
          const f = tx(chosen.firstToken[r.id]);
          const e = tx(chosen.finish[r.id]);
          return (
            <g key={r.id}>
              <rect x={a} y={y} width={Math.max(1, f - a)} height={5} fill={wait} />
              <rect x={f} y={y} width={Math.max(1, e - f)} height={5} fill={decode} opacity={0.75} />
            </g>
          );
        })}
        {[0, 0.25, 0.5, 0.75, 1].map((q) => (
          <text key={q} className={s.tick} x={tx(q * horizon)} y={H - 10} textAnchor="middle">
            {(q * horizon / 1000).toFixed(1)}
          </text>
        ))}
        <text className={s.axisLabel} x={left.x + left.w / 2} y={H - 0} textAnchor="middle">seconds</text>
        <text className={s.axisLabel} x={410} y={14}>tokens per second</text>
        {POLICIES.map((name, p) => {
          const y = 28 + p * 58;
          const width = Math.max(2, (results[p].tokensPerS / maxTokens) * 150);
          return (
            <g key={name}>
              <text className={s.tick} x={410} y={y + 8}>{SHORT[p]}</text>
              <rect x={410} y={y + 14} width={width} height={20} rx={3} fill={p === policy ? decode : muted}
                    opacity={p === policy ? 1 : 0.55} />
              <text className={s.dataLabel} x={410 + width + 6} y={y + 29}>
                {Math.round(results[p].tokensPerS).toLocaleString('en-US')}
              </text>
            </g>
          );
        })}
      </svg>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.4rem'}}>
        <span aria-live="polite">
          {POLICIES[policy]}: {Math.round(chosen.tokensPerS).toLocaleString('en-US')} tokens per second, mean latency{' '}
          {chosen.latencyMeanS.toFixed(2)} s, p99 {chosen.latencyP99S.toFixed(2)} s, time to first token mean{' '}
          {chosen.ttftMeanMs.toFixed(1)} ms and p99 {chosen.ttftP99Ms.toFixed(1)} ms, p99 inter-token gap{' '}
          {chosen.gapP99Ms.toFixed(2)} ms.
        </span>
      </div>
    </VizPanel>
  );
}
