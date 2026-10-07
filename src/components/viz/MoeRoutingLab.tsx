import {useState} from 'react';

import {simulateRouting} from './moeMath';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 250;
const LEFT = 40;
const RIGHT = 16;
const TOP = 24;
const BOTTOM = 36;

const EXPERT_OPTIONS = [4, 8, 16, 32, 64];
const TOKEN_OPTIONS = [256, 512, 1024, 2048];

export default function MoeRoutingLab() {
  const dark = useDarkViz();
  const [experts, setExperts] = useState(16);
  const [topK, setTopK] = useState(2);
  const [tokens, setTokens] = useState(1024);
  const [cf, setCf] = useState(1.25);
  const [skew, setSkew] = useState(0.5);

  const k = Math.min(topK, experts);
  const r = simulateRouting(experts, k, tokens, cf, skew);
  const slots = tokens * k;
  const mean = slots / experts;
  const yMax = Math.max(...r.counts, r.capacity, mean) * 1.1;
  const innerW = W - LEFT - RIGHT;
  const innerH = H - TOP - BOTTOM;
  const bw = innerW / experts;
  const y = (v: number) => TOP + innerH - (v / yMax) * innerH;
  const status = `${experts} experts, top ${k}: balance term ${r.balance.toFixed(4)}, busiest expert ${r.busiest.toFixed(2)} times the average, ${(
    r.dropped * 100
  ).toFixed(1)} per cent of slots dropped, ${(r.used * 100).toFixed(1)} per cent of capacity used`;

  return (
    <VizPanel
      title="Router balance, capacity and dropped tokens"
      hint="The defaults, 16 experts, top 2, 1,024 tokens, capacity factor 1.25 and skew 0.5, give the row printed by block 1: balance term 1.5836, busiest expert 3.72 times the average, 28.5 per cent dropped, 57.2 per cent of capacity used."
      legend={[
        {label: 'slots sent to the expert', color: seriesColor(0, dark)},
        {label: 'slots beyond capacity (dropped)', color: seriesColor(1, dark)},
      ]}
      table={{
        columns: ['expert', 'slots received', 'kept', 'dropped'],
        rows: [
          ...r.counts.map((c, i) => [i, c, Math.min(c, r.capacity), Math.max(0, c - r.capacity)]),
          ['capacity per expert', r.capacity, '', ''],
          ['balance term', r.balance.toFixed(4), '', ''],
        ],
      }}
      controls={
        <>
          <label className={s.control}>
            experts
            <select className={s.select} value={experts} onChange={(e) => setExperts(Number(e.target.value))}>
              {EXPERT_OPTIONS.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            experts per token (top-k)
            <select className={s.select} value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
              {[1, 2, 3, 4].map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            tokens
            <select className={s.select} value={tokens} onChange={(e) => setTokens(Number(e.target.value))}>
              {TOKEN_OPTIONS.map((o) => (
                <option key={o} value={o}>
                  {o.toLocaleString('en-GB')}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            capacity factor
            <input type="range" min={1} max={3} step={0.05} value={cf} onChange={(e) => setCf(Number(e.target.value))} />
            <span className={s.value}>{cf.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            router skew
            <input type="range" min={0} max={1.5} step={0.05} value={skew} onChange={(e) => setSkew(Number(e.target.value))} />
            <span className={s.value}>{skew.toFixed(2)}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Bars of slots per expert against a capacity line. ${status}`}>
        <line className={s.axis} x1={LEFT} y1={TOP + innerH} x2={W - RIGHT} y2={TOP + innerH} />
        {r.counts.map((c, i) => {
          const kept = Math.min(c, r.capacity);
          const x = LEFT + i * bw + bw * 0.12;
          const w = bw * 0.76;
          return (
            <g key={i}>
              <rect x={x} y={y(kept)} width={w} height={TOP + innerH - y(kept)} fill={seriesColor(0, dark)} />
              {c > kept && <rect x={x} y={y(c)} width={w} height={y(kept) - y(c)} fill={seriesColor(1, dark)} />}
              {experts <= 16 && (
                <text className={s.tick} x={x + w / 2} y={H - 18} textAnchor="middle">
                  {i}
                </text>
              )}
            </g>
          );
        })}
        <line x1={LEFT} y1={y(r.capacity)} x2={W - RIGHT} y2={y(r.capacity)} stroke="var(--text-strong)" strokeWidth={2} strokeDasharray="6 4" />
        <line x1={LEFT} y1={y(mean)} x2={W - RIGHT} y2={y(mean)} stroke="#868e96" strokeWidth={1.5} strokeDasharray="2 4" />
        <text className={s.tick} x={W - RIGHT} y={y(r.capacity) - 4} textAnchor="end">
          capacity {r.capacity}
        </text>
        <text className={s.tick} x={W - RIGHT} y={y(mean) + 12} textAnchor="end">
          even share {mean.toFixed(1)}
        </text>
        <text className={s.dataLabel} x={W / 2} y={H - 2} textAnchor="middle">
          {`balance ${r.balance.toFixed(4)}, dropped ${(r.dropped * 100).toFixed(1)}%, capacity used ${(r.used * 100).toFixed(1)}%`}
        </text>
      </svg>
    </VizPanel>
  );
}
