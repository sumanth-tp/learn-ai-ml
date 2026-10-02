import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const WRITE_READ = 0.1;

export const TAUS = [0.55, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95];
export const HIT = [0.7548, 0.6278, 0.443, 0.3284, 0.2235, 0.0896, 0];
export const WRONG = [0.2115, 0.1296, 0.0972, 0.0547, 0.0612, 0.0208, 0];
const EMBED = 0.02;

export const CASCADE_T = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8];
export const ESCALATED = [0.171, 0.258, 0.288, 0.353, 0.482, 0.656];
export const CASCADE_ACC = [0.877, 0.954, 0.961, 0.957, 0.953, 0.949];
export const CASCADE_COST = [3.56, 4.87, 5.33, 6.3, 8.23, 10.85];
const SMALL = {acc: 0.734, cost: 1};
const LARGE = {acc: 0.945, cost: 15};

type Mode = 'prompt' | 'semantic' | 'cascade';

const W = 640;
const H = 280;

export function promptCost(requests: number, write: number): number {
  return write + (requests - 1) * WRITE_READ;
}

export function semanticCost(index: number, wrongCost: number): number {
  return EMBED + (1 - HIT[index]) + WRONG[index] * wrongCost;
}

export default function CacheRoutingLab() {
  const dark = useDarkViz();
  const [mode, setMode] = useState<Mode>('semantic');
  const [requests, setRequests] = useState(5);
  const [write, setWrite] = useState(1.25);
  const [tau, setTau] = useState(3);
  const [wrongCost, setWrongCost] = useState(10);
  const [casc, setCasc] = useState(2);

  const c0 = seriesColor(0, dark);
  const c1 = seriesColor(1, dark);
  const c2 = seriesColor(2, dark);
  const neg = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;
  const mid = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;

  const status =
    mode === 'prompt'
      ? `${requests} requests share the prefix: uncached ${requests.toFixed(2)}, cached ${promptCost(requests, write).toFixed(2)} (units of one uncached prefix)`
      : mode === 'semantic'
        ? `threshold ${TAUS[tau].toFixed(2)}: hit rate ${HIT[tau].toFixed(3)}, wrong answers per request ${WRONG[tau].toFixed(4)}, cost ${semanticCost(tau, wrongCost).toFixed(2)} (no cache 1.00)`
        : `confidence ${CASCADE_T[casc].toFixed(1)}: escalated ${ESCALATED[casc].toFixed(3)}, accuracy ${CASCADE_ACC[casc].toFixed(3)}, cost ${CASCADE_COST[casc].toFixed(2)} (large alone ${LARGE.acc.toFixed(3)} at ${LARGE.cost.toFixed(2)})`;

  const table =
    mode === 'prompt'
      ? {
          columns: ['requests', 'uncached', 'cached'],
          rows: [1, 2, 3, 5, 10, 50].map((n) => [n, n.toFixed(2), promptCost(n, write).toFixed(2)]),
        }
      : mode === 'semantic'
        ? {
            columns: ['threshold', 'hit rate', 'wrong per request', `cost, wrong = ${wrongCost}`],
            rows: TAUS.map((t, i) => [t.toFixed(2), HIT[i].toFixed(4), WRONG[i].toFixed(4), semanticCost(i, wrongCost).toFixed(2)]),
          }
        : {
            columns: ['confidence', 'escalated', 'accuracy', 'cost'],
            rows: CASCADE_T.map((t, i) => [t.toFixed(1), ESCALATED[i].toFixed(3), CASCADE_ACC[i].toFixed(3), CASCADE_COST[i].toFixed(2)]),
          };

  const barH = (value: number, max: number, height: number) => (value / max) * height;
  const base = 230;

  return (
    <VizPanel
      title="Caching and routing, three cost levers"
      hint="Prompt cache: the write premium is repaid from the second request. Semantic cache: raise the cost of a wrong answer and the cheapest threshold climbs until no cache wins (the measured table is the chapter's 48-query MiniLM experiment). Cascade: a confidence gate reaches the large model's accuracy at about a third of its cost, but only because this simulation's confidence is informative. Defaults: semantic, threshold 0.80, wrong answer costing 10, where the cost is about 1.24 (the chapter's code prints 1.238 from unrounded values)."
      legend={[
        {label: mode === 'prompt' ? 'uncached' : mode === 'semantic' ? 'hit rate' : 'cost', color: c0},
        {label: mode === 'prompt' ? 'cached' : mode === 'semantic' ? 'wrong answers per request' : 'accuracy', color: mode === 'semantic' ? neg : c1},
        {label: 'selected', color: c2},
      ]}
      table={table}
      controls={
        <>
          <label className={s.control}>
            lever
            <select className={s.select} value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              <option value="prompt">prompt cache</option>
              <option value="semantic">semantic cache threshold</option>
              <option value="cascade">two-model cascade</option>
            </select>
          </label>
          {mode === 'prompt' && (
            <>
              <label className={s.control}>
                requests reusing the prefix
                <input type="range" min={1} max={50} step={1} value={requests} onChange={(e) => setRequests(Number(e.target.value))} />
                <span className={s.value}>{requests}</span>
              </label>
              <label className={s.control}>
                write multiplier
                <select className={s.select} value={write} onChange={(e) => setWrite(Number(e.target.value))}>
                  <option value={1.25}>1.25 (5-minute)</option>
                  <option value={2}>2.00 (1-hour)</option>
                </select>
              </label>
            </>
          )}
          {mode === 'semantic' && (
            <>
              <label className={s.control}>
                threshold
                <input type="range" min={0} max={TAUS.length - 1} step={1} value={tau} onChange={(e) => setTau(Number(e.target.value))} />
                <span className={s.value}>{TAUS[tau].toFixed(2)}</span>
              </label>
              <label className={s.control}>
                cost of a wrong answer
                <select className={s.select} value={wrongCost} onChange={(e) => setWrongCost(Number(e.target.value))}>
                  {[1, 2, 5, 10, 50].map((c) => (
                    <option key={c} value={c}>
                      {c} LLM calls
                    </option>
                  ))}
                </select>
              </label>
            </>
          )}
          {mode === 'cascade' && (
            <label className={s.control}>
              confidence threshold
              <input type="range" min={0} max={CASCADE_T.length - 1} step={1} value={casc} onChange={(e) => setCasc(Number(e.target.value))} />
              <span className={s.value}>{CASCADE_T[casc].toFixed(1)}</span>
            </label>
          )}
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={status}>
        <line className={s.axis} x1={40} y1={base} x2={W - 20} y2={base} />
        {mode === 'prompt' && (
          <g>
            {[
              {label: 'uncached', value: requests, colour: c0},
              {label: 'cached', value: promptCost(requests, write), colour: c2},
            ].map((b, i) => {
              const h = barH(b.value, Math.max(requests, 2.2), 190);
              return (
                <g key={b.label}>
                  <rect x={150 + i * 200} y={base - h} width={110} height={h} rx={4} fill={b.colour} />
                  <text className={s.dataLabel} x={205 + i * 200} y={base - h - 8} textAnchor="middle">
                    {b.value.toFixed(2)}
                  </text>
                  <text className={s.tick} x={205 + i * 200} y={base + 18} textAnchor="middle">
                    {b.label}
                  </text>
                </g>
              );
            })}
          </g>
        )}
        {mode === 'semantic' && (
          <g>
            {TAUS.map((t, i) => {
              const x = 60 + i * 80;
              const hh = barH(HIT[i], 0.8, 170);
              const wh = barH(WRONG[i], 0.25, 170);
              return (
                <g key={t}>
                  <rect x={x} y={base - hh} width={26} height={hh} rx={3} fill={i === tau ? c2 : c0} />
                  <rect x={x + 30} y={base - wh} width={26} height={wh} rx={3} fill={neg} />
                  <text className={s.tick} x={x + 28} y={base + 16} textAnchor="middle">
                    {t.toFixed(2)}
                  </text>
                  <text className={s.tick} x={x + 28} y={base + 32} textAnchor="middle" fontWeight={i === tau ? 700 : 400}>
                    {semanticCost(i, wrongCost).toFixed(2)}
                  </text>
                </g>
              );
            })}
            <text className={s.tick} x={10} y={base + 32}>
              cost
            </text>
          </g>
        )}
        {mode === 'cascade' && (
          <g>
            {CASCADE_T.map((t, i) => {
              const x = 70 + i * 85;
              const ch = barH(CASCADE_COST[i], 15, 170);
              const ah = barH(CASCADE_ACC[i], 1, 170);
              return (
                <g key={t}>
                  <rect x={x} y={base - ch} width={26} height={ch} rx={3} fill={i === casc ? c2 : c0} />
                  <rect x={x + 30} y={base - ah} width={26} height={ah} rx={3} fill={c1} opacity={0.85} />
                  <text className={s.tick} x={x + 28} y={base + 16} textAnchor="middle" fontWeight={i === casc ? 700 : 400}>
                    {t.toFixed(1)}
                  </text>
                </g>
              );
            })}
            <line x1={40} y1={base - barH(LARGE.cost, 15, 170)} x2={W - 20} y2={base - barH(LARGE.cost, 15, 170)} stroke={mid} strokeDasharray="4 4" />
            <text className={s.tick} x={W - 22} y={base - barH(LARGE.cost, 15, 170) - 5} textAnchor="end">
              large alone: cost 15.00, accuracy 0.945
            </text>
          </g>
        )}
      </svg>
    </VizPanel>
  );
}
