import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const PAD = {top: 16, right: 40, bottom: 40, left: 72};

const PRICES = {bigIn: 3.0, bigOut: 15.0, smallIn: 0.3, smallOut: 1.5};
const CACHE_READ = 0.1;
const OUTPUT_TOKENS = 20;

export type Gap = 'knowledge' | 'format' | 'reasoning' | 'latency_cost';

export const GAPS: {value: Gap; label: string}[] = [
  {value: 'knowledge', label: 'Knowledge gap'},
  {value: 'format', label: 'Behaviour or format gap'},
  {value: 'reasoning', label: 'Reasoning gap'},
  {value: 'latency_cost', label: 'Latency or cost'},
];

type Option = {name: string; variable: number; fixed: number};

export function perRequest(
  promptTokens: number,
  outputTokens: number,
  priceIn: number,
  priceOut: number,
  cachedShare = 0,
): number {
  const effectiveIn = promptTokens * (1 - cachedShare) + promptTokens * cachedShare * CACHE_READ;
  return (effectiveIn * priceIn + outputTokens * priceOut) / 1_000_000;
}

export function options(promptTokens: number, cachedShare: number): Option[] {
  return [
    {
      name: 'long prompt',
      variable: perRequest(promptTokens, OUTPUT_TOKENS, PRICES.bigIn, PRICES.bigOut, cachedShare),
      fixed: 0,
    },
    {
      name: 'RAG',
      variable: perRequest(1000, OUTPUT_TOKENS, PRICES.bigIn, PRICES.bigOut),
      fixed: 600,
    },
    {
      name: 'tuned small model',
      variable: perRequest(60, OUTPUT_TOKENS, PRICES.smallIn, PRICES.smallOut),
      fixed: 3000 / 12 + 8 * 120,
    },
  ];
}

export function breakEven(a: Option, b: Option): number {
  return (b.fixed - a.fixed) / (a.variable - b.variable);
}

export function advise(gap: Gap, factsChange: boolean, labelled: number, requests: number, breakEvenVolume: number): string[] {
  const steps = ['write the prompt and a held-out eval set'];
  if (gap === 'knowledge') {
    steps.push('retrieval over the source documents (RAG)');
    if (factsChange) steps.push('keep the index fresh; do not train facts in');
    return steps;
  }
  if (gap === 'format') {
    steps.push('few-shot examples in the prompt');
    steps.push(
      labelled >= 500
        ? 'supervised fine-tuning with LoRA'
        : 'collect more labelled examples before tuning',
    );
    return steps;
  }
  if (gap === 'reasoning') {
    steps.push('a stronger model or step-by-step decomposition');
    if (labelled >= 5000) steps.push('fine-tune on verified reasoning traces');
    return steps;
  }
  steps.push('shorter prompt and prompt caching');
  steps.push(
    requests >= breakEvenVolume && labelled >= 500
      ? "tune a smaller model on the big model's outputs"
      : 'stay on the large model: volume is below break-even',
  );
  return steps;
}

const fmt = (v: number) => Math.round(v).toLocaleString('en-GB');

export default function AdaptationDecisionLab() {
  const dark = useDarkViz();
  const [gap, setGap] = useState<Gap>('format');
  const [factsChange, setFactsChange] = useState(false);
  const [labelled, setLabelled] = useState(2000);
  const [logRequests, setLogRequests] = useState(5);
  const [promptTokens, setPromptTokens] = useState(3000);
  const [cachedShare, setCachedShare] = useState(0);

  const requests = Math.round(10 ** logRequests);
  const opts = options(promptTokens, cachedShare);
  const [longPrompt, rag, tuned] = opts;
  const beLong = breakEven(longPrompt, tuned);
  const beRag = breakEven(rag, tuned);
  const steps = advise(gap, factsChange, labelled, requests, Math.round(beLong));

  const cost = (o: Option, volume: number) => o.variable * volume + o.fixed;

  const xMin = 3;
  const xMax = 7;
  const yMin = 1;
  const yMax = 5;
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const x = (lv: number) => PAD.left + ((lv - xMin) / (xMax - xMin)) * innerW;
  const y = (value: number) => {
    const lv = Math.log10(Math.max(value, 10 ** yMin));
    return PAD.top + innerH - ((Math.min(lv, yMax) - yMin) / (yMax - yMin)) * innerH;
  };

  const grid: number[] = [];
  for (let lv = xMin; lv <= xMax + 1e-9; lv += 0.05) grid.push(Number(lv.toFixed(2)));
  const path = (o: Option) =>
    grid.map((lv, i) => `${i ? 'L' : 'M'}${x(lv).toFixed(1)},${y(cost(o, 10 ** lv)).toFixed(1)}`).join(' ');

  const volumes = [10_000, 100_000, 1_000_000];
  const rows: (string | number)[][] = volumes.map((v) => [
    fmt(v),
    fmt(cost(longPrompt, v)),
    fmt(cost(rag, v)),
    fmt(cost(tuned, v)),
  ]);
  rows.push(['break-even long prompt vs tuned', fmt(beLong), '', '']);
  rows.push(['break-even RAG vs tuned', '', fmt(beRag), '']);

  const yTicks = [10, 100, 1000, 10000, 100000];
  const compact = (v: number) => (v >= 1_000_000 ? `${v / 1_000_000}M` : v >= 1000 ? `${v / 1000}k` : String(v));
  const xTicks = [3, 4, 5, 6, 7];

  return (
    <VizPanel
      title="Which lever, and when does tuning pay?"
      hint="Defaults: 100,000 requests a month and a 3,000-token prompt give 930, 930 and 1,215 cost units for the long prompt, RAG and the tuned small model, with a break-even of 130,783 requests. Set the cached share to 0.9 to see 201 and 616,718. Prices are synthetic."
      legend={opts.map((o, i) => ({label: o.name, color: seriesColor(i, dark)}))}
      table={{columns: ['requests per month', 'long prompt', 'RAG', 'tuned small model'], rows}}
      controls={
        <>
          <label className={s.control}>
            failure
            <select className={s.select} value={gap} onChange={(e) => setGap(e.target.value as Gap)}>
              {GAPS.map((g) => (
                <option key={g.value} value={g.value}>
                  {g.label}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            labelled examples
            <select className={s.select} value={labelled} onChange={(e) => setLabelled(Number(e.target.value))}>
              {[0, 120, 500, 2000, 5000].map((n) => (
                <option key={n} value={n}>
                  {n.toLocaleString('en-GB')}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={factsChange} onChange={(e) => setFactsChange(e.target.checked)} />
            facts change often
          </label>
          <label className={s.control}>
            requests per month
            <input
              type="range"
              min={3}
              max={7}
              step={0.05}
              value={logRequests}
              onChange={(e) => setLogRequests(Number(e.target.value))}
            />
            <span className={s.value}>{fmt(requests)}</span>
          </label>
          <label className={s.control}>
            long prompt tokens
            <input
              type="range"
              min={500}
              max={6000}
              step={100}
              value={promptTokens}
              onChange={(e) => setPromptTokens(Number(e.target.value))}
            />
            <span className={s.value}>{promptTokens}</span>
          </label>
          <label className={s.control}>
            cached share
            <input
              type="range"
              min={0}
              max={0.9}
              step={0.1}
              value={cachedShare}
              onChange={(e) => setCachedShare(Number(e.target.value))}
            />
            <span className={s.value}>{cachedShare.toFixed(1)}</span>
          </label>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Monthly cost against requests per month. At ${fmt(requests)} requests: long prompt ${fmt(cost(longPrompt, requests))}, RAG ${fmt(cost(rag, requests))}, tuned small model ${fmt(cost(tuned, requests))} units.`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line className={s.grid} x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} />
            <text className={s.tick} x={PAD.left - 8} y={y(t) + 3} textAnchor="end">
              {compact(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} className={s.tick} x={x(t)} y={H - 20} textAnchor="middle">
            {compact(10 ** t)}
          </text>
        ))}
        <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} />
        <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 4} textAnchor="middle">
          requests per month
        </text>
        <text
          className={s.axisLabel}
          x={12}
          y={PAD.top + innerH / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${PAD.top + innerH / 2})`}>
          cost units per month
        </text>
        {opts.map((o, i) => (
          <path key={o.name} d={path(o)} fill="none" stroke={seriesColor(i, dark)} strokeWidth={2.2} />
        ))}
        <line
          x1={x(logRequests)}
          x2={x(logRequests)}
          y1={PAD.top}
          y2={PAD.top + innerH}
          stroke="var(--text-strong)"
          strokeDasharray="4 3"
        />
        {opts.map((o, i) => (
          <circle
            key={`m${o.name}`}
            cx={x(logRequests)}
            cy={y(cost(o, requests))}
            r={4.5}
            fill={seriesColor(i, dark)}
            stroke="var(--surface-raised)"
            strokeWidth={1.5}
          />
        ))}
      </svg>
      <p className={s.value} style={{padding: '0.5rem 0 0'}} aria-live="polite">
        At {fmt(requests)} requests a month: long prompt {fmt(cost(longPrompt, requests))}, RAG{' '}
        {fmt(cost(rag, requests))}, tuned small model {fmt(cost(tuned, requests))} units. Break-even of the long prompt
        against the tuned model: {fmt(beLong)} requests.
      </p>
      <ol style={{margin: '0.5rem 0 0', paddingLeft: '1.4rem'}}>
        {steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ol>
    </VizPanel>
  );
}
