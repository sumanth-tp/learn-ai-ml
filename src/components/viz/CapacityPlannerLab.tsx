import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export const MODELS = {
  '8B': {label: 'Llama 3.1 8B', params: 8030261248, layers: 32, kvHeads: 8, headDim: 128},
  '70B': {label: 'Llama 3.1 70B', params: 70553706496, layers: 80, kvHeads: 8, headDim: 128},
} as const;

export const GPUS = {
  l4: {label: 'L4 24GB', memoryGb: 24, bandwidthGbs: 300, tflops: 121},
  a100: {label: 'A100 80GB SXM', memoryGb: 80, bandwidthGbs: 2039, tflops: 312},
  h100: {label: 'H100 SXM 80GB', memoryGb: 80, bandwidthGbs: 3350, tflops: 989.5},
  nvl: {label: 'H100 NVL 94GB', memoryGb: 94, bandwidthGbs: 3900, tflops: 835.5},
} as const;

type ModelKey = keyof typeof MODELS;
type GpuKey = keyof typeof GPUS;

export type Plan =
  | {ok: false; reason: string; poolGb: number; weightsGb: number}
  | {
      ok: true;
      weightsGb: number;
      kvKb: number;
      kvGb: number;
      poolGb: number;
      seqsMemory: number;
      seqsSlo: number;
      batch: number;
      stepMs: number;
      ttft: number;
      inFlight: number;
      decode: number;
      prefill: number;
      replicas: number;
      gpus: number;
      gpuHoursPerMtok: number;
    };

export type Inputs = {
  model: ModelKey;
  gpu: GpuKey;
  tp: number;
  qps: number;
  prompt: number;
  output: number;
  sloMs: number;
  efficiency: number;
  mfu: number;
  share: number;
};

export function plan(i: Inputs): Plan {
  const m = MODELS[i.model];
  const g = GPUS[i.gpu];
  const weights = m.params * 2;
  const kvPerToken = 2 * m.layers * m.kvHeads * m.headDim * 2;
  const pool = i.tp * (g.memoryGb * 0.9 - 2) * 1e9;
  const kvBudget = pool - weights;
  if (kvBudget <= 0) return {ok: false, reason: 'weights do not fit', poolGb: pool / 1e9, weightsGb: weights / 1e9};
  const ctx = i.prompt + i.output / 2;
  const seqsMemory = Math.floor(kvBudget / (kvPerToken * (i.prompt + i.output)));
  const bw = i.tp * g.bandwidthGbs * 1e9 * i.efficiency;
  const step = (b: number) => (weights + b * ctx * kvPerToken) / bw;
  let seqsSlo = 0;
  while (seqsSlo < seqsMemory && step(seqsSlo + 1) <= i.sloMs / 1000) seqsSlo += 1;
  const batch = Math.min(seqsMemory, seqsSlo);
  if (batch === 0) return {ok: false, reason: 'TPOT SLO unreachable', poolGb: pool / 1e9, weightsGb: weights / 1e9};
  const prefillRate = (i.tp * g.tflops * 1e12 * i.mfu) / (2 * m.params);
  const ttft = i.prompt / prefillRate;
  const latency = ttft + i.output * step(batch);
  const inFlight = i.qps * latency;
  const decode = Math.ceil(inFlight / batch);
  const prefill = Math.ceil((i.qps * i.prompt) / (prefillRate * i.share));
  const replicas = Math.max(decode, prefill);
  return {
    ok: true,
    weightsGb: weights / 1e9,
    kvKb: kvPerToken / 1e3,
    kvGb: kvBudget / 1e9,
    poolGb: pool / 1e9,
    seqsMemory,
    seqsSlo,
    batch,
    stepMs: step(batch) * 1000,
    ttft,
    inFlight,
    decode,
    prefill,
    replicas,
    gpus: replicas * i.tp,
    gpuHoursPerMtok: ((replicas * i.tp) / (i.qps * i.output * 3600)) * 1e6,
  };
}

const W = 640;
const H = 250;

export default function CapacityPlannerLab() {
  const dark = useDarkViz();
  const [model, setModel] = useState<ModelKey>('8B');
  const [gpu, setGpu] = useState<GpuKey>('h100');
  const [tp, setTp] = useState(1);
  const [qps, setQps] = useState(20);
  const [prompt, setPrompt] = useState(1000);
  const [output, setOutput] = useState(300);
  const [sloMs, setSloMs] = useState(40);
  const [efficiency, setEfficiency] = useState(0.6);
  const [mfu, setMfu] = useState(0.4);
  const [share, setShare] = useState(0.3);

  const p = useMemo(
    () => plan({model, gpu, tp, qps, prompt, output, sloMs, efficiency, mfu, share}),
    [model, gpu, tp, qps, prompt, output, sloMs, efficiency, mfu, share],
  );

  const c0 = seriesColor(0, dark);
  const c1 = seriesColor(1, dark);
  const c2 = seriesColor(2, dark);
  const mid = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;
  const neg = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;

  const status = p.ok
    ? `${p.gpus} GPUs: decode needs ${p.decode} replica${p.decode === 1 ? '' : 's'}, prefill needs ${p.prefill}; ${p.gpuHoursPerMtok.toFixed(3)} GPU-hours per million output tokens`
    : `no plan: ${p.reason}`;

  const rows: (string | number)[][] = p.ok
    ? [
        ['weights (GB)', p.weightsGb.toFixed(1)],
        ['KV per token (kB)', p.kvKb.toFixed(1)],
        ['memory pool (GB)', p.poolGb.toFixed(1)],
        ['KV budget (GB)', p.kvGb.toFixed(1)],
        ['sequences by memory', p.seqsMemory],
        ['sequences within the TPOT SLO', p.seqsSlo],
        ['decode step (ms)', p.stepMs.toFixed(1)],
        ['unloaded TTFT (s)', p.ttft.toFixed(2)],
        ['requests in flight (Little)', p.inFlight.toFixed(1)],
        ['replicas for decode', p.decode],
        ['replicas for prefill', p.prefill],
        ['GPUs', p.gpus],
        ['GPU-hours per million output tokens', p.gpuHoursPerMtok.toFixed(3)],
      ]
    : [['result', p.reason]];

  const barW = 520;
  const poolTotal = p.poolGb || 1;
  const kvShare = p.ok ? p.kvGb / poolTotal : 0;
  const wShare = Math.min(1, p.weightsGb / poolTotal);
  const maxRep = p.ok ? Math.max(p.decode, p.prefill, 1) : 1;

  return (
    <VizPanel
      title="GPU capacity planner"
      hint="Memory first (weights, then what is left for the KV cache), then Little's law for decode and a compute estimate for prefill; the larger replica count binds. Hardware figures are datasheet values fetched 2026-10-02, the efficiency, MFU and prefill share are assumptions you should replace with measurements. Defaults (8B, H100 SXM, 20 QPS, 1000 + 300 tokens, 40 ms) give 316 sequences by memory, a 31.7 ms step, 1 decode replica, 3 prefill replicas, 3 GPUs and 0.139 GPU-hours per million output tokens, as the chapter's code prints."
      legend={[
        {label: 'weights', color: c0},
        {label: 'KV cache budget', color: c2},
        {label: 'reserve and unused', color: mid},
        {label: 'replicas needed', color: c1},
      ]}
      table={{columns: ['quantity', 'value'], rows}}
      controls={
        <>
          <label className={s.control}>
            model
            <select className={s.select} value={model} onChange={(e) => setModel(e.target.value as ModelKey)}>
              {Object.entries(MODELS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.label}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            GPU
            <select className={s.select} value={gpu} onChange={(e) => setGpu(e.target.value as GpuKey)}>
              {Object.entries(GPUS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.label}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            tensor parallel
            <select className={s.select} value={tp} onChange={(e) => setTp(Number(e.target.value))}>
              {[1, 2, 4, 8].map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            QPS
            <input type="range" min={1} max={100} step={1} value={qps} onChange={(e) => setQps(Number(e.target.value))} />
            <span className={s.value}>{qps}</span>
          </label>
          <label className={s.control}>
            prompt tokens
            <input type="range" min={100} max={8000} step={100} value={prompt} onChange={(e) => setPrompt(Number(e.target.value))} />
            <span className={s.value}>{prompt}</span>
          </label>
          <label className={s.control}>
            output tokens
            <input type="range" min={50} max={1000} step={50} value={output} onChange={(e) => setOutput(Number(e.target.value))} />
            <span className={s.value}>{output}</span>
          </label>
          <label className={s.control}>
            TPOT SLO (ms)
            <input type="range" min={10} max={100} step={1} value={sloMs} onChange={(e) => setSloMs(Number(e.target.value))} />
            <span className={s.value}>{sloMs}</span>
          </label>
          <label className={s.control}>
            bandwidth efficiency
            <input type="range" min={0.3} max={0.9} step={0.05} value={efficiency} onChange={(e) => setEfficiency(Number(e.target.value))} />
            <span className={s.value}>{efficiency.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            prefill MFU
            <input type="range" min={0.2} max={0.6} step={0.05} value={mfu} onChange={(e) => setMfu(Number(e.target.value))} />
            <span className={s.value}>{mfu.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            prefill share of a replica
            <input type="range" min={0.1} max={0.8} step={0.1} value={share} onChange={(e) => setShare(Number(e.target.value))} />
            <span className={s.value}>{share.toFixed(1)}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={status}>
        <text className={s.tick} x={60} y={24}>
          memory pool, {p.poolGb.toFixed(1)} GB across {tp} GPU{tp === 1 ? '' : 's'} at 90% utilisation, 2 GB reserved each
        </text>
        <rect x={60} y={34} width={barW} height={30} rx={4} fill={mid} opacity={0.35} />
        <rect x={60} y={34} width={barW * wShare} height={30} rx={4} fill={c0} />
        {p.ok && <rect x={60 + barW * wShare} y={34} width={barW * kvShare} height={30} fill={c2} />}
        <text className={s.dataLabel} x={64} y={54} fill="#fff">
          weights {p.weightsGb.toFixed(1)} GB
        </text>
        {p.ok && (
          <text className={s.tick} x={60 + barW * wShare + 6} y={80}>
            KV budget {p.kvGb.toFixed(1)} GB = {p.seqsMemory} sequences of {prompt + output} tokens
          </text>
        )}
        {!p.ok && (
          <text className={s.dataLabel} x={60} y={92} fill={neg}>
            {p.reason}
          </text>
        )}
        {p.ok && (
          <g>
            {[
              {label: 'decode (Little)', value: p.decode},
              {label: 'prefill (compute)', value: p.prefill},
            ].map((b, i) => (
              <g key={b.label}>
                <text className={s.tick} x={160} y={140 + i * 40} textAnchor="end">
                  {b.label}
                </text>
                <rect x={168} y={126 + i * 40} width={(b.value / maxRep) * 380} height={22} rx={3} fill={c1} opacity={b.value === p.replicas ? 1 : 0.5} />
                <text className={s.dataLabel} x={176 + (b.value / maxRep) * 380} y={142 + i * 40}>
                  {b.value} replica{b.value === 1 ? '' : 's'}
                </text>
              </g>
            ))}
            <text className={s.tick} x={60} y={H - 22}>
              binding side: {p.decode >= p.prefill ? 'decode' : 'prefill'} | {p.gpus} GPUs ({p.replicas} replicas x {tp}) | step {p.stepMs.toFixed(1)} ms at batch {p.batch}
            </text>
          </g>
        )}
      </svg>
    </VizPanel>
  );
}
