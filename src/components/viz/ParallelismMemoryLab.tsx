import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const GB = 1e9;
const VOCAB = 128000;

type Spec = {layers: number; h: number; heads: number; kv: number; ffn: number};

export const MODELS: Record<string, Spec> = {
  'Llama 3 8B': {layers: 32, h: 4096, heads: 32, kv: 8, ffn: 14336},
  'Llama 3 70B': {layers: 80, h: 8192, heads: 64, kv: 8, ffn: 28672},
  'Llama 3 405B': {layers: 126, h: 16384, heads: 128, kv: 8, ffn: 53248},
};

export type Mode = 'none' | 'tp' | 'tp+sp' | 'tp+sp+selective' | 'full';

export const MODES: {value: Mode; label: string}[] = [
  {value: 'none', label: 'no parallelism, nothing saved'},
  {value: 'tp', label: 'tensor parallel only'},
  {value: 'tp+sp', label: 'tensor + sequence parallel'},
  {value: 'tp+sp+selective', label: 'tensor + sequence + selective recompute'},
  {value: 'full', label: 'full recomputation'},
];

export function layerParams(m: Spec): number {
  const kvDim = (m.h * m.kv) / m.heads;
  return 2 * m.h * m.h + 2 * m.h * kvDim + 3 * m.h * m.ffn + 2 * m.h;
}

export function paramCount(m: Spec): number {
  return 2 * VOCAB * m.h + m.layers * layerParams(m) + m.h;
}

export type Config = {
  model: string;
  t: number;
  p: number;
  c: number;
  d: number;
  zero: number;
  seq: number;
  batch: number;
  mode: Mode;
};

export function memory(cfg: Config) {
  const m = MODELS[cfg.model];
  const P = paramCount(m);
  const shard = cfg.t * cfg.p;
  let weights = (2 * P) / shard;
  let grads = (2 * P) / shard;
  let opt = (12 * P) / shard;
  let gathered = 0;
  if (cfg.zero >= 1) opt /= cfg.d;
  if (cfg.zero >= 2) grads /= cfg.d;
  if (cfg.zero >= 3) {
    weights /= cfg.d;
    gathered = (2 * layerParams(m)) / cfg.t;
  }
  const sl = cfg.seq / cfg.c;
  const sbh = sl * cfg.batch * m.h;
  let perLayer: number;
  if (cfg.mode === 'none') perLayer = sbh * (34 + (5 * m.heads * sl) / m.h);
  else if (cfg.mode === 'tp') perLayer = sbh * (10 + 24 / cfg.t + (5 * m.heads * sl) / (m.h * cfg.t));
  else if (cfg.mode === 'tp+sp') perLayer = (sbh / cfg.t) * (34 + (5 * m.heads * sl) / m.h);
  else if (cfg.mode === 'tp+sp+selective') perLayer = (sbh * 34) / cfg.t;
  else perLayer = 2 * sbh;
  const acts = perLayer * m.layers;
  const w = weights + gathered;
  return {params: P, weights: w, grads, opt, acts, total: w + grads + opt + acts};
}

const W = 640;
const H = 150;
const LEFT = 20;
const RIGHT = 20;

const T_OPTIONS = [1, 2, 4, 8, 16];
const P_OPTIONS = [1, 2, 4, 8, 16, 32];
const C_OPTIONS = [1, 2, 4, 8, 16];
const D_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
const SEQ_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

export default function ParallelismMemoryLab() {
  const dark = useDarkViz();
  const [cfg, setCfg] = useState<Config>({
    model: 'Llama 3 405B',
    t: 8,
    p: 16,
    c: 1,
    d: 128,
    zero: 2,
    seq: 8192,
    batch: 1,
    mode: 'tp+sp+selective',
  });
  const [hbm, setHbm] = useState(80);

  const set = <K extends keyof Config>(key: K, value: Config[K]) => setCfg({...cfg, [key]: value});
  const mem = memory(cfg);
  const gpus = cfg.t * cfg.p * cfg.c * cfg.d;
  const parts = [
    {label: 'weights', value: mem.weights / GB, color: seriesColor(0, dark)},
    {label: 'gradients', value: mem.grads / GB, color: seriesColor(1, dark)},
    {label: 'optimiser state', value: mem.opt / GB, color: seriesColor(2, dark)},
    {label: 'activations', value: mem.acts / GB, color: seriesColor(3, dark)},
  ];
  const total = mem.total / GB;
  const scaleMax = Math.max(total, hbm) * 1.08;
  const x = (v: number) => LEFT + (v / scaleMax) * (W - LEFT - RIGHT);
  let cursor = 0;
  const fits = total <= hbm;
  const status = `${gpus.toLocaleString('en-GB')} GPUs, ${total.toFixed(1)} GB per GPU, ${
    fits ? 'fits in' : 'does not fit in'
  } ${hbm} GB`;

  const select = (label: string, value: number, options: number[], key: keyof Config) => (
    <label className={s.control}>
      {label}
      <select className={s.select} value={value} onChange={(e) => set(key, Number(e.target.value) as never)}>
        {options.map((o) => (
          <option key={o} value={o}>
            {o.toLocaleString('en-GB')}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <VizPanel
      title="Memory per GPU for a dense transformer"
      hint="The defaults are the Llama 3 405B layout from the paper (tensor 8, pipeline 16, data 128, ZeRO 2 style sharding, 8,192 tokens) and give 78.6 GB. Turn off selective recomputation to see why attention scores had to go."
      legend={parts.map((p) => ({label: p.label, color: p.color}))}
      table={{
        columns: ['component', 'GB per GPU'],
        rows: [
          ...parts.map((p) => [p.label, p.value.toFixed(2)]),
          ['total', total.toFixed(1)],
          ['GPUs (t x p x c x d)', gpus],
        ],
      }}
      controls={
        <>
          <label className={s.control}>
            model
            <select className={s.select} value={cfg.model} onChange={(e) => set('model', e.target.value)}>
              {Object.keys(MODELS).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          {select('tensor', cfg.t, T_OPTIONS, 't')}
          {select('pipeline', cfg.p, P_OPTIONS, 'p')}
          {select('context', cfg.c, C_OPTIONS, 'c')}
          {select('data', cfg.d, D_OPTIONS, 'd')}
          <label className={s.control}>
            ZeRO stage
            <select className={s.select} value={cfg.zero} onChange={(e) => set('zero', Number(e.target.value))}>
              {[0, 1, 2, 3].map((z) => (
                <option key={z} value={z}>
                  {z}
                </option>
              ))}
            </select>
          </label>
          {select('tokens per sequence', cfg.seq, SEQ_OPTIONS, 'seq')}
          {select('micro-batch', cfg.batch, [1, 2, 4], 'batch')}
          <label className={s.control}>
            activations
            <select className={s.select} value={cfg.mode} onChange={(e) => set('mode', e.target.value as Mode)}>
              {MODES.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            GPU memory (GB)
            <input type="range" min={16} max={192} step={1} value={hbm} onChange={(e) => setHbm(Number(e.target.value))} />
            <span className={s.value}>{hbm}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Stacked bar of memory per GPU: ${status}`}>
        <line className={s.axis} x1={LEFT} y1={100} x2={W - RIGHT} y2={100} />
        {parts.map((p) => {
          const x0 = x(cursor);
          cursor += p.value;
          return (
            <rect key={p.label} x={x0} y={46} width={Math.max(0, x(cursor) - x0)} height={40} fill={p.color} opacity={0.9} />
          );
        })}
        <line x1={x(hbm)} y1={26} x2={x(hbm)} y2={104} stroke={fits ? 'var(--text-strong)' : '#e34948'} strokeWidth={2.5} strokeDasharray="5 3" />
        <text className={s.dataLabel} x={Math.min(x(hbm), W - 90)} y={20} textAnchor="middle">
          {hbm} GB
        </text>
        <text className={s.tick} x={LEFT} y={120} textAnchor="start">
          0
        </text>
        <text className={s.tick} x={W - RIGHT} y={120} textAnchor="end">
          {scaleMax.toFixed(0)} GB
        </text>
        <text className={s.dataLabel} x={W / 2} y={142} textAnchor="middle">
          {`total ${total.toFixed(1)} GB per GPU on ${gpus.toLocaleString('en-GB')} GPUs`}
        </text>
      </svg>
    </VizPanel>
  );
}
