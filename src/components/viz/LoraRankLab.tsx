import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 250;

const HIDDEN = 576;
const INTER = 1536;
const KV = 192;
const LAYERS = 30;
export const BASE_PARAMS = 134_515_008;

const SHAPES: Record<string, [number, number]> = {
  q_proj: [HIDDEN, HIDDEN],
  k_proj: [HIDDEN, KV],
  v_proj: [HIDDEN, KV],
  o_proj: [HIDDEN, HIDDEN],
  gate_proj: [HIDDEN, INTER],
  up_proj: [HIDDEN, INTER],
  down_proj: [INTER, HIDDEN],
};

export const TARGETS: {value: string; label: string; modules: string[]}[] = [
  {value: 'qv', label: 'q, v', modules: ['q_proj', 'v_proj']},
  {value: 'attn', label: 'q, k, v, o', modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']},
  {value: 'all', label: 'all linear', modules: Object.keys(SHAPES)},
];

export const RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const MB = 1_000_000;

export function loraParams(rank: number, modules: string[]): number {
  return LAYERS * modules.reduce((sum, m) => sum + rank * (SHAPES[m][0] + SHAPES[m][1]), 0);
}

const fmt = (n: number) => n.toLocaleString('en-GB');

export default function LoraRankLab() {
  const dark = useDarkViz();
  const [rank, setRank] = useState(8);
  const [target, setTarget] = useState('all');
  const [alpha, setAlpha] = useState(16);
  const [rs, setRs] = useState(false);
  const [bytes, setBytes] = useState(4);

  const modules = TARGETS.find((t) => t.value === target)!.modules;
  const trainable = loraParams(rank, modules);
  const scaling = rs ? alpha / Math.sqrt(rank) : alpha / rank;

  const fullWeights = BASE_PARAMS * 4;
  const fullGrads = BASE_PARAMS * 4;
  const fullMoments = BASE_PARAMS * 8;
  const fullTotal = fullWeights + fullGrads + fullMoments;
  const loraWeights = BASE_PARAMS * bytes + trainable * 4;
  const loraGrads = trainable * 4;
  const loraMoments = trainable * 8;
  const loraTotal = loraWeights + loraGrads + loraMoments;

  const colours = [seriesColor(0, dark), seriesColor(1, dark), seriesColor(2, dark)];
  const barX = 150;
  const barW = W - barX - 24;
  const scale = (barW - 70) / fullTotal;

  const stack = (y: number, parts: number[], label: string) => {
    let x = barX;
    return (
      <g>
        <text className={s.axisLabel} x={barX - 8} y={y + 17} textAnchor="end">
          {label}
        </text>
        {parts.map((p, i) => {
          const w = Math.max(p * scale, p > 0 ? 1 : 0);
          const rect = <rect key={i} x={x} y={y} width={w} height={26} fill={colours[i]} />;
          x += w;
          return rect;
        })}
        <text className={s.dataLabel} x={x + 6} y={y + 17}>
          {Math.round(parts.reduce((a, b) => a + b, 0) / MB)} MB
        </text>
      </g>
    );
  };

  const pct = (trainable / BASE_PARAMS) * 100;
  const status = `Rank ${rank} on ${TARGETS.find((t) => t.value === target)!.label}: ${fmt(trainable)} trainable parameters, ${pct.toFixed(2)}% of the ${fmt(BASE_PARAMS)} base. Scaling ${scaling.toFixed(2)}.`;

  const rows = RANKS.map((r) => [r, fmt(loraParams(r, modules)), `${((loraParams(r, modules) / BASE_PARAMS) * 100).toFixed(2)}%`]);

  return (
    <VizPanel
      title="LoRA: trainable parameters and memory against rank"
      hint="Default: rank 8 on all linear layers gives 2,442,240 trainable parameters (1.82% of the base), adapter weights 9.8 MB, gradients 9.8 MB and Adam moments 19.5 MB, against 2,152 MB of training state for full fine-tuning in float32. Memory figures are estimates and leave out activations."
      legend={[
        {label: 'weights', color: colours[0]},
        {label: 'gradients', color: colours[1]},
        {label: 'Adam moments', color: colours[2]},
      ]}
      table={{columns: ['rank', 'trainable parameters', 'share of base'], rows}}
      controls={
        <>
          <label className={s.control}>
            rank
            <select className={s.select} value={rank} onChange={(e) => setRank(Number(e.target.value))}>
              {RANKS.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            target modules
            <select className={s.select} value={target} onChange={(e) => setTarget(e.target.value)}>
              {TARGETS.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            lora_alpha
            <input type="range" min={1} max={256} step={1} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
            <span className={s.value}>{alpha}</span>
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={rs} onChange={(e) => setRs(e.target.checked)} />
            rsLoRA scaling
          </label>
          <label className={s.control}>
            frozen base
            <select className={s.select} value={bytes} onChange={(e) => setBytes(Number(e.target.value))}>
              <option value={4}>float32</option>
              <option value={2}>bfloat16</option>
            </select>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={status}>
        <text className={s.axisLabel} x={barX - 8} y={30} textAnchor="end">
          trainable
        </text>
        <rect x={barX} y={14} width={barW} height={24} fill="none" stroke="var(--border-strong)" />
        <rect x={barX} y={14} width={Math.max((trainable / BASE_PARAMS) * barW, 2)} height={24} fill={seriesColor(3, dark)} />
        <text className={s.dataLabel} x={barX + 6} y={56}>
          {fmt(trainable)} parameters, {pct.toFixed(2)}% of {fmt(BASE_PARAMS)}
        </text>
        {stack(84, [fullWeights, fullGrads, fullMoments], 'full fine-tuning')}
        {stack(130, [loraWeights, loraGrads, loraMoments], 'LoRA')}
        <text className={s.dataLabel} x={barX} y={188}>
          LoRA training state is {Math.round((loraTotal / fullTotal) * 100)}% of full fine-tuning
        </text>
        <text className={s.dataLabel} x={barX} y={210}>
          adapter: weights {(trainable * 4 / MB).toFixed(1)} MB, gradients {(trainable * 4 / MB).toFixed(1)} MB, moments{' '}
          {(trainable * 8 / MB).toFixed(1)} MB
        </text>
        <text className={s.dataLabel} x={barX} y={232}>
          scaling {rs ? 'alpha / sqrt(r)' : 'alpha / r'} = {scaling.toFixed(2)}
        </text>
      </svg>
    </VizPanel>
  );
}
