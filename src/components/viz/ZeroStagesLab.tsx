import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const GB = 1e9;

export const MODELS: {label: string; params: number}[] = [
  {label: '7.5B (ZeRO paper example)', params: 7.5e9},
  {label: 'Llama 3.1 8B (8.03B)', params: 8030261248},
  {label: 'Llama 3.1 70B (70.55B)', params: 70553706496},
  {label: 'Llama 3.1 405B (405.85B)', params: 405853388800},
];

export const RECIPES: {label: string; w: number; g: number; o: number}[] = [
  {label: 'bf16 weights and gradients, fp32 Adam state (2 + 2 + 12)', w: 2, g: 2, o: 12},
  {label: 'fp32 everywhere (4 + 4 + 8)', w: 4, g: 4, o: 8},
];

export const TRAFFIC = [2, 2, 2, 3];

export function perGpu(params: number, n: number, stage: number, recipe: number) {
  const r = RECIPES[recipe];
  const w = (stage >= 3 ? r.w / n : r.w) * params;
  const g = (stage >= 2 ? r.g / n : r.g) * params;
  const o = (stage >= 1 ? r.o / n : r.o) * params;
  return {w: w / GB, g: g / GB, o: o / GB, total: (w + g + o) / GB};
}

export function smallestFit(params: number, stage: number, recipe: number, hbm: number): number | null {
  for (let n = 1; n <= 4096; n++) {
    if (perGpu(params, n, stage, recipe).total <= hbm) return n;
  }
  return null;
}

const W = 640;
const H = 250;
const LEFT = 78;
const RIGHT = 70;
const D_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

export default function ZeroStagesLab() {
  const dark = useDarkViz();
  const [model, setModel] = useState(0);
  const [n, setN] = useState(64);
  const [stage, setStage] = useState(1);
  const [recipe, setRecipe] = useState(0);
  const [hbm, setHbm] = useState(80);

  const params = MODELS[model].params;
  const stages = [0, 1, 2, 3].map((st) => ({stage: st, ...perGpu(params, n, st, recipe), fit: smallestFit(params, st, recipe, hbm)}));
  const colors = [seriesColor(0, dark), seriesColor(1, dark), seriesColor(3, dark)];
  const scaleMax = Math.max(...stages.map((r) => r.total), hbm) * 1.08;
  const x = (v: number) => LEFT + (v / scaleMax) * (W - LEFT - RIGHT);
  const picked = stages[stage];
  const status = `ZeRO stage ${stage}: ${picked.total.toFixed(1)} GB per GPU on ${n} GPUs, ${
    picked.total <= hbm ? 'fits in' : 'does not fit in'
  } ${hbm} GB`;

  return (
    <VizPanel
      title="Model state per GPU under DDP and ZeRO"
      hint="The defaults are the ZeRO paper's example: 7.5B parameters on 64 GPUs. The four bars read 120.0, 31.4, 16.6 and 1.9 GB. Activations are not included."
      legend={[
        {label: 'weights', color: colors[0]},
        {label: 'gradients', color: colors[1]},
        {label: 'optimiser state', color: colors[2]},
      ]}
      table={{
        columns: ['stage', 'weights GB', 'gradients GB', 'optimiser GB', 'total GB', 'traffic x model', `smallest N in ${hbm} GB`],
        rows: stages.map((r) => [
          r.stage === 0 ? '0 (DDP)' : r.stage,
          r.w.toFixed(2),
          r.g.toFixed(2),
          r.o.toFixed(2),
          r.total.toFixed(1),
          TRAFFIC[r.stage].toFixed(1),
          r.fit ?? 'never',
        ]),
      }}
      controls={
        <>
          <label className={s.control}>
            model
            <select className={s.select} value={model} onChange={(e) => setModel(Number(e.target.value))}>
              {MODELS.map((m, i) => (
                <option key={m.label} value={i}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            data-parallel GPUs N
            <select className={s.select} value={n} onChange={(e) => setN(Number(e.target.value))}>
              {D_OPTIONS.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            highlight stage
            <select className={s.select} value={stage} onChange={(e) => setStage(Number(e.target.value))}>
              {[0, 1, 2, 3].map((st) => (
                <option key={st} value={st}>
                  {st === 0 ? '0 (plain DDP)' : st}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            bytes per parameter
            <select className={s.select} value={recipe} onChange={(e) => setRecipe(Number(e.target.value))}>
              {RECIPES.map((r, i) => (
                <option key={r.label} value={i}>
                  {r.label}
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
            {status}. Smallest N that fits: {picked.fit ?? 'none up to 4,096'}. Traffic: {TRAFFIC[stage].toFixed(1)}x the model size.
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Four stacked bars of model state per GPU. ${status}`}>
        {stages.map((r, i) => {
          const y = 22 + i * 52;
          let cursor = 0;
          const parts = [r.w, r.g, r.o];
          return (
            <g key={r.stage} opacity={r.stage === stage ? 1 : 0.62}>
              <text className={s.dataLabel} x={LEFT - 8} y={y + 24} textAnchor="end" fontWeight={r.stage === stage ? 700 : 400}>
                {r.stage === 0 ? 'DDP' : `stage ${r.stage}`}
              </text>
              {parts.map((v, k) => {
                const x0 = x(cursor);
                cursor += v;
                return <rect key={k} x={x0} y={y} width={Math.max(0, x(cursor) - x0)} height={32} fill={colors[k]} />;
              })}
              <text className={s.dataLabel} x={x(r.total) + 6} y={y + 21} textAnchor="start">
                {r.total.toFixed(1)}
              </text>
            </g>
          );
        })}
        <line x1={x(hbm)} y1={10} x2={x(hbm)} y2={H - 28} stroke="#e34948" strokeWidth={2.5} strokeDasharray="5 3" />
        <text className={s.tick} x={x(hbm)} y={H - 12} textAnchor="middle">
          {hbm} GB card
        </text>
        <text className={s.tick} x={LEFT} y={H - 12} textAnchor="start">
          0
        </text>
      </svg>
    </VizPanel>
  );
}
