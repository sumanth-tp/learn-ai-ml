import {useState} from 'react';

import {rooflinePoint} from './inferenceMath';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 320;
const PAD = {top: 18, right: 20, bottom: 40, left: 54};
const X_MIN = 0.5;
const X_MAX = 4096;
const Y_MIN = 0.1;
const Y_MAX = 10000;
const BATCHES = [1, 8, 32, 64, 128, 295, 512];
const PROMPTS = [16, 128, 512, 2048];
const FORMATS: {label: string; bytes: number}[] = [
  {label: 'bf16 (2 bytes)', bytes: 2},
  {label: 'int8 (1 byte)', bytes: 1},
  {label: 'int4 (0.5 bytes)', bytes: 0.5},
];

const log10 = Math.log10;

export default function RooflineLab() {
  const dark = useDarkViz();
  const [peak, setPeak] = useState(989.5);
  const [bandwidth, setBandwidth] = useState(3.35);
  const [params, setParams] = useState(8.03);
  const [format, setFormat] = useState(0);
  const [batch, setBatch] = useState(1);
  const [prompt, setPrompt] = useState(512);

  const bytes = FORMATS[format].bytes;
  const ridge = peak / bandwidth;
  const decode = rooflinePoint(peak, bandwidth, params, bytes, batch);
  const prefill = rooflinePoint(peak, bandwidth, params, bytes, prompt);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const px = (v: number) => PAD.left + ((log10(v) - log10(X_MIN)) / (log10(X_MAX) - log10(X_MIN))) * innerW;
  const py = (v: number) =>
    PAD.top + innerH - ((log10(Math.max(v, Y_MIN)) - log10(Y_MIN)) / (log10(Y_MAX) - log10(Y_MIN))) * innerH;
  const attain = (i: number) => Math.min(peak, bandwidth * i);

  const memoryColor = seriesColor(1, dark);
  const decodeColor = seriesColor(0, dark);
  const prefillColor = seriesColor(2, dark);

  const roofPath = [X_MIN, Math.min(ridge, X_MAX), X_MAX]
    .map((i, k) => `${k ? 'L' : 'M'}${px(i).toFixed(1)},${py(attain(i)).toFixed(1)}`)
    .join(' ');

  const xTicks = [1, 8, 64, 512, 4096];
  const yTicks = [0.1, 1, 10, 100, 1000, 10000];

  const rows = BATCHES.map((b) => {
    const r = rooflinePoint(peak, bandwidth, params, bytes, b);
    return [
      b,
      r.intensity.toFixed(1),
      r.stepMs.toFixed(2),
      Math.round(r.tokensPerSecond).toLocaleString('en-US'),
      r.computeUtil.toFixed(3),
      r.bound,
    ];
  });

  return (
    <VizPanel
      title="Roofline: where one decode step lands"
      hint="Batch 1 sits far left on the sloped memory roof, so the step time is the time to read the weights and compute is almost idle. Raise the batch until the marker reaches the ridge: the step time stops being flat. Defaults are Llama 3.1 8B in bf16 on the H100 SXM figures from the chapter: step 4.79 ms, 209 tokens per second, utilisation 0.003."
      legend={[
        {label: 'memory roof (bandwidth x intensity)', color: memoryColor},
        {label: `decode, batch ${batch}`, color: decodeColor},
        {label: `prefill, ${prompt} tokens`, color: prefillColor},
      ]}
      table={{
        columns: ['batch', 'FLOP per byte', 'step ms', 'tokens/s', 'compute util', 'bound'],
        rows,
      }}
      controls={
        <>
          <label className={s.control}>
            peak TFLOP/s
            <input type="range" min={100} max={2000} step={10} value={peak}
                   onChange={(e) => setPeak(Number(e.target.value))} />
            <span className={s.value}>{peak}</span>
          </label>
          <label className={s.control}>
            bandwidth TB/s
            <input type="range" min={0.5} max={8} step={0.05} value={bandwidth}
                   onChange={(e) => setBandwidth(Number(e.target.value))} />
            <span className={s.value}>{bandwidth.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            parameters (B)
            <input type="range" min={1} max={70} step={0.01} value={params}
                   onChange={(e) => setParams(Number(e.target.value))} />
            <span className={s.value}>{params.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            weights
            <select className={s.select} value={format} onChange={(e) => setFormat(Number(e.target.value))}>
              {FORMATS.map((f, i) => (
                <option key={f.label} value={i}>{f.label}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            decode batch
            <select className={s.select} value={batch} onChange={(e) => setBatch(Number(e.target.value))}>
              {BATCHES.map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            prompt tokens
            <select className={s.select} value={prompt} onChange={(e) => setPrompt(Number(e.target.value))}>
              {PROMPTS.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Roofline with ridge at ${ridge.toFixed(1)} FLOP per byte. Decode at batch ${batch} takes ${decode.stepMs.toFixed(2)} milliseconds and is ${decode.bound}-bound.`}>
        <rect x={PAD.left} y={PAD.top} width={innerW} height={innerH} fill="none" stroke="var(--border-strong)" />
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line className={s.grid} x1={px(t)} y1={PAD.top} x2={px(t)} y2={PAD.top + innerH} />
            <text className={s.tick} x={px(t)} y={PAD.top + innerH + 14} textAnchor="middle">{t}</text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className={s.grid} x1={PAD.left} y1={py(t)} x2={PAD.left + innerW} y2={py(t)} />
            <text className={s.tick} x={PAD.left - 6} y={py(t) + 3} textAnchor="end">{t}</text>
          </g>
        ))}
        <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 6} textAnchor="middle">
          arithmetic intensity (FLOP per byte of weights read)
        </text>
        <text className={s.axisLabel} x={14} y={PAD.top + innerH / 2} textAnchor="middle"
              transform={`rotate(-90 14 ${PAD.top + innerH / 2})`}>
          attainable TFLOP/s
        </text>
        <path d={roofPath} fill="none" stroke={memoryColor} strokeWidth={2.5} />
        {ridge <= X_MAX && (
          <g>
            <line x1={px(ridge)} y1={py(peak)} x2={px(ridge)} y2={PAD.top + innerH} stroke={memoryColor}
                  strokeDasharray="4 4" />
            <text className={s.dataLabel} x={px(ridge) + 6} y={py(peak) + 14}>ridge {ridge.toFixed(1)}</text>
          </g>
        )}
        <circle cx={px(Math.min(Math.max(decode.intensity, X_MIN), X_MAX))} cy={py(attain(decode.intensity))} r={6}
                fill={decodeColor} stroke="var(--surface-raised)" strokeWidth={2} />
        <text className={s.dataLabel} x={px(Math.min(Math.max(decode.intensity, X_MIN), X_MAX)) + 9}
              y={py(attain(decode.intensity)) + 16}>
          decode {decode.stepMs.toFixed(2)} ms
        </text>
        <circle cx={px(Math.min(Math.max(prefill.intensity, X_MIN), X_MAX))} cy={py(attain(prefill.intensity))} r={6}
                fill={prefillColor} stroke="var(--surface-raised)" strokeWidth={2} />
        <text className={s.dataLabel} x={px(Math.min(Math.max(prefill.intensity, X_MIN), X_MAX)) - 9}
              y={py(attain(prefill.intensity)) - 10} textAnchor="end">
          prefill {prefill.stepMs.toFixed(2)} ms
        </text>
      </svg>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.4rem'}}>
        <span aria-live="polite">
          decode batch {batch}: {decode.stepMs.toFixed(2)} ms per step, {Math.round(decode.tokensPerSecond).toLocaleString('en-US')} tokens
          per second, compute utilisation {decode.computeUtil.toFixed(3)}, {decode.bound}-bound. Ridge batch for this
          weight format: {Math.round((ridge * bytes) / 2)}.
        </span>
      </div>
    </VizPanel>
  );
}
