import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const STOPS = [0, 0.5, 0.8, 0.9, 0.95, 0.98];
const BEFORE = [0.9759, 0.9759, 0.9648, 0.7685, 0.3778, 0.3148];
const AFTER = [0.9759, 0.9778, 0.9815, 0.9759, 0.9481, 0.2889];
const LAYER_ZEROS = ['0/0/0', '0.23/0.54/0.44', '0.45/0.85/0.72', '0.61/0.94/0.87', '0.74/0.98/0.95', '0.88/0.99/0.99'];

const TWO_FOUR = {before: 0.9778};
const STRUCTURED = {before: 0.9407, after: 0.9796};

export const TIMINGS: {label: string; ms: number}[] = [
  {label: 'dense', ms: 0.39},
  {label: '90% zeros, dense tensor', ms: 0.47},
  {label: '90% zeros, CSR', ms: 14.75},
  {label: 'half the rows removed', ms: 0.19},
];

type Method = 'unstructured' | 'two_four' | 'structured';

const GRID = 16;
const CELL = 11;

function weightsBlock(): number[] {
  let a = 20260;
  const rand = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out: number[] = [];
  for (let i = 0; i < GRID * GRID; i += 1) {
    const u = Math.max(rand(), 1e-9);
    const v = rand();
    out.push(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
  }
  return out;
}

const BLOCK = weightsBlock();

export function maskFor(method: Method, sparsity: number): boolean[] {
  const keep = new Array<boolean>(GRID * GRID).fill(true);
  if (sparsity === 0) return keep;
  if (method === 'unstructured') {
    const order = BLOCK.map((w, i) => [Math.abs(w), i]).sort((a, b) => b[0] - a[0]);
    const kept = Math.max(1, Math.round(GRID * GRID * (1 - sparsity)));
    keep.fill(false);
    order.slice(0, kept).forEach(([, i]) => {
      keep[i] = true;
    });
    return keep;
  }
  if (method === 'two_four') {
    keep.fill(false);
    for (let g = 0; g < (GRID * GRID) / 4; g += 1) {
      const idx = [0, 1, 2, 3].map((k) => g * 4 + k).sort((a, b) => Math.abs(BLOCK[b]) - Math.abs(BLOCK[a]));
      keep[idx[0]] = true;
      keep[idx[1]] = true;
    }
    return keep;
  }
  const norms = Array.from({length: GRID}, (_, r) =>
    BLOCK.slice(r * GRID, (r + 1) * GRID).reduce((acc, w) => acc + w * w, 0),
  );
  const drop = Math.round(GRID * sparsity);
  const dropped = new Set(
    norms
      .map((n, r) => [n, r])
      .sort((a, b) => a[0] - b[0])
      .slice(0, drop)
      .map(([, r]) => r),
  );
  return keep.map((_, i) => !dropped.has(Math.floor(i / GRID)));
}

const W = 640;
const H = 300;

function wrapNote(text: string, width: number): string[] {
  const out: string[] = [];
  let cur = '';
  text.split(' ').forEach((word) => {
    if ((cur + ' ' + word).trim().length > width) {
      out.push(cur);
      cur = word;
    } else {
      cur = (cur + ' ' + word).trim();
    }
  });
  if (cur) out.push(cur);
  return out;
}

export default function PruningLab() {
  const dark = useDarkViz();
  const [method, setMethod] = useState<Method>('unstructured');
  const [stop, setStop] = useState(3);

  const sparsity = method === 'two_four' ? 0.5 : STOPS[stop];
  const mask = useMemo(() => maskFor(method, sparsity), [method, sparsity]);
  const kept = mask.filter(Boolean).length;

  const before = method === 'two_four' ? TWO_FOUR.before : method === 'structured' && stop === 1 ? STRUCTURED.before : BEFORE[stop];
  const after = method === 'two_four' ? null : method === 'structured' && stop === 1 ? STRUCTURED.after : AFTER[stop];
  const note =
    method === 'two_four'
      ? '2:4 keeps two of every four weights, so sparsity is fixed at 0.50; measured without fine-tuning.'
      : method === 'structured' && stop !== 1
        ? 'Structured pruning was only measured at 0.50 (256 of 512 first-layer units); the grid still shows the pattern.'
        : method === 'structured'
          ? 'Structured: whole units removed, so the layer is physically smaller.'
          : `Per-layer zeros at this sparsity: ${LAYER_ZEROS[stop]}.`;

  const c0 = seriesColor(1, dark);
  const c1 = seriesColor(0, dark);
  const neutral = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;

  const cx0 = 262;
  const cw = 356;
  const cy0 = 22;
  const ch = 124;
  const px = (sp: number) => cx0 + (sp / 0.98) * cw;
  const py = (a: number) => cy0 + ch - ((a - 0.2) / 0.8) * ch;
  const path = (vals: number[]) => vals.map((v, i) => `${i ? 'L' : 'M'}${px(STOPS[i]).toFixed(1)},${py(v).toFixed(1)}`).join(' ');

  const maxMs = Math.max(...TIMINGS.map((t) => t.ms));
  const barScale = (ms: number) => Math.max(3, (Math.log10(ms * 10) / Math.log10(maxMs * 10)) * 120);

  return (
    <VizPanel
      title="Pruning: what the zeros buy"
      hint="Left: a 16 x 16 block of seeded weights, filled where a weight survives. Right: accuracy of the chapter's digits MLP (540 test images, dense 0.9759) before and after fine-tuning. The default, 0.90 unstructured, gives 0.7685 before and 0.9759 after, as the chapter's code prints. Bottom: timings from one run (they vary on a shared machine, so read the ratios): zeros in a dense tensor cost about the same as a dense matrix, CSR was far slower, and only removing rows made the matmul faster."
      legend={[
        {label: 'no fine-tune', color: c0},
        {label: 'fine-tuned 8 epochs', color: c1},
        {label: 'pruned weight', color: neutral},
      ]}
      table={{
        columns: ['sparsity', 'accuracy before', 'accuracy after', 'zeros per layer'],
        rows: STOPS.map((sp, i) => [sp.toFixed(2), BEFORE[i].toFixed(4), AFTER[i].toFixed(4), LAYER_ZEROS[i]]),
      }}
      controls={
        <>
          <label className={s.control}>
            method
            <select className={s.select} value={method} onChange={(e) => setMethod(e.target.value as Method)}>
              <option value="unstructured">unstructured magnitude</option>
              <option value="two_four">2:4 pattern</option>
              <option value="structured">structured units</option>
            </select>
          </label>
          <label className={s.control}>
            sparsity
            <select className={s.select} value={stop} onChange={(e) => setStop(Number(e.target.value))} disabled={method === 'two_four'}>
              {STOPS.map((sp, i) => (
                <option key={sp} value={i}>
                  {sp.toFixed(2)}
                </option>
              ))}
            </select>
          </label>
          <span className={s.value} aria-live="polite">
            kept {kept} of {GRID * GRID} | before {before.toFixed(4)}
            {after !== null ? ` | after ${after.toFixed(4)}` : ''}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Pruning at sparsity ${sparsity.toFixed(2)}: ${kept} of 256 weights kept, accuracy ${before.toFixed(4)} before fine-tuning`}>
        {mask.map((k, i) => (
          <rect
            key={i}
            x={20 + (i % GRID) * CELL}
            y={22 + Math.floor(i / GRID) * CELL}
            width={CELL - 1.5}
            height={CELL - 1.5}
            rx={2}
            fill={k ? seriesColor(0, dark) : 'none'}
            stroke={k ? 'none' : neutral}
            strokeWidth={0.8}
            opacity={k ? Math.min(1, 0.35 + Math.abs(BLOCK[i]) / 2.5) : 0.7}
          />
        ))}
        <text className={s.tick} x={20} y={16}>
          16 x 16 weights ({method === 'unstructured' ? 'global magnitude' : method === 'two_four' ? 'top 2 of every 4' : 'whole rows'})
        </text>
        {wrapNote(note, 38).map((ln, i) => (
          <text key={i} className={s.tick} x={20} y={22 + GRID * CELL + 14 + i * 12}>
            {ln}
          </text>
        ))}

        <rect x={cx0} y={cy0} width={cw} height={ch} fill="none" stroke="var(--border-strong)" />
        {[0.2, 0.6, 1.0].map((a) => (
          <text key={a} className={s.tick} x={cx0 - 6} y={py(a) + 3} textAnchor="end">
            {a.toFixed(1)}
          </text>
        ))}
        {STOPS.map((sp) => (
          <text key={sp} className={s.tick} x={px(sp)} y={cy0 + ch + (sp === 0.95 ? 25 : 14)} textAnchor={sp === 0.98 ? 'start' : 'middle'}>
            {sp}
          </text>
        ))}
        <text className={s.axisLabel} x={cx0 + cw / 2} y={cy0 + ch + 36} textAnchor="middle">
          sparsity
        </text>
        <path d={path(BEFORE)} fill="none" stroke={c0} strokeWidth={2.4} />
        <path d={path(AFTER)} fill="none" stroke={c1} strokeWidth={2.4} />
        <circle cx={px(sparsity)} cy={py(before)} r={5.5} fill={c0} stroke="var(--surface-raised)" strokeWidth={2} />
        {after !== null && <circle cx={px(sparsity)} cy={py(after)} r={5.5} fill={c1} stroke="var(--surface-raised)" strokeWidth={2} />}

        {TIMINGS.map((t, i) => (
          <g key={t.label}>
            <text className={s.tick} x={cx0 + 150} y={216 + i * 22} textAnchor="end">
              {t.label}
            </text>
            <rect x={cx0 + 158} y={204 + i * 22} width={barScale(t.ms)} height={14} rx={3} fill={seriesColor(i === 2 ? 1 : 2, dark)} />
            <text className={s.dataLabel} x={cx0 + 164 + barScale(t.ms)} y={216 + i * 22}>
              {t.ms.toFixed(2)} ms
            </text>
          </g>
        ))}
        <text className={s.tick} x={20} y={H - 6}>
          Timings: 32 x 2048 by 2048 x 2048 matmul, log-scaled bars
        </text>
      </svg>
    </VizPanel>
  );
}
