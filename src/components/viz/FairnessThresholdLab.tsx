import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Counts = {P: number; N: number; pos: number[]; neg: number[]};

const GROUPS: Record<'A' | 'B', Counts> = {
  A: {P: 1861, N: 4095,
    pos: [1861, 1861, 1858, 1846, 1831, 1793, 1749, 1694, 1618, 1552, 1487, 1418, 1332, 1250, 1162, 1078, 992, 927, 849, 789, 734, 677, 612, 551, 492, 441, 382, 347, 309, 261, 229, 202, 170, 140, 116, 98, 75, 58, 46, 35, 24, 17, 10, 7, 1, 0, 0, 0, 0, 0, 0],
    neg: [4095, 4073, 3966, 3757, 3522, 3227, 2940, 2647, 2406, 2156, 1947, 1710, 1529, 1339, 1163, 1017, 886, 766, 663, 568, 503, 433, 363, 307, 258, 214, 175, 146, 122, 97, 73, 53, 42, 29, 19, 16, 12, 7, 7, 5, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]},
  B: {P: 675, N: 3369,
    pos: [675, 675, 674, 667, 654, 635, 618, 595, 575, 557, 533, 506, 475, 450, 415, 388, 358, 328, 302, 283, 257, 226, 203, 176, 162, 140, 118, 102, 92, 78, 71, 58, 48, 35, 26, 25, 22, 16, 14, 12, 8, 6, 4, 3, 1, 0, 0, 0, 0, 0, 0],
    neg: [3369, 3343, 3190, 2982, 2751, 2470, 2211, 1984, 1763, 1556, 1384, 1232, 1096, 960, 836, 725, 629, 539, 455, 395, 322, 282, 238, 201, 167, 141, 111, 88, 67, 50, 44, 36, 26, 21, 13, 9, 5, 4, 3, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
};

const STEPS = 50;
const W = 640;
const H = 280;
const BOX = 220;
const ROC = {x: 44, y: 22};
const BARS = {x: 340, y: 22, w: 270};

type Point = {sel: number; tpr: number; fpr: number; ppv: number};

function at(g: Counts, i: number): Point {
  const tp = g.pos[i];
  const fp = g.neg[i];
  return {sel: (tp + fp) / (g.P + g.N), tpr: tp / g.P, fpr: fp / g.N, ppv: tp + fp === 0 ? 0 : tp / (tp + fp)};
}

const METRICS: {key: keyof Point; label: string}[] = [
  {key: 'sel', label: 'selection rate'},
  {key: 'tpr', label: 'true positive rate'},
  {key: 'fpr', label: 'false positive rate'},
  {key: 'ppv', label: 'precision'},
];

function nearest(target: number, key: keyof Point, g: Counts): number {
  let best = 0;
  let bestGap = Infinity;
  for (let i = 0; i <= STEPS; i += 1) {
    const gap = Math.abs(at(g, i)[key] - target);
    if (gap < bestGap) {
      best = i;
      bestGap = gap;
    }
  }
  return best;
}

export default function FairnessThresholdLab() {
  const dark = useDarkViz();
  const [ia, setIa] = useState(15);
  const [ib, setIb] = useState(15);
  const [locked, setLocked] = useState(true);

  const a = at(GROUPS.A, ia);
  const b = at(GROUPS.B, ib);
  const colA = seriesColor(0, dark);
  const colB = seriesColor(1, dark);
  const base = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;

  const curve = (g: Counts) =>
    Array.from({length: STEPS + 1}, (_, i) => at(g, i))
      .map((p, i) => `${i ? 'L' : 'M'}${(ROC.x + p.fpr * BOX).toFixed(1)},${(ROC.y + (1 - p.tpr) * BOX).toFixed(1)}`)
      .join(' ');
  const pathA = useMemo(() => curve(GROUPS.A), []);
  const pathB = useMemo(() => curve(GROUPS.B), []);

  const setA = (i: number) => {
    setIa(i);
    if (locked) setIb(i);
  };
  const preset = (key: keyof Point) => {
    setLocked(false);
    setIb(nearest(a[key], key, GROUPS.B));
  };

  const gaps = {
    dp: Math.abs(a.sel - b.sel),
    eo: Math.max(Math.abs(a.tpr - b.tpr), Math.abs(a.fpr - b.fpr)),
  };

  const rows = METRICS.map((m) => [
    m.label,
    a[m.key].toFixed(3),
    b[m.key].toFixed(3),
    Math.abs(a[m.key] - b[m.key]).toFixed(3),
  ]);

  const barW = (v: number) => v * (BARS.w - 120);

  return (
    <VizPanel
      title="One score, two thresholds: which fairness gap moves"
      hint="Defaults are the chapter's first block: both thresholds 0.30 give selection 0.352 and 0.275, TPR 0.579 and 0.575, FPR 0.248 and 0.215, precision 0.515 and 0.349. Use a preset to equalise one metric for group B and watch the others open up."
      legend={[
        {label: 'group A (60 percent of applicants)', color: colA},
        {label: 'group B (40 percent)', color: colB},
        {label: 'chance diagonal', color: base},
      ]}
      table={{columns: ['metric', 'group A', 'group B', 'gap'], rows}}
      controls={
        <>
          <label className={s.control}>
            threshold A
            <input type="range" min={0} max={STEPS} step={1} value={ia} onChange={(e) => setA(Number(e.target.value))} />
            <span className={s.value}>{(ia * 0.02).toFixed(2)}</span>
          </label>
          <label className={s.control}>
            threshold B
            <input type="range" min={0} max={STEPS} step={1} value={ib}
                   onChange={(e) => { setLocked(false); setIb(Number(e.target.value)); }} />
            <span className={s.value}>{(ib * 0.02).toFixed(2)}</span>
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={locked} onChange={(e) => { setLocked(e.target.checked); if (e.target.checked) setIb(ia); }} />
            lock together
          </label>
          {METRICS.map((m) => (
            <button key={m.key} type="button" className={s.button} onClick={() => preset(m.key)}>
              match {m.label}
            </button>
          ))}
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="ROC curves for two groups and bars for four fairness metrics at the chosen thresholds">
        <rect x={ROC.x} y={ROC.y} width={BOX} height={BOX} fill="none" stroke="var(--border-strong)" />
        <line x1={ROC.x} y1={ROC.y + BOX} x2={ROC.x + BOX} y2={ROC.y} stroke={base} strokeDasharray="4 4" />
        <path d={pathA} fill="none" stroke={colA} strokeWidth={2.5} />
        <path d={pathB} fill="none" stroke={colB} strokeWidth={2.5} />
        <circle cx={ROC.x + a.fpr * BOX} cy={ROC.y + (1 - a.tpr) * BOX} r={5} fill={colA} stroke="var(--surface-raised)" strokeWidth={2} />
        <circle cx={ROC.x + b.fpr * BOX} cy={ROC.y + (1 - b.tpr) * BOX} r={5} fill={colB} stroke="var(--surface-raised)" strokeWidth={2} />
        <text className={s.tick} x={ROC.x} y={ROC.y + BOX + 14} textAnchor="middle">0</text>
        <text className={s.tick} x={ROC.x + BOX} y={ROC.y + BOX + 14} textAnchor="middle">1</text>
        <text className={s.axisLabel} x={ROC.x + BOX / 2} y={ROC.y + BOX + 30} textAnchor="middle">false positive rate</text>
        <text className={s.axisLabel} x={ROC.x - 30} y={ROC.y + BOX / 2} textAnchor="middle"
              transform={`rotate(-90 ${ROC.x - 30} ${ROC.y + BOX / 2})`}>true positive rate</text>
        {METRICS.map((m, i) => {
          const y = BARS.y + i * 52;
          return (
            <g key={m.key}>
              <text className={s.axisLabel} x={BARS.x} y={y + 8}>{m.label}</text>
              <rect x={BARS.x} y={y + 14} width={Math.max(barW(a[m.key]), 1)} height={12} rx={3} fill={colA} />
              <rect x={BARS.x} y={y + 28} width={Math.max(barW(b[m.key]), 1)} height={12} rx={3} fill={colB} />
              <text className={s.dataLabel} x={BARS.x + barW(a[m.key]) + 6} y={y + 24}>{a[m.key].toFixed(3)}</text>
              <text className={s.dataLabel} x={BARS.x + barW(b[m.key]) + 6} y={y + 38}>{b[m.key].toFixed(3)}</text>
            </g>
          );
        })}
        <text className={s.dataLabel} x={BARS.x} y={H - 18}>
          demographic parity gap {gaps.dp.toFixed(3)}, equalised odds gap {gaps.eo.toFixed(3)}
        </text>
      </svg>
    </VizPanel>
  );
}
