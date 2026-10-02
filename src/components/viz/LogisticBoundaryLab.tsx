import {useId, useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

type Point = {x1: number; x2: number; label: 0 | 1};

function makeData(): Point[] {
  const draw = mulberry32(7);
  const normal = () => {
    const u1 = draw();
    const u2 = draw();
    return Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2);
  };
  const out: Point[] = [];
  const centres: [number, number][] = [[-1, -0.5], [1, 0.5]];
  centres.forEach(([cx, cy], label) => {
    for (let i = 0; i < 40; i += 1) {
      const x1 = cx + 1.1 * normal();
      const x2 = cy + 1.1 * normal();
      out.push({x1, x2, label: label as 0 | 1});
    }
  });
  return out;
}

function fit(data: Point[]): [number, number, number] {
  const theta: [number, number, number] = [0, 0, 0];
  const n = data.length;
  for (let it = 0; it < 4000; it += 1) {
    const g = [0, 0, 0];
    for (const d of data) {
      const e = sigmoid(theta[0] + theta[1] * d.x1 + theta[2] * d.x2) - d.label;
      g[0] += e;
      g[1] += e * d.x1;
      g[2] += e * d.x2;
    }
    theta[0] -= (0.5 * g[0]) / n;
    theta[1] -= (0.5 * g[1]) / n;
    theta[2] -= (0.5 * g[2]) / n;
  }
  return theta;
}

type Model = {
  data: Point[];
  theta: [number, number, number];
  prob: number[];
  roc: [number, number][];
  auc: number;
};

function build(): Model {
  const data = makeData();
  const theta = fit(data);
  const prob = data.map((d) => sigmoid(theta[0] + theta[1] * d.x1 + theta[2] * d.x2));
  const order = prob.map((_, i) => i).sort((a, b) => prob[b] - prob[a]);
  const pos = data.filter((d) => d.label === 1).length;
  const neg = data.length - pos;
  const roc: [number, number][] = [[0, 0]];
  let tp = 0;
  let fp = 0;
  let auc = 0;
  for (const i of order) {
    if (data[i].label === 1) tp += 1;
    else fp += 1;
    const prev = roc[roc.length - 1];
    const next: [number, number] = [fp / neg, tp / pos];
    auc += ((next[0] - prev[0]) * (next[1] + prev[1])) / 2;
    roc.push(next);
  }
  return {data, theta, prob, roc, auc};
}

function counts(model: Model, t: number) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  model.data.forEach((d, i) => {
    const predicted = model.prob[i] >= t;
    if (predicted && d.label === 1) tp += 1;
    else if (predicted) fp += 1;
    else if (d.label === 1) fn += 1;
    else tn += 1;
  });
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return {tp, fp, fn, tn, precision, recall, f1, accuracy: (tp + tn) / model.data.length,
    tpr: recall, fpr: fp + tn ? fp / (fp + tn) : 0};
}

const SWEEP = [0.9, 0.7, 0.5, 0.3, 0.1];
const W = 300;
const H = 260;
const PAD = {top: 12, right: 12, bottom: 32, left: 38};
const RANGE = 4;

export default function LogisticBoundaryLab() {
  const dark = useDarkViz();
  const clipId = `lb${useId().replace(/:/g, '')}`;
  const [threshold, setThreshold] = useState(0.5);
  const [z, setZ] = useState(1);
  const model = useMemo(build, []);
  const m = counts(model, threshold);

  const colNeg = seriesColor(0, dark);
  const colPos = seriesColor(1, dark);
  const colLine = seriesColor(3, dark);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const sx = (v: number) => PAD.left + ((v + RANGE) / (2 * RANGE)) * innerW;
  const sy = (v: number) => PAD.top + innerH - ((v + RANGE) / (2 * RANGE)) * innerH;
  const [t0, t1, t2] = model.theta;
  const level = Math.log(threshold / (1 - threshold));
  const lineY = (x1: number) => (level - t0 - t1 * x1) / t2;
  const big = t2 > 0 ? RANGE * 3 : -RANGE * 3;
  const shade = `${sx(-RANGE)},${sy(lineY(-RANGE))} ${sx(RANGE)},${sy(lineY(RANGE))} ${sx(RANGE)},${sy(big)} ${sx(-RANGE)},${sy(big)}`;

  const rx = (v: number) => PAD.left + v * innerW;
  const ry = (v: number) => PAD.top + innerH - v * innerH;
  const rocPath = model.roc.map((p, i) => `${i ? 'L' : 'M'}${rx(p[0]).toFixed(1)},${ry(p[1]).toFixed(1)}`).join(' ');

  const zx = (v: number) => PAD.left + ((v + 6) / 12) * innerW;
  const curve = Array.from({length: 121}, (_, i) => {
    const v = -6 + (12 * i) / 120;
    return `${i ? 'L' : 'M'}${zx(v).toFixed(1)},${ry(sigmoid(v)).toFixed(1)}`;
  }).join(' ');
  const probe = sigmoid(z);

  const wrong = (i: number) => (model.prob[i] >= threshold ? 1 : 0) !== model.data[i].label;

  const axes = (xLabel: string, yLabel: string) => (
    <>
      <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} />
      <line className={s.axis} x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + innerH} />
      <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 4} textAnchor="middle">{xLabel}</text>
      <text className={s.axisLabel} x={10} y={PAD.top + innerH / 2} textAnchor="middle"
            transform={`rotate(-90 10 ${PAD.top + innerH / 2})`}>{yLabel}</text>
    </>
  );

  return (
    <VizPanel
      title="Logistic regression: score, probability, threshold"
      hint="Slide the threshold: the boundary moves, the counts change and the dot slides along the ROC curve, but the curve itself (and the AUC) never changes. The score slider is the lecture’s decision-rule widget: z = 1 gives 0.731, class 1."
      legend={[
        {label: 'class 0 (circles)', color: colNeg},
        {label: 'class 1 (diamonds)', color: colPos},
        {label: 'decision boundary', color: colLine},
      ]}
      table={{
        columns: ['threshold', 'TPR (recall)', 'FPR', 'precision'],
        rows: SWEEP.map((t) => {
          const c = counts(model, t);
          return [t.toFixed(1), c.tpr.toFixed(3), c.fpr.toFixed(3), c.tp + c.fp ? c.precision.toFixed(3) : 'n/a'];
        }),
      }}
      controls={
        <>
          <label className={s.control}>
            threshold
            <input type="range" min={0.05} max={0.95} step={0.01} value={threshold}
                   onChange={(e) => setThreshold(Number(e.target.value))} />
            <span className={s.value}>{threshold.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            score z
            <input type="range" min={-6} max={6} step={0.5} value={z}
                   onChange={(e) => setZ(Number(e.target.value))} />
            <span className={s.value}>{z.toFixed(1)}</span>
          </label>
        </>
      }>
      <div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap'}}>
        <svg className={s.svg} style={{flex: '1 1 200px', minWidth: 0}} viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label="Two classes of points with the logistic regression decision boundary">
          <defs>
            <clipPath id={clipId}>
              <rect x={PAD.left} y={PAD.top} width={innerW} height={innerH} />
            </clipPath>
          </defs>
          {axes('feature 1', 'feature 2')}
          <g clipPath={`url(#${clipId})`}>
            <polygon points={shade} fill={colPos} opacity={0.1} />
            <line x1={sx(-RANGE)} y1={sy(lineY(-RANGE))} x2={sx(RANGE)} y2={sy(lineY(RANGE))} stroke={colLine} strokeWidth={2.2} />
            {model.data.map((d, i) => {
              const cx = sx(d.x1);
              const cy = sy(d.x2);
              const color = d.label === 1 ? colPos : colNeg;
              return (
                <g key={i}>
                  {d.label === 1 ? (
                    <rect x={cx - 3.6} y={cy - 3.6} width={7.2} height={7.2} fill={color}
                          transform={`rotate(45 ${cx} ${cy})`} />
                  ) : (
                    <circle cx={cx} cy={cy} r={3.8} fill={color} />
                  )}
                  {wrong(i) && <circle cx={cx} cy={cy} r={7} fill="none" stroke="var(--text-strong)" strokeWidth={1.2} />}
                </g>
              );
            })}
          </g>
        </svg>
        <svg className={s.svg} style={{flex: '1 1 200px', minWidth: 0}} viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label="ROC curve with the current operating point">
          {axes('false positive rate', 'true positive rate')}
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <text className={s.tick} x={rx(v)} y={PAD.top + innerH + 12} textAnchor="middle">{v}</text>
              <text className={s.tick} x={PAD.left - 5} y={ry(v) + 3} textAnchor="end">{v}</text>
            </g>
          ))}
          <line x1={rx(0)} y1={ry(0)} x2={rx(1)} y2={ry(1)} stroke="var(--border-strong)" strokeDasharray="4 4" />
          <path d={rocPath} fill="none" stroke={colNeg} strokeWidth={2.2} />
          <circle cx={rx(m.fpr)} cy={ry(m.tpr)} r={5.5} fill={colPos} stroke="var(--surface-raised)" strokeWidth={1.5} />
          <text className={s.dataLabel} x={PAD.left + innerW - 4} y={PAD.top + innerH - 8} textAnchor="end">
            AUC {model.auc.toFixed(4)}
          </text>
        </svg>
        <svg className={s.svg} style={{flex: '1 1 200px', minWidth: 0}} viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label="Sigmoid curve with the threshold and the chosen score">
          {axes('score z', 'probability')}
          {[-6, -3, 0, 3, 6].map((v) => (
            <text key={v} className={s.tick} x={zx(v)} y={PAD.top + innerH + 12} textAnchor="middle">{v}</text>
          ))}
          {[0, 0.5, 1].map((v) => (
            <text key={v} className={s.tick} x={PAD.left - 5} y={ry(v) + 3} textAnchor="end">{v}</text>
          ))}
          <line x1={PAD.left} y1={ry(threshold)} x2={W - PAD.right} y2={ry(threshold)} stroke={colLine} strokeDasharray="4 3" />
          <path d={curve} fill="none" stroke={colNeg} strokeWidth={2.2} />
          <line x1={zx(z)} y1={ry(0)} x2={zx(z)} y2={ry(probe)} stroke={probe >= threshold ? colPos : colNeg} strokeDasharray="2 3" />
          <circle cx={zx(z)} cy={ry(probe)} r={5.5} fill={probe >= threshold ? colPos : colNeg}
                  stroke="var(--surface-raised)" strokeWidth={1.5} />
        </svg>
      </div>
      <p style={{margin: '0.5rem 0 0', fontSize: '0.82rem', color: 'var(--text-muted)'}}>
        weights (bias, w1, w2) = <code>{t0.toFixed(3)}, {t1.toFixed(3)}, {t2.toFixed(3)}</code>. At threshold{' '}
        {threshold.toFixed(2)}: TP <code>{m.tp}</code> FP <code>{m.fp}</code> FN <code>{m.fn}</code> TN <code>{m.tn}</code>;
        precision <code>{m.precision.toFixed(3)}</code>, recall <code>{m.recall.toFixed(3)}</code>, F1{' '}
        <code>{m.f1.toFixed(3)}</code>, accuracy <code>{m.accuracy.toFixed(3)}</code>. Probe: σ({z.toFixed(1)}) ={' '}
        <code>{probe.toFixed(3)}</code>, so class <code>{probe >= threshold ? 1 : 0}</code>.
      </p>
    </VizPanel>
  );
}
