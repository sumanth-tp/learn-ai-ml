import {useMemo, useState} from 'react';

import {areaUnderPr, binormalAt, binormalCurves} from './evalMath';
import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const N_CASES = 1000;
const W = 640;
const H = 300;
const BOX = 230;
const LEFT = {x: 44, y: 22};
const RIGHT = {x: 370, y: 22};

export default function RocPrLab() {
  const dark = useDarkViz();
  const [separation, setSeparation] = useState(1.5);
  const [prevalence, setPrevalence] = useState(0.1);
  const [threshold, setThreshold] = useState(1.0);

  const curves = useMemo(() => binormalCurves(separation, prevalence), [separation, prevalence]);
  const area = useMemo(() => areaUnderPr(separation, prevalence), [separation, prevalence]);
  const at = binormalAt(separation, prevalence, threshold, N_CASES);
  const f1 = at.precision + at.tpr === 0 ? 0 : (2 * at.precision * at.tpr) / (at.precision + at.tpr);

  const rocColor = seriesColor(0, dark);
  const prColor = seriesColor(1, dark);
  const base = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;

  const px = (box: {x: number}, v: number) => box.x + v * BOX;
  const py = (box: {y: number}, v: number) => box.y + (1 - v) * BOX;
  const line = (pts: [number, number][]) =>
    pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');

  const rocPath = line(curves.roc.map((p) => [px(LEFT, p.fpr), py(LEFT, p.tpr)]));
  const prPath = line(curves.pr.map((p) => [px(RIGHT, p.recall), py(RIGHT, p.precision)]));

  const sweep = [-1, 0, 0.5, 1, 1.5, 2, 3].map((t) => {
    const b = binormalAt(separation, prevalence, t, N_CASES);
    return [t.toFixed(1), b.tpr.toFixed(3), b.fpr.toFixed(3), b.precision.toFixed(3)];
  });

  const ticks = [0, 0.5, 1];
  const axes = (box: {x: number; y: number}, xl: string, yl: string) => (
    <g>
      <rect x={box.x} y={box.y} width={BOX} height={BOX} fill="none" stroke="var(--border-strong)" />
      {ticks.map((t) => (
        <g key={t}>
          <text className={s.tick} x={px(box, t)} y={box.y + BOX + 14} textAnchor="middle">{t}</text>
          <text className={s.tick} x={box.x - 6} y={py(box, t) + 3} textAnchor="end">{t}</text>
        </g>
      ))}
      <text className={s.axisLabel} x={box.x + BOX / 2} y={box.y + BOX + 30} textAnchor="middle">{xl}</text>
      <text className={s.axisLabel} x={box.x - 32} y={box.y + BOX / 2} textAnchor="middle"
            transform={`rotate(-90 ${box.x - 32} ${box.y + BOX / 2})`}>{yl}</text>
    </g>
  );

  return (
    <VizPanel
      title="ROC and precision-recall curves, one threshold"
      hint="Drag prevalence down: the ROC curve and its AUC do not move, but the precision-recall curve collapses toward the dashed baseline. Defaults match the chapter's code: AUC 0.856, TP 69, FP 143, precision 0.326."
      legend={[
        {label: 'ROC curve', color: rocColor},
        {label: 'precision-recall curve', color: prColor},
        {label: 'chance / prevalence baseline', color: base},
      ]}
      table={{columns: ['threshold', 'TPR (recall)', 'FPR', 'precision'], rows: sweep}}
      controls={
        <>
          <label className={s.control}>
            separation d
            <input type="range" min={0} max={4} step={0.1} value={separation}
                   onChange={(e) => setSeparation(Number(e.target.value))} />
            <span className={s.value}>{separation.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            prevalence
            <input type="range" min={0.01} max={0.5} step={0.01} value={prevalence}
                   onChange={(e) => setPrevalence(Number(e.target.value))} />
            <span className={s.value}>{prevalence.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            threshold
            <input type="range" min={-2} max={4} step={0.1} value={threshold}
                   onChange={(e) => setThreshold(Number(e.target.value))} />
            <span className={s.value}>{threshold.toFixed(1)}</span>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="ROC curve and precision-recall curve with a movable threshold">
        {axes(LEFT, 'false positive rate', 'true positive rate')}
        {axes(RIGHT, 'recall', 'precision')}
        <line x1={px(LEFT, 0)} y1={py(LEFT, 0)} x2={px(LEFT, 1)} y2={py(LEFT, 1)} stroke={base} strokeDasharray="4 4" />
        <line x1={px(RIGHT, 0)} y1={py(RIGHT, prevalence)} x2={px(RIGHT, 1)} y2={py(RIGHT, prevalence)}
              stroke={base} strokeDasharray="4 4" />
        <path d={rocPath} fill="none" stroke={rocColor} strokeWidth={2.5} />
        <path d={prPath} fill="none" stroke={prColor} strokeWidth={2.5} />
        <circle cx={px(LEFT, at.fpr)} cy={py(LEFT, at.tpr)} r={5} fill={rocColor} stroke="var(--surface-raised)" strokeWidth={2} />
        <circle cx={px(RIGHT, at.tpr)} cy={py(RIGHT, at.precision)} r={5} fill={prColor} stroke="var(--surface-raised)" strokeWidth={2} />
        <text className={s.dataLabel} x={px(LEFT, 0.97)} y={py(LEFT, 0.06)} textAnchor="end">AUC {at.auc.toFixed(3)}</text>
        <text className={s.dataLabel} x={px(RIGHT, 0.97)} y={py(RIGHT, 0.94)} textAnchor="end">area {area.toFixed(3)}</text>
      </svg>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.4rem'}}>
        <span>
          at threshold {threshold.toFixed(1)} of {N_CASES} cases: TP {at.tp.toFixed(0)}, FN {at.fn.toFixed(0)},
          FP {at.fp.toFixed(0)}, TN {at.tn.toFixed(0)} | TPR {at.tpr.toFixed(3)}, FPR {at.fpr.toFixed(3)},
          precision {at.precision.toFixed(3)}, F1 {f1.toFixed(3)}
        </span>
      </div>
    </VizPanel>
  );
}
