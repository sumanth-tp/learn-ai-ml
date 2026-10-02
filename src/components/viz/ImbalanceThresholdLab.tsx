import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const PREVALENCES = [0.005, 0.02, 0.05, 0.1, 0.2, 0.5];
const CASES = 10000;

function erfc(x: number): number {
  if (x < 0) return 2 - erfc(-x);
  if (x < 2.5) {
    let term = x;
    let sum = x;
    for (let n = 1; n < 80; n += 1) {
      term *= (-x * x) / n;
      sum += term / (2 * n + 1);
    }
    return 1 - (2 / Math.sqrt(Math.PI)) * sum;
  }
  let t = x;
  for (let k = 80; k >= 1; k -= 1) t = x + k / 2 / t;
  return Math.exp(-x * x) / (Math.sqrt(Math.PI) * t);
}

const survival = (z: number) => 0.5 * erfc(z / Math.SQRT2);
const density = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

function rates(threshold: number, d: number) {
  return {tpr: survival(threshold - d), fpr: survival(threshold)};
}

function precisionAt(prevalence: number, threshold: number, d: number) {
  const {tpr, fpr} = rates(threshold, d);
  const denominator = prevalence * tpr + (1 - prevalence) * fpr;
  return denominator === 0 ? 1 : (prevalence * tpr) / denominator;
}

function averagePrecision(prevalence: number, d: number) {
  const steps = 20000;
  const low = -8;
  const high = d + 8;
  let previous = rates(low, d).tpr;
  let total = 0;
  for (let i = 1; i <= steps; i += 1) {
    const t = low + ((high - low) * i) / steps;
    const {tpr} = rates(t, d);
    total += (previous - tpr) * precisionAt(prevalence, t, d);
    previous = tpr;
  }
  return total;
}

const W = 250;
const H = 230;

export default function ImbalanceThresholdLab() {
  const dark = useDarkViz();
  const [prevalenceIndex, setPrevalenceIndex] = useState(1);
  const [d, setD] = useState(2);
  const [threshold, setThreshold] = useState(1);
  const [missCost, setMissCost] = useState(10);

  const prevalence = PREVALENCES[prevalenceIndex];
  const {tpr, fpr} = rates(threshold, d);
  const precision = precisionAt(prevalence, threshold, d);
  const caught = CASES * prevalence * tpr;
  const missed = CASES * prevalence * (1 - tpr);
  const alarms = CASES * (1 - prevalence) * fpr;
  const rejected = CASES * (1 - prevalence) * (1 - fpr);
  const accuracy = prevalence * tpr + (1 - prevalence) * (1 - fpr);
  const f1 = precision + tpr === 0 ? 0 : (2 * precision * tpr) / (precision + tpr);
  const cost = missed * missCost + alarms;
  const bestThreshold = (Math.log(((1 - prevalence) * 1) / (prevalence * missCost)) + (d * d) / 2) / d;
  const bestRates = rates(bestThreshold, d);
  const bestCost =
    CASES * (prevalence * (1 - bestRates.tpr) * missCost + (1 - prevalence) * bestRates.fpr);

  const aucRoc = 1 - survival(d / Math.SQRT2);
  const ap = useMemo(() => averagePrecision(prevalence, d), [prevalence, d]);

  const negColor = seriesColor(0, dark);
  const posColor = seriesColor(1, dark);
  const pointColor = seriesColor(3, dark);

  const lo = -4;
  const hi = d + 4;
  const peak = density(0);
  const dx = (v: number) => 12 + ((v - lo) / (hi - lo)) * (W - 24);
  const dy = (v: number) => 14 + (1 - v / (peak * 1.05)) * (H - 56);
  const curves = useMemo(() => {
    const neg: string[] = [];
    const pos: string[] = [];
    for (let i = 0; i <= 160; i += 1) {
      const v = lo + ((hi - lo) * i) / 160;
      neg.push(`${i ? 'L' : 'M'}${dx(v).toFixed(1)},${dy((1 - prevalence) * density(v)).toFixed(1)}`);
      pos.push(`${i ? 'L' : 'M'}${dx(v).toFixed(1)},${dy(prevalence * density(v - d)).toFixed(1)}`);
    }
    return {neg: neg.join(' '), pos: pos.join(' ')};
  }, [prevalence, d]);

  const sweep = useMemo(() => {
    const roc: string[] = [];
    const pr: string[] = [];
    for (let i = 0; i <= 300; i += 1) {
      const t = d + 6 - ((d + 12) * i) / 300;
      const r = rates(t, d);
      roc.push(`${i ? 'L' : 'M'}${(36 + r.fpr * (W - 52)).toFixed(1)},${(14 + (1 - r.tpr) * (H - 56)).toFixed(1)}`);
      pr.push(
        `${i ? 'L' : 'M'}${(36 + r.tpr * (W - 52)).toFixed(1)},${(14 + (1 - precisionAt(prevalence, t, d)) * (H - 56)).toFixed(1)}`,
      );
    }
    return {roc: roc.join(' '), pr: pr.join(' ')};
  }, [prevalence, d]);

  const axisX = (v: number) => 36 + v * (W - 52);
  const axisY = (v: number) => 14 + (1 - v) * (H - 56);

  const ticks = [0, 0.5, 1];
  const frame = (xLabel: string, yLabel: string) => (
    <>
      {ticks.map((t) => (
        <g key={t}>
          <line className={s.grid} x1={36} y1={axisY(t)} x2={W - 16} y2={axisY(t)} />
          <text className={s.tick} x={30} y={axisY(t) + 3} textAnchor="end">
            {t}
          </text>
          <text className={s.tick} x={axisX(t)} y={H - 26} textAnchor="middle">
            {t}
          </text>
        </g>
      ))}
      <text className={s.axisLabel} x={(W + 20) / 2} y={H - 10} textAnchor="middle">
        {xLabel}
      </text>
      <text className={s.axisLabel} x={10} y={H / 2 - 14} textAnchor="middle" transform={`rotate(-90 10 ${H / 2 - 14})`}>
        {yLabel}
      </text>
    </>
  );

  const fixed = (v: number, p = 1) => v.toFixed(p);

  return (
    <VizPanel
      title="Rare positives: accuracy, ROC and precision move apart"
      hint="Scores for negatives follow a bell curve at 0 and for positives a bell curve d higher. Make positives rare: the ROC curve does not move and accuracy looks superb, while precision at the same threshold collapses. The best threshold depends on what a miss costs."
      legend={[
        {label: 'negatives (scaled by 1 - prevalence)', color: negColor},
        {label: 'positives (scaled by prevalence)', color: posColor},
        {label: 'operating point', color: pointColor},
      ]}
      table={{
        columns: ['quantity', 'value'],
        rows: [
          ['caught (true positives)', fixed(caught)],
          ['missed (false negatives)', fixed(missed)],
          ['false alarms (false positives)', fixed(alarms)],
          ['correct rejections', fixed(rejected)],
          ['recall', fixed(tpr, 4)],
          ['false-positive rate', fixed(fpr, 4)],
          ['precision', fixed(precision, 4)],
          ['accuracy', fixed(accuracy, 4)],
          ['accuracy of always negative', fixed(1 - prevalence, 4)],
          ['F1', fixed(f1, 4)],
          ['ROC-AUC', fixed(aucRoc, 3)],
          ['average precision', fixed(ap, 4)],
          ['expected cost', fixed(cost, 0)],
          ['minimum cost', fixed(bestCost, 0)],
          ['cost-optimal threshold', fixed(bestThreshold, 2)],
        ],
      }}
      controls={
        <>
          <label className={s.control}>
            prevalence
            <input
              type="range"
              min={0}
              max={PREVALENCES.length - 1}
              step={1}
              value={prevalenceIndex}
              onChange={(e) => setPrevalenceIndex(Number(e.target.value))}
            />
            <span className={s.value}>{(prevalence * 100).toFixed(1)}%</span>
          </label>
          <label className={s.control}>
            separation d
            <input
              type="range"
              min={0.5}
              max={4}
              step={0.1}
              value={d}
              onChange={(e) => setD(Number(e.target.value))}
            />
            <span className={s.value}>{d.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            threshold
            <input
              type="range"
              min={-2}
              max={6}
              step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <span className={s.value}>{threshold.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            cost of a miss
            <input
              type="range"
              min={1}
              max={100}
              step={1}
              value={missCost}
              onChange={(e) => setMissCost(Number(e.target.value))}
            />
            <span className={s.value}>{missCost}</span>
          </label>
          <button
            type="button"
            className={s.button}
            onClick={() => setThreshold(Math.max(-2, Math.min(6, Math.round(bestThreshold * 100) / 100)))}>
            jump to cost-optimal threshold
          </button>
        </>
      }>
      <div style={{display: 'flex', gap: '0.75rem', flexWrap: 'wrap'}}>
        <svg
          className={s.svg}
          style={{flex: '1 1 220px', minWidth: 0}}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label="Score distributions of negatives and positives with the threshold line">
          <line className={s.axis} x1={12} y1={dy(0)} x2={W - 12} y2={dy(0)} />
          <path d={curves.neg} fill="none" stroke={negColor} strokeWidth={2.5} />
          <path d={curves.pos} fill="none" stroke={posColor} strokeWidth={2.5} />
          <line
            x1={dx(threshold)}
            y1={14}
            x2={dx(threshold)}
            y2={dy(0)}
            stroke={pointColor}
            strokeWidth={2}
            strokeDasharray="4 3"
          />
          <text className={s.axisLabel} x={W / 2} y={H - 10} textAnchor="middle">
            score
          </text>
          <text className={s.tick} x={dx(0)} y={dy(0) + 14} textAnchor="middle">
            0
          </text>
          <text className={s.tick} x={dx(d)} y={dy(0) + 14} textAnchor="middle">
            {d.toFixed(1)}
          </text>
        </svg>
        <svg
          className={s.svg}
          style={{flex: '1 1 220px', minWidth: 0}}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label="ROC curve with the operating point">
          {frame('false-positive rate', 'recall')}
          <line x1={axisX(0)} y1={axisY(0)} x2={axisX(1)} y2={axisY(1)} stroke="var(--text-faint)" strokeDasharray="4 3" />
          <path d={sweep.roc} fill="none" stroke={negColor} strokeWidth={2.5} />
          <circle cx={axisX(fpr)} cy={axisY(tpr)} r={5} fill={pointColor} stroke="var(--surface-raised)" strokeWidth={2} />
          <text className={s.dataLabel} x={W - 20} y={axisY(0) - 8} textAnchor="end">
            ROC-AUC {aucRoc.toFixed(3)}
          </text>
        </svg>
        <svg
          className={s.svg}
          style={{flex: '1 1 220px', minWidth: 0}}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label="Precision-recall curve with the operating point">
          {frame('recall', 'precision')}
          <line
            x1={axisX(0)}
            y1={axisY(prevalence)}
            x2={axisX(1)}
            y2={axisY(prevalence)}
            stroke="var(--text-faint)"
            strokeDasharray="4 3"
          />
          <path d={sweep.pr} fill="none" stroke={posColor} strokeWidth={2.5} />
          <circle cx={axisX(tpr)} cy={axisY(precision)} r={5} fill={pointColor} stroke="var(--surface-raised)" strokeWidth={2} />
          <text className={s.dataLabel} x={W - 20} y={30} textAnchor="end">
            AP {ap.toFixed(3)}
          </text>
        </svg>
      </div>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem', display: 'block'}}>
        <div>
          per {CASES.toLocaleString('en-GB')} cases: caught <code>{fixed(caught)}</code>, missed <code>{fixed(missed)}</code>,
          false alarms <code>{fixed(alarms)}</code>, correct rejections <code>{fixed(rejected)}</code>
        </div>
        <div style={{marginTop: '0.25rem'}}>
          recall <code>{fixed(tpr, 4)}</code>, false-positive rate <code>{fixed(fpr, 4)}</code>, precision{' '}
          <code>{fixed(precision, 4)}</code>, F1 <code>{fixed(f1, 3)}</code>
        </div>
        <div style={{marginTop: '0.25rem'}}>
          accuracy <code>{fixed(accuracy, 4)}</code> against <code>{fixed(1 - prevalence, 4)}</code> for always
          answering negative
        </div>
        <div style={{marginTop: '0.25rem'}}>
          expected cost <code>{fixed(cost, 0)}</code>; the minimum is <code>{fixed(bestCost, 0)}</code> at threshold{' '}
          <code>{fixed(bestThreshold, 2)}</code>
        </div>
      </div>
    </VizPanel>
  );
}
