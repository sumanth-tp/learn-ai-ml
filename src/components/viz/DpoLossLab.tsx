import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 280;
const PAD = {top: 16, right: 24, bottom: 36, left: 48};
const GAP_MAX = 25;

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
export const dpoLoss = (margin: number) => Math.log(1 + Math.exp(-margin));
export const gapForLoss = (target: number, beta: number) => -Math.log(Math.exp(target) - 1) / beta;
export const BETAS = [0.01, 0.05, 0.1, 0.5, 1.0];

export default function DpoLossLab() {
  const dark = useDarkViz();
  const [beta, setBeta] = useState(0.1);
  const [chosen, setChosen] = useState(2.0);
  const [rejected, setRejected] = useState(-1.5);

  const gap = chosen - rejected;
  const rewardChosen = beta * chosen;
  const rewardRejected = beta * rejected;
  const margin = rewardChosen - rewardRejected;
  const loss = dpoLoss(margin);
  const weight = sigmoid(-margin);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const yMax = 3;
  const x = (g: number) => PAD.left + ((g + GAP_MAX) / (2 * GAP_MAX)) * innerW;
  const y = (v: number) => PAD.top + innerH - (Math.min(v, yMax) / yMax) * innerH;
  const lossColor = seriesColor(0, dark);
  const weightColor = seriesColor(1, dark);

  const samples: number[] = [];
  for (let g = -GAP_MAX; g <= GAP_MAX + 1e-9; g += 0.5) samples.push(g);
  const curve = (f: (g: number) => number) =>
    samples.map((g, i) => `${i ? 'L' : 'M'}${x(g).toFixed(1)},${y(f(g)).toFixed(1)}`).join(' ');

  const rows = BETAS.map((b) => {
    const m = b * gap;
    return [b, m.toFixed(3), dpoLoss(m).toFixed(4), sigmoid(-m).toFixed(4), gapForLoss(0.1, b).toFixed(1)];
  });

  const status = `Rewards ${rewardChosen.toFixed(3)} and ${rewardRejected.toFixed(3)}, margin ${margin.toFixed(3)}, loss ${loss.toFixed(4)}, gradient weight ${weight.toFixed(4)}.`;
  const zeroX = x(0);
  const barScale = 200 / Math.max(0.5, Math.abs(rewardChosen), Math.abs(rewardRejected));
  const bar = (value: number, yPos: number, color: string, label: string) => (
    <g>
      <rect
        x={value >= 0 ? zeroX : zeroX + Math.max(value * barScale, -(zeroX - PAD.left))}
        y={yPos}
        width={Math.min(Math.abs(value) * barScale, value >= 0 ? W - PAD.right - zeroX : zeroX - PAD.left)}
        height={14}
        fill={color}
      />
      <text className={s.dataLabel} x={PAD.left} y={yPos + 11}>
        {label} {value.toFixed(3)}
      </text>
    </g>
  );

  return (
    <VizPanel
      title="The DPO loss, step by step"
      hint="Default: log-ratios +2.0 and -1.5 at beta 0.1 give rewards +0.200 and -0.150, margin 0.350, loss 0.5334 and gradient weight 0.4134. Set both log-ratios to 0 for loss 0.6931, the value at the start of training. Raise beta to 0.5 and the same log-ratios give loss 0.1602."
      legend={[
        {label: 'loss', color: lossColor},
        {label: 'gradient weight', color: weightColor},
        {label: 'implicit reward', color: dark ? DIVERGING.dark.positive : DIVERGING.light.positive},
      ]}
      table={{columns: ['beta', 'margin', 'loss', 'gradient weight', 'gap for loss 0.1'], rows}}
      controls={
        <>
          <label className={s.control}>
            beta
            <input type="range" min={0.01} max={1} step={0.01} value={beta} onChange={(e) => setBeta(Number(e.target.value))} />
            <span className={s.value}>{beta.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            chosen log-ratio
            <input type="range" min={-10} max={10} step={0.1} value={chosen} onChange={(e) => setChosen(Number(e.target.value))} />
            <span className={s.value}>{chosen.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            rejected log-ratio
            <input type="range" min={-10} max={10} step={0.1} value={rejected} onChange={(e) => setRejected(Number(e.target.value))} />
            <span className={s.value}>{rejected.toFixed(1)}</span>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H + 70}`} role="img" aria-label={status}>
        {[0, 1, 2, 3].map((t) => (
          <g key={t}>
            <line className={s.grid} x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} />
            <text className={s.tick} x={PAD.left - 8} y={y(t) + 3} textAnchor="end">
              {t}
            </text>
          </g>
        ))}
        {[-20, -10, 0, 10, 20].map((t) => (
          <text key={t} className={s.tick} x={x(t)} y={PAD.top + innerH + 16} textAnchor="middle">
            {t}
          </text>
        ))}
        <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 2} textAnchor="middle">
          log-ratio gap, chosen minus rejected
        </text>
        <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} />
        <path d={curve((g) => dpoLoss(beta * g))} fill="none" stroke={lossColor} strokeWidth={2.2} />
        <path d={curve((g) => sigmoid(-beta * g))} fill="none" stroke={weightColor} strokeWidth={2.2} />
        <line x1={x(Math.max(-GAP_MAX, Math.min(GAP_MAX, gap)))} x2={x(Math.max(-GAP_MAX, Math.min(GAP_MAX, gap)))} y1={PAD.top} y2={PAD.top + innerH} stroke="var(--text-strong)" strokeDasharray="4 3" />
        <circle cx={x(Math.max(-GAP_MAX, Math.min(GAP_MAX, gap)))} cy={y(loss)} r={4.5} fill={lossColor} stroke="var(--surface-raised)" strokeWidth={1.5} />
        <circle cx={x(Math.max(-GAP_MAX, Math.min(GAP_MAX, gap)))} cy={y(weight)} r={4.5} fill={weightColor} stroke="var(--surface-raised)" strokeWidth={1.5} />
        <line className={s.axis} x1={zeroX} y1={H + 6} x2={zeroX} y2={H + 62} />
        {bar(rewardChosen, H + 8, dark ? DIVERGING.dark.positive : DIVERGING.light.positive, 'chosen reward')}
        {bar(rewardRejected, H + 34, dark ? DIVERGING.dark.negative : DIVERGING.light.negative, 'rejected reward')}
      </svg>
      <p className={s.value} style={{padding: '0.4rem 0 0'}} aria-live="polite">
        {status} Gap {gap.toFixed(1)}.
      </p>
    </VizPanel>
  );
}
