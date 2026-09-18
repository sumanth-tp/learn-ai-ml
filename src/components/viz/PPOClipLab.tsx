import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 260;
const PAD = {top: 16, right: 16, bottom: 34, left: 48};

/** The PPO clipped surrogate objective, as a function of the probability ratio. */
function objective(ratio: number, advantage: number, eps: number) {
  const unclipped = ratio * advantage;
  const clipped = Math.min(Math.max(ratio, 1 - eps), 1 + eps) * advantage;
  return Math.min(unclipped, clipped);
}

export default function PPOClipLab() {
  const dark = useDarkViz();
  const [eps, setEps] = useState(0.2);
  const [advantage, setAdvantage] = useState(1);

  const points = useMemo(() => {
    const out: {ratio: number; clipped: number; unclipped: number}[] = [];
    for (let i = 0; i <= 200; i += 1) {
      const ratio = i / 100;                       // 0 … 2
      out.push({
        ratio,
        clipped: objective(ratio, advantage, eps),
        unclipped: ratio * advantage,
      });
    }
    return out;
  }, [eps, advantage]);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const yMax = 2.2;
  const x = (ratio: number) => PAD.left + (ratio / 2) * innerW;
  const y = (value: number) => PAD.top + innerH / 2 - (value / yMax) * (innerH / 2);

  const path = (key: 'clipped' | 'unclipped') =>
    points.map((p, i) => `${i ? 'L' : 'M'}${x(p.ratio).toFixed(1)},${y(p[key]).toFixed(1)}`).join(' ');

  const clipColor = seriesColor(0, dark);
  const rawColor = seriesColor(1, dark);
  const zone = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;

  const gradientDeadAbove = advantage > 0;

  return (
    <VizPanel
      title="The PPO clipped objective"
      hint={
        gradientDeadAbove
          ? 'With a positive advantage the objective flattens once the ratio passes 1+ε — the gradient switches off, so the update cannot push this action much further. That flat region is the trust region.'
          : 'With a negative advantage the flat region is below 1−ε instead: the update can reduce this action’s probability, but only so far in one step.'
      }
      legend={[
        {label: 'clipped objective (what PPO maximises)', color: clipColor},
        {label: 'unclipped ρ·A', color: rawColor, note: 'vanilla policy gradient'},
        {label: 'trust region 1±ε', color: zone},
      ]}
      table={{
        columns: ['ratio ρ', 'unclipped ρ·A', 'clipped objective'],
        rows: [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0].map((r) => [
          r.toFixed(2),
          (r * advantage).toFixed(2),
          objective(r, advantage, eps).toFixed(2),
        ]),
      }}
      controls={
        <>
          <label className={s.control}>
            ε (clip range)
            <input
              type="range"
              min={0.05}
              max={0.5}
              step={0.01}
              value={eps}
              onChange={(e) => setEps(Number(e.target.value))}
            />
            <span className={s.value}>{eps.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            advantage A
            <select
              className={s.select}
              value={advantage}
              onChange={(e) => setAdvantage(Number(e.target.value))}>
              <option value={1}>+1 (good action)</option>
              <option value={-1}>−1 (bad action)</option>
            </select>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="PPO clipped objective against the probability ratio">
        {/* trust region band */}
        <rect
          x={x(1 - eps)}
          y={PAD.top}
          width={x(1 + eps) - x(1 - eps)}
          height={innerH}
          fill={zone}
          opacity={0.12}
        />
        {/* axes */}
        <line className={s.axis} x1={PAD.left} y1={y(0)} x2={W - PAD.right} y2={y(0)} />
        <line className={s.axis} x1={x(1)} y1={PAD.top} x2={x(1)} y2={PAD.top + innerH}
              strokeDasharray="3 3" />
        {[0, 0.5, 1, 1.5, 2].map((tick) => (
          <text key={tick} className={s.tick} x={x(tick)} y={H - 12} textAnchor="middle">
            {tick.toFixed(1)}
          </text>
        ))}
        {[-2, -1, 0, 1, 2].map((tick) => (
          <text key={tick} className={s.tick} x={PAD.left - 8} y={y(tick) + 3} textAnchor="end">
            {tick}
          </text>
        ))}
        <text className={s.axisLabel} x={W / 2} y={H - 1} textAnchor="middle">
          probability ratio ρ = π_new / π_old
        </text>

        <path d={path('unclipped')} fill="none" stroke={rawColor} strokeWidth={2}
              strokeDasharray="5 4" />
        <path d={path('clipped')} fill="none" stroke={clipColor} strokeWidth={2.5} />

        {/* markers at the clip boundaries */}
        {[1 - eps, 1 + eps].map((r) => (
          <circle key={r} cx={x(r)} cy={y(objective(r, advantage, eps))} r={4}
                  fill={clipColor} stroke="var(--surface-raised)" strokeWidth={2} />
        ))}
      </svg>
    </VizPanel>
  );
}
