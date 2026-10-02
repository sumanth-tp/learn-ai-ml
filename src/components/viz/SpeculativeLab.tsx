import {useMemo, useState} from 'react';

import {draftTimeline, expectedTokens, speculativeSpeedup} from './inferenceMath';
import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 320;
const STEPS = 8;
const ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9];
const GAMMAS = [1, 2, 3, 4, 5, 6, 7, 8];

export default function SpeculativeLab() {
  const dark = useDarkViz();
  const [alpha, setAlpha] = useState(0.8);
  const [gamma, setGamma] = useState(4);
  const [c, setC] = useState(0.05);
  const [v, setV] = useState(1);

  const speedups = GAMMAS.map((g) => speculativeSpeedup(alpha, g, c, v));
  const best = speedups.indexOf(Math.max(...speedups)) + 1;
  const tokens = expectedTokens(alpha, gamma);
  const speedup = speculativeSpeedup(alpha, gamma, c, v);
  const timeline = useMemo(() => draftTimeline(alpha, gamma, STEPS), [alpha, gamma]);
  const measured = timeline.reduce((a, t) => a + t.accepted + 1, 0) / STEPS;

  const good = DIVERGING.light.positive;
  const bad = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;
  const mid = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;
  const bar = seriesColor(0, dark);
  const bonus = seriesColor(3, dark);

  const left = {x: 40, y: 30, w: 270, h: 220};
  const top = Math.max(1.2, ...speedups) * 1.1;
  const bx = (g: number) => left.x + ((g - 0.5) / 8) * left.w;
  const by = (val: number) => left.y + left.h - (val / top) * left.h;

  const rows = ALPHAS.map((a) => [
    a.toFixed(1),
    ...GAMMAS.map((g) => speculativeSpeedup(a, g, c, v).toFixed(2)),
  ]);

  return (
    <VizPanel
      title="Draft and verify: tokens per step and speedup"
      hint="Raise alpha and the draft is trusted for longer. Raise c and long drafts stop paying. Raise v to see what happens when verifying several tokens is not free, as on the CPU in the chapter. Defaults match block 1: 3.362 tokens per target step and a speedup of 2.80."
      legend={[
        {label: 'speedup for each draft length', color: bar},
        {label: 'accepted draft token', color: good},
        {label: 'rejected draft token', color: bad},
        {label: 'extra token from the target', color: bonus},
      ]}
      table={{columns: ['alpha', ...GAMMAS.map((g) => `gamma ${g}`)], rows}}
      controls={
        <>
          <label className={s.control}>
            acceptance alpha
            <input type="range" min={0.3} max={0.95} step={0.01} value={alpha}
                   onChange={(e) => setAlpha(Number(e.target.value))} />
            <span className={s.value}>{alpha.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            draft length gamma
            <input type="range" min={1} max={8} step={1} value={gamma} onChange={(e) => setGamma(Number(e.target.value))} />
            <span className={s.value}>{gamma}</span>
          </label>
          <label className={s.control}>
            draft cost c
            <input type="range" min={0.01} max={0.5} step={0.01} value={c} onChange={(e) => setC(Number(e.target.value))} />
            <span className={s.value}>{c.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            verify cost v
            <input type="range" min={1} max={4} step={0.1} value={v} onChange={(e) => setV(Number(e.target.value))} />
            <span className={s.value}>{v.toFixed(1)}</span>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Speculative decoding with acceptance ${alpha.toFixed(2)}, draft length ${gamma}, draft cost ${c.toFixed(2)} and verify cost ${v.toFixed(1)}: ${tokens.toFixed(3)} tokens per step and speedup ${speedup.toFixed(2)}.`}>
        <text className={s.axisLabel} x={left.x} y={16}>speedup by draft length</text>
        <rect x={left.x} y={left.y} width={left.w} height={left.h} fill="none" stroke="var(--border-strong)" />
        <line x1={left.x} y1={by(1)} x2={left.x + left.w} y2={by(1)} stroke={mid} strokeDasharray="4 4" />
        <text className={s.tick} x={left.x - 6} y={by(1) + 3} textAnchor="end">1.0</text>
        <text className={s.tick} x={left.x - 6} y={left.y + 8} textAnchor="end">{top.toFixed(1)}</text>
        {GAMMAS.map((g, i) => (
          <g key={g}>
            <rect x={bx(g) - 11} y={by(speedups[i])} width={22} height={left.y + left.h - by(speedups[i])} rx={2}
                  fill={bar} opacity={g === gamma ? 1 : 0.5} stroke={g === best ? 'var(--text-strong)' : 'none'} strokeWidth={2} />
            <text className={s.tick} x={bx(g)} y={left.y + left.h + 14} textAnchor="middle">{g}</text>
            <text className={s.dataLabel} x={bx(g)} y={by(speedups[i]) - 4} textAnchor="middle">{speedups[i].toFixed(1)}</text>
          </g>
        ))}
        <text className={s.axisLabel} x={left.x + left.w / 2} y={left.y + left.h + 30} textAnchor="middle">
          gamma (outlined bar is the best)
        </text>
        <text className={s.axisLabel} x={350} y={16}>eight target steps, seeded</text>
        {timeline.map((t, k) => {
          const y = 28 + k * 26;
          return (
            <g key={k}>
              {Array.from({length: gamma}, (_, i) => {
                const state = i < t.accepted ? 'ok' : i === t.accepted ? 'bad' : 'skip';
                return (
                  <rect key={i} x={350 + i * 21} y={y} width={18} height={18} rx={3}
                        fill={state === 'ok' ? good : state === 'bad' ? bad : 'none'}
                        stroke={state === 'skip' ? mid : 'none'} strokeDasharray={state === 'skip' ? '3 2' : undefined} />
                );
              })}
              <rect x={350 + gamma * 21 + 5} y={y} width={18} height={18} rx={3} fill={bonus} />
              <text className={s.tick} x={350 + gamma * 21 + 28} y={y + 13}>{t.accepted + 1} tokens</text>
            </g>
          );
        })}
        <text className={s.dataLabel} x={350} y={28 + STEPS * 26 + 14}>
          these steps: {measured.toFixed(2)} tokens each; formula {tokens.toFixed(2)}
        </text>
      </svg>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.4rem'}}>
        <span aria-live="polite">
          alpha {alpha.toFixed(2)}, gamma {gamma}: {tokens.toFixed(3)} tokens per target step, speedup {speedup.toFixed(2)} with c {c.toFixed(2)} and v{' '}
          {v.toFixed(1)}. Best draft length {best} gives {Math.max(...speedups).toFixed(2)}.
        </span>
      </div>
    </VizPanel>
  );
}
