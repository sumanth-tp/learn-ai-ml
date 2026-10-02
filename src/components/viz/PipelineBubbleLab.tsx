import {useState} from 'react';

import {sequentialColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const TABLE_MICRO = [1, 2, 4, 8, 16, 32, 64];

export function bubble(stages: number, micro: number): number {
  return (stages - 1) / (micro + stages - 1);
}

export default function PipelineBubbleLab() {
  const dark = useDarkViz();
  const [stages, setStages] = useState(4);
  const [micro, setMicro] = useState(8);

  const ticks = micro + stages - 1;
  const busy = stages * micro;
  const frac = bubble(stages, micro);

  const left = 80;
  const top = 50;
  const cw = (W - left - 20) / ticks;
  const ch = Math.min(32, (H - top - 56) / stages);
  const idle = dark ? '#2a303a' : '#e4e7ec';

  const rows = TABLE_MICRO.map((m) => [m, m + stages - 1, bubble(stages, m).toFixed(3), m >= 4 * stages ? 'yes' : 'no']);

  return (
    <VizPanel
      title="Pipeline bubble"
      hint="Stage s starts micro-batch j at tick s + j, so the first stages wait to fill and the last stages sit idle while the pipe drains. With 4 stages and 8 micro-batches there are 11 ticks, 32 busy cells out of 44, and an idle share of 3/11 = 0.273, as in the chapter's code. More micro-batches shrink the bubble."
      legend={[
        {label: 'busy (colour = micro-batch)', color: sequentialColor(0.6, dark)},
        {label: 'idle: the bubble', color: idle},
      ]}
      table={{columns: ['micro-batches', 'ticks', `bubble at ${stages} stages`, 'm at least 4 x stages'], rows}}
      controls={
        <>
          <label className={s.control}>
            stages
            <input type="range" min={2} max={8} step={1} value={stages} onChange={(e) => setStages(Number(e.target.value))} />
            <span className={s.value}>{stages}</span>
          </label>
          <label className={s.control}>
            micro-batches
            <input type="range" min={1} max={32} step={1} value={micro} onChange={(e) => setMicro(Number(e.target.value))} />
            <span className={s.value}>{micro}</span>
          </label>
          <span className={s.value} aria-live="polite">
            ticks {ticks}, busy {busy} of {stages * ticks}, bubble {frac.toFixed(3)} ({stages - 1}/{ticks})
            {micro >= 4 * stages ? ', at least 4 x stages' : ''}
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Pipeline with ${stages} stages and ${micro} micro-batches: ${ticks} ticks, bubble ${frac.toFixed(3)}`}>
        <text className={s.axisLabel} x={left + (cw * ticks) / 2} y={30} textAnchor="middle">
          time in ticks, one micro-batch per stage per tick
        </text>
        {Array.from({length: stages}, (_, st) => (
          <g key={st}>
            <text className={s.tick} x={left - 8} y={top + st * ch + ch / 2 + 4} textAnchor="end">
              stage {st}
            </text>
            {Array.from({length: ticks}, (_, t) => {
              const j = t - st;
              const on = j >= 0 && j < micro;
              return (
                <rect
                  key={t}
                  x={left + t * cw + 0.5}
                  y={top + st * ch + 1}
                  width={Math.max(cw - 1, 1)}
                  height={ch - 2}
                  rx={2}
                  fill={on ? sequentialColor(0.2 + 0.8 * (j / Math.max(micro - 1, 1)), dark) : idle}
                />
              );
            })}
          </g>
        ))}
        <text className={s.axisLabel} x={W / 2} y={H - 12} textAnchor="middle">
          bubble = idle cells / all cells = {frac.toFixed(3)}
        </text>
      </svg>
    </VizPanel>
  );
}
