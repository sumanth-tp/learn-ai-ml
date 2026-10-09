import {useState} from 'react';

import {ctcPaths} from './speechMath';
import {SpeechSlider} from './speechLabParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const DEFAULT_FRAMES = [
  {a: 0.6, b: 0.3},
  {a: 0.3, b: 0.5},
  {a: 0.1, b: 0.6},
];

const W = 640;
const LEFT = 90;
const ROW = 32;

export default function CtcPathLab() {
  const dark = useDarkViz();
  const [frames, setFrames] = useState(DEFAULT_FRAMES);
  const result = ctcPaths(frames, 'ab');
  const maxP = Math.max(0.0001, ...result.paths.map((p) => p.probability));
  const height = result.paths.length * ROW + 40;

  const update = (index: number, key: 'a' | 'b', value: number) => {
    setFrames((current) => current.map((frame, i) => (i === index ? {...frame, [key]: value} : frame)));
  };

  const rows: (string | number)[][] = [
    ...result.paths.map((p) => [`path ${p.path}`, p.probability.toFixed(4)]),
    ['sum = P("ab")', result.total.toFixed(4)],
    ['loss = -ln P', Number.isFinite(result.loss) ? result.loss.toFixed(4) : 'infinite'],
    ['greedy path', result.greedyPath],
    ['greedy text', result.greedyText || '(empty)'],
  ];

  return (
    <div data-testid="ctc-lab">
      <VizPanel
        title="How CTC adds up every way of spelling the answer"
        hint="Three frames, three symbols: blank (_), a and b. Defaults are the chapter's worked example: the five paths that collapse to ab add up to 0.4680, a loss of 0.7593, and the greedy path abb collapses to ab."
        table={{columns: ['Quantity', 'Value'], rows}}
        legend={[{label: 'probability of one path', color: seriesColor(0, dark)}]}
        controls={
          <>
            {frames.map((frame, i) => (
              <span key={i} style={{display: 'inline-flex', gap: '0.8rem', flexWrap: 'wrap'}}>
                <SpeechSlider label={`Frame ${i + 1}: P(a)`} value={frame.a} min={0} max={1} step={0.05} digits={2} onChange={(v) => update(i, 'a', v)} />
                <SpeechSlider label={`Frame ${i + 1}: P(b)`} value={Math.min(frame.b, 1 - frame.a)} min={0} max={1} step={0.05} digits={2} onChange={(v) => update(i, 'b', v)} />
              </span>
            ))}
            <span className={s.value} aria-live="polite" data-testid="ctc-summary">
              P(ab) = {result.total.toFixed(4)}, loss {Number.isFinite(result.loss) ? result.loss.toFixed(4) : 'infinite'}, greedy {result.greedyPath} gives {result.greedyText || 'nothing'}
            </span>
          </>
        }>
        <svg className={s.svg} viewBox={`0 0 ${W} ${height}`} role="img" aria-label="Probability of each path that collapses to ab">
          {result.paths.map((p, i) => {
            const y = 8 + i * ROW;
            const width = (p.probability / maxP) * (W - LEFT - 80);
            return (
              <g key={p.path}>
                <text className={s.dataLabel} x={LEFT - 10} y={y + 17} textAnchor="end">
                  {p.path}
                </text>
                <rect x={LEFT} y={y} width={Math.max(1, width)} height={22} fill={seriesColor(0, dark)} opacity={0.9} />
                <text className={s.dataLabel} x={LEFT + Math.max(1, width) + 6} y={y + 16}>
                  {p.probability.toFixed(4)}
                </text>
              </g>
            );
          })}
          <text className={s.axisLabel} x={W / 2} y={height - 8} textAnchor="middle">
            probability of each path whose collapse is ab
          </text>
        </svg>
      </VizPanel>
    </div>
  );
}
