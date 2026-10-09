import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const SWEEP = [
  {k: 0, dev: 0.25, test: 0.25, tokens: 135},
  {k: 4, dev: 0.375, test: 0.375, tokens: 215},
  {k: 8, dev: 0.458, test: 0.479, tokens: 297},
  {k: 12, dev: 0.5, test: 0.583, tokens: 378},
  {k: 16, dev: 0.583, test: 0.604, tokens: 458},
  {k: 24, dev: 0.625, test: 0.771, tokens: 618},
];

const CANDIDATES = [
  {seed: -3, label: 'zero-shot', dev: 0.25, test: 0.25},
  {seed: -2, label: 'labelled demos', dev: 0.375, test: 0.375},
  {seed: -1, label: 'bootstrapped, unshuffled', dev: 0.25, test: 0.25},
  {seed: 0, label: 'shuffled 0', dev: 0.25, test: 0.25},
  {seed: 1, label: 'shuffled 1', dev: 0.333, test: 0.354},
  {seed: 2, label: 'shuffled 2', dev: 0.25, test: 0.292},
  {seed: 3, label: 'shuffled 3', dev: 0.333, test: 0.312},
  {seed: 4, label: 'shuffled 4', dev: 0.333, test: 0.271},
  {seed: 5, label: 'shuffled 5', dev: 0.333, test: 0.292},
  {seed: 6, label: 'shuffled 6', dev: 0.375, test: 0.333},
  {seed: 7, label: 'shuffled 7', dev: 0.292, test: 0.292},
];

const W = 640;
const H = 250;
const LEFT = 44;
const RIGHT = 14;
const TOP = 24;
const BOTTOM = 40;

export default function PromptSearchLab() {
  const dark = useDarkViz();
  const [k, setK] = useState(8);
  const [tried, setTried] = useState(11);

  const point = SWEEP.find((p) => p.k === k) ?? SWEEP[2];
  const seen = CANDIDATES.slice(0, tried);
  let winner = seen[0];
  for (const c of seen) if (c.dev > winner.dev) winner = c;
  const bestDev = Math.max(...seen.map((c) => c.dev));

  const x = (i: number, n: number) => LEFT + ((i + 0.5) / n) * (W - LEFT - RIGHT);
  const y = (v: number) => H - BOTTOM - v * (H - TOP - BOTTOM);
  const devColour = seriesColor(0, dark);
  const testColour = seriesColor(1, dark);
  const hot = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;
  const barW = ((W - LEFT - RIGHT) / CANDIDATES.length) * 0.34;

  const rows = [
    ...SWEEP.map((p) => [`${p.k} labelled demos`, p.dev.toFixed(3), p.test.toFixed(3), p.tokens]),
    ...CANDIDATES.map((c) => [`search seed ${c.seed}: ${c.label}`, c.dev.toFixed(3), c.test.toFixed(3), '']),
  ];

  return (
    <VizPanel
      title="How much does searching for a prompt buy?"
      hint="Everything here was printed by chapter code block 3, run on a deterministic stand-in model that answers by copying the label of the most similar demonstration, so it shows the optimiser's mechanics and not a language model's power. With 8 labelled demos the dev accuracy is 0.458, the test accuracy 0.479 and the prompt about 297 tokens. With all 11 candidates tried, the best on dev is 0.375 and its test score is 0.375."
      legend={[
        {label: 'accuracy on the dev set the search used', color: devColour},
        {label: 'accuracy on held-out test messages', color: testColour},
        {label: 'the candidate chosen on dev', color: hot},
      ]}
      table={{columns: ['program', 'dev accuracy', 'test accuracy', 'prompt tokens'], rows}}
      controls={
        <>
          <label className={s.control}>
            labelled demonstrations
            <select className={s.select} value={k} onChange={(e) => setK(Number(e.target.value))}>
              {SWEEP.map((p) => (
                <option key={p.k} value={p.k}>
                  {p.k}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            candidates tried
            <input type="range" min={1} max={CANDIDATES.length} step={1} value={tried} onChange={(e) => setTried(Number(e.target.value))} />
            <span className={s.value}>{tried}</span>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Dev and test accuracy of 11 candidate prompts. After ${tried} candidates the best on dev is ${bestDev.toFixed(3)} and its test accuracy is ${winner.test.toFixed(3)}.`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className={s.grid} x1={LEFT} x2={W - RIGHT} y1={y(t)} y2={y(t)} />
            <text className={s.tick} x={LEFT - 6} y={y(t) + 3} textAnchor="end">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {CANDIDATES.map((c, i) => {
          const on = i < tried;
          const chosen = on && c === winner;
          return (
            <g key={c.seed} opacity={on ? 1 : 0.25}>
              <rect x={x(i, CANDIDATES.length) - barW - 1} y={y(c.dev)} width={barW} height={y(0) - y(c.dev)} fill={devColour} />
              <rect x={x(i, CANDIDATES.length) + 1} y={y(c.test)} width={barW} height={y(0) - y(c.test)} fill={testColour} />
              {chosen && <rect x={x(i, CANDIDATES.length) - barW - 4} y={TOP - 6} width={2 * barW + 8} height={H - BOTTOM - TOP + 6} fill="none" stroke={hot} strokeWidth={2} />}
              <text className={s.tick} x={x(i, CANDIDATES.length)} y={H - BOTTOM + 14} textAnchor="middle">
                {c.seed}
              </text>
            </g>
          );
        })}
        <text className={s.axisLabel} x={LEFT + (W - LEFT - RIGHT) / 2} y={H - 6} textAnchor="middle">
          candidate (random-search seed; -3 to -1 are the fixed starting points)
        </text>
      </svg>
      <p className={s.hint} style={{padding: '0.4rem 0 0'}} aria-live="polite">
        {k} labelled demonstrations: dev {point.dev.toFixed(3)}, test {point.test.toFixed(3)}, about {point.tokens} prompt tokens per call. After {tried} candidates the best on dev is {bestDev.toFixed(3)} (seed {winner.seed}), and on the test messages it scores {winner.test.toFixed(3)}.
      </p>
    </VizPanel>
  );
}
