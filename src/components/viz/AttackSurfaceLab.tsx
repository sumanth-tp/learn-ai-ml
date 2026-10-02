import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const LAYERS = [
  {key: 0, label: 'input filter'},
  {key: 1, label: 'output filter'},
  {key: 2, label: 'tool policy'},
  {key: 3, label: 'delimit untrusted data'},
  {key: 4, label: 'no secrets in the prompt'},
];

const CATEGORIES = [
  'direct injection',
  'jailbreak',
  'exfiltration',
  'indirect injection',
  'tool abuse',
  'knowledge poisoning',
];

const WINS: number[][] = [
  [57, 57, 65, 68, 50, 64], [24, 27, 16, 16, 15, 12], [57, 0, 16, 68, 50, 64], [24, 0, 16, 16, 15, 12],
  [57, 57, 65, 0, 3, 64], [24, 27, 16, 0, 3, 12], [57, 0, 16, 0, 3, 64], [24, 0, 16, 0, 3, 12],
  [57, 57, 65, 14, 50, 15], [24, 27, 16, 6, 15, 2], [57, 0, 16, 14, 50, 15], [24, 0, 16, 6, 15, 2],
  [57, 57, 65, 0, 3, 15], [24, 27, 16, 0, 3, 2], [57, 0, 16, 0, 3, 15], [24, 0, 16, 0, 3, 2],
  [57, 57, 0, 68, 50, 64], [24, 27, 0, 16, 15, 12], [57, 0, 0, 68, 50, 64], [24, 0, 0, 16, 15, 12],
  [57, 57, 0, 0, 3, 64], [24, 27, 0, 0, 3, 12], [57, 0, 0, 0, 3, 64], [24, 0, 0, 0, 3, 12],
  [57, 57, 0, 14, 50, 15], [24, 27, 0, 6, 15, 2], [57, 0, 0, 14, 50, 15], [24, 0, 0, 6, 15, 2],
  [57, 57, 0, 0, 3, 15], [24, 27, 0, 0, 3, 2], [57, 0, 0, 0, 3, 15], [24, 0, 0, 0, 3, 2],
];

const PER_CATEGORY = 80;
const W = 640;
const H = 270;
const LEFT = 150;
const RIGHT = 70;
const TOP = 16;
const ROW = 38;

export default function AttackSurfaceLab() {
  const dark = useDarkViz();
  const [mask, setMask] = useState(0);

  const row = WINS[mask];
  const base = WINS[0];
  const landed = row.reduce((a, b) => a + b, 0);
  const total = PER_CATEGORY * CATEGORIES.length;
  const bar = seriesColor(1, dark);
  const tick = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;
  const innerW = W - LEFT - RIGHT;

  const toggle = (bit: number) => setMask(mask ^ (1 << bit));

  const tableRows = CATEGORIES.map((c, i) => [
    c,
    `${row[i]} of ${PER_CATEGORY}`,
    (row[i] / PER_CATEGORY).toFixed(3),
    (base[i] / PER_CATEGORY).toFixed(3),
  ]);

  return (
    <VizPanel
      title="Which attacks get through, layer by layer"
      hint="Defaults match the chapter's printed table: with no layers 361 of 480 attacks land (0.752); with all five layers 29 land (0.060). Turn one layer on at a time and watch which categories it moves. These numbers describe the stubbed shop assistant, not a real model."
      legend={[
        {label: 'attack success rate with the chosen layers', color: bar},
        {label: 'rate with no defences', color: tick},
      ]}
      table={{columns: ['category', 'attacks that land', 'rate', 'rate with no defences'], rows: tableRows}}
      controls={
        <>
          {LAYERS.map((layer) => (
            <label key={layer.key} className={s.control}>
              <input
                type="checkbox"
                checked={Boolean(mask & (1 << layer.key))}
                onChange={() => toggle(layer.key)}
              />
              {layer.label}
            </label>
          ))}
          <button type="button" className={s.button} onClick={() => setMask(0)} disabled={mask === 0}>
            Reset
          </button>
          <span className={s.value} aria-live="polite">
            {landed} of {total} land, overall {(landed / total).toFixed(3)}
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Attack success rate per category, ${landed} of ${total} attacks land`}>
        {CATEGORIES.map((name, i) => {
          const y = TOP + i * ROW;
          const rate = row[i] / PER_CATEGORY;
          const baseRate = base[i] / PER_CATEGORY;
          return (
            <g key={name}>
              <text className={s.axisLabel} x={LEFT - 10} y={y + 17} textAnchor="end">{name}</text>
              <rect x={LEFT} y={y} width={innerW} height={24} rx={4} fill="none" stroke="var(--border-subtle)" />
              <rect x={LEFT} y={y} width={Math.max(rate * innerW, rate > 0 ? 2 : 0)} height={24} rx={4} fill={bar} />
              <line x1={LEFT + baseRate * innerW} y1={y - 3} x2={LEFT + baseRate * innerW} y2={y + 27}
                    stroke={tick} strokeWidth={2} strokeDasharray="3 3" />
              <text className={s.dataLabel} x={LEFT + innerW + 8} y={y + 17}>{rate.toFixed(3)}</text>
            </g>
          );
        })}
        <text className={s.tick} x={LEFT} y={H - 6} textAnchor="middle">0</text>
        <text className={s.tick} x={LEFT + innerW} y={H - 6} textAnchor="middle">1</text>
      </svg>
    </VizPanel>
  );
}
