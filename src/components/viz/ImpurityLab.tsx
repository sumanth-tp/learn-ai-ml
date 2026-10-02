import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const entropyOf = (pos: number, neg: number) => {
  const n = pos + neg;
  if (n === 0) return 0;
  return [pos, neg].reduce((a, c) => (c > 0 ? a - (c / n) * Math.log2(c / n) : a), 0);
};
const giniOf = (pos: number, neg: number) => {
  const n = pos + neg;
  if (n === 0) return 0;
  return 1 - (pos / n) ** 2 - (neg / n) ** 2;
};
const errorOf = (pos: number, neg: number) => {
  const n = pos + neg;
  return n === 0 ? 0 : 1 - Math.max(pos, neg) / n;
};

type Child = {name: string; pos: number; neg: number};

const SPLITS: Record<string, Child[]> = {
  Outlook: [
    {name: 'Sunny', pos: 2, neg: 3},
    {name: 'Overcast', pos: 4, neg: 0},
    {name: 'Rain', pos: 3, neg: 2},
  ],
  Temperature: [
    {name: 'Hot', pos: 2, neg: 2},
    {name: 'Mild', pos: 4, neg: 2},
    {name: 'Cool', pos: 3, neg: 1},
  ],
  Humidity: [
    {name: 'High', pos: 3, neg: 4},
    {name: 'Normal', pos: 6, neg: 1},
  ],
  Wind: [
    {name: 'Weak', pos: 6, neg: 2},
    {name: 'Strong', pos: 3, neg: 3},
  ],
};

type Criterion = 'entropy' | 'gini';

function gainOf(children: Child[], criterion: Criterion) {
  const f = criterion === 'entropy' ? entropyOf : giniOf;
  const pos = children.reduce((a, c) => a + c.pos, 0);
  const neg = children.reduce((a, c) => a + c.neg, 0);
  const n = pos + neg;
  const weighted = children.reduce((a, c) => a + ((c.pos + c.neg) / n) * f(c.pos, c.neg), 0);
  return {parent: f(pos, neg), weighted, gain: f(pos, neg) - weighted};
}

const W = 420;
const H = 250;
const PAD = {top: 14, right: 14, bottom: 34, left: 44};

export default function ImpurityLab() {
  const dark = useDarkViz();
  const [pos, setPos] = useState(9);
  const [neg, setNeg] = useState(5);
  const [attribute, setAttribute] = useState('Outlook');
  const [criterion, setCriterion] = useState<Criterion>('entropy');

  const colEntropy = seriesColor(0, dark);
  const colGini = seriesColor(1, dark);
  const colError = seriesColor(2, dark);
  const colPos = seriesColor(0, dark);
  const colNeg = seriesColor(1, dark);

  const n = pos + neg;
  const p = n ? pos / n : null;
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const px = (v: number) => PAD.left + v * innerW;
  const py = (v: number) => PAD.top + innerH - v * innerH;
  const curve = (f: (a: number, b: number) => number) =>
    Array.from({length: 101}, (_, i) => {
      const q = i / 100;
      return `${i ? 'L' : 'M'}${px(q).toFixed(1)},${py(f(q, 1 - q)).toFixed(1)}`;
    }).join(' ');

  const children = SPLITS[attribute];
  const split = gainOf(children, criterion);
  const allGains = useMemo(
    () => Object.entries(SPLITS).map(([name, c]) => ({name, ...gainOf(c, criterion)})),
    [criterion],
  );
  const maxGain = Math.max(...allGains.map((g) => g.gain));
  const f = criterion === 'entropy' ? entropyOf : giniOf;

  const grid = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 9 / 14, 0.7, 0.8, 0.9, 1];

  return (
    <VizPanel
      title="Impurity calculator and split gain"
      hint="Set positives and negatives to see entropy, Gini and error: all are zero for a pure node and peak at 50/50. Then pick an attribute of the 14-row play-tennis table to see the weighted child impurity and the gain. The defaults give 0.940, 0.459 and, for Outlook, a gain of 0.247."
      legend={[
        {label: 'entropy (bits)', color: colEntropy},
        {label: 'Gini', color: colGini},
        {label: 'misclassification error', color: colError},
      ]}
      table={{
        columns: ['share of positives', 'entropy', 'Gini', 'error'],
        rows: grid.map((q) => [
          q.toFixed(3),
          entropyOf(q * 1000, (1 - q) * 1000).toFixed(3),
          giniOf(q * 1000, (1 - q) * 1000).toFixed(3),
          errorOf(q * 1000, (1 - q) * 1000).toFixed(3),
        ]),
      }}
      controls={
        <>
          <label className={s.control}>
            positives
            <input type="range" min={0} max={50} step={1} value={pos} onChange={(e) => setPos(Number(e.target.value))} />
            <span className={s.value}>{pos}</span>
          </label>
          <label className={s.control}>
            negatives
            <input type="range" min={0} max={50} step={1} value={neg} onChange={(e) => setNeg(Number(e.target.value))} />
            <span className={s.value}>{neg}</span>
          </label>
          <label className={s.control}>
            split on
            <select className={s.select} value={attribute} onChange={(e) => setAttribute(e.target.value)}>
              {Object.keys(SPLITS).map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            criterion
            <select className={s.select} value={criterion} onChange={(e) => setCriterion(e.target.value as Criterion)}>
              <option value="entropy">entropy</option>
              <option value="gini">Gini</option>
            </select>
          </label>
        </>
      }>
      <div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap'}}>
        <svg className={s.svg} style={{flex: '1 1 260px', minWidth: 0}} viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label="Entropy, Gini and error against the share of positives">
          <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} />
          <line className={s.axis} x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + innerH} />
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <text className={s.tick} x={px(v)} y={H - 16} textAnchor="middle">{v}</text>
              <text className={s.tick} x={PAD.left - 6} y={py(v) + 3} textAnchor="end">{v}</text>
            </g>
          ))}
          <text className={s.axisLabel} x={W / 2} y={H - 2} textAnchor="middle">share of positives p</text>
          <path d={curve(errorOf)} fill="none" stroke={colError} strokeWidth={2} strokeDasharray="5 4" />
          <path d={curve(giniOf)} fill="none" stroke={colGini} strokeWidth={2.2} />
          <path d={curve(entropyOf)} fill="none" stroke={colEntropy} strokeWidth={2.5} />
          {p !== null && (
            <>
              <line x1={px(p)} y1={PAD.top} x2={px(p)} y2={PAD.top + innerH} stroke="var(--border-strong)" strokeDasharray="3 3" />
              <circle cx={px(p)} cy={py(entropyOf(pos, neg))} r={5} fill={colEntropy} stroke="var(--surface-raised)" strokeWidth={1.5} />
              <circle cx={px(p)} cy={py(giniOf(pos, neg))} r={5} fill={colGini} stroke="var(--surface-raised)" strokeWidth={1.5} />
            </>
          )}
        </svg>
        <svg className={s.svg} style={{flex: '1 1 260px', minWidth: 0}} viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label="Child nodes of the chosen split and the gain of each attribute">
          <rect x={78} y={0} width={9} height={9} fill={colPos} />
          <text className={s.tick} x={91} y={8}>positive</text>
          <rect x={146} y={0} width={9} height={9} fill={colNeg} />
          <text className={s.tick} x={159} y={8}>negative</text>
          {children.map((c, i) => {
            const total = c.pos + c.neg;
            const y = 22 + i * 34;
            const bar = 150;
            return (
              <g key={c.name}>
                <text className={s.dataLabel} x={2} y={y + 13}>{c.name}</text>
                <rect x={78} y={y} width={(bar * c.pos) / total} height={18} fill={colPos} />
                <rect x={78 + (bar * c.pos) / total} y={y} width={(bar * c.neg) / total} height={18} fill={colNeg} />
                <text className={s.tick} x={78 + bar + 8} y={y + 13}>
                  [{c.pos}+,{c.neg}−] {criterion === 'entropy' ? 'H' : 'G'} = {f(c.pos, c.neg).toFixed(3)}
                </text>
              </g>
            );
          })}
          <text className={s.axisLabel} x={2} y={22 + children.length * 34 + 6}>
            gain by {criterion === 'entropy' ? 'information gain' : 'Gini decrease'}
          </text>
          {allGains.map((g, i) => {
            const y = 22 + children.length * 34 + 16 + i * 24;
            const w = maxGain ? (g.gain / maxGain) * 190 : 0;
            return (
              <g key={g.name}>
                <text className={s.dataLabel} x={2} y={y + 12} fontWeight={g.name === attribute ? 800 : 600}>{g.name}</text>
                <rect x={92} y={y} width={w} height={16} fill={colEntropy} opacity={g.name === attribute ? 1 : 0.4} />
                <text className={s.tick} x={92 + w + 6} y={y + 12}>{g.gain.toFixed(3)}</text>
              </g>
            );
          })}
        </svg>
      </div>
      <p style={{margin: '0.5rem 0 0', fontSize: '0.82rem', color: 'var(--text-muted)'}}>
        node [{pos}+,{neg}−]: p = <code>{p === null ? 'n/a' : p.toFixed(3)}</code>, entropy <code>{entropyOf(pos, neg).toFixed(3)}</code>,
        Gini <code>{giniOf(pos, neg).toFixed(3)}</code>, error <code>{errorOf(pos, neg).toFixed(3)}</code>. Split on {attribute}:
        parent <code>{split.parent.toFixed(3)}</code>, weighted children <code>{split.weighted.toFixed(3)}</code>, gain{' '}
        <code>{split.gain.toFixed(3)}</code>.
      </p>
    </VizPanel>
  );
}
