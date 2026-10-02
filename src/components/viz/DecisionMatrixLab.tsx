import {useMemo, useState} from 'react';

import {MATRIX_CRITERIA, MATRIX_DEFAULT_WEIGHTS, MATRIX_OPTIONS, matrixTotals, matrixWinShare} from './craftMath';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const ROW = 62;
const H = 40 + ROW * 3 + 20;
const LEFT = 280;
const PRIVACY = MATRIX_CRITERIA.indexOf('privacy and residency');

export default function DecisionMatrixLab() {
  const dark = useDarkViz();
  const [weights, setWeights] = useState<number[]>(MATRIX_DEFAULT_WEIGHTS);
  const [gate, setGate] = useState(false);

  const totals = useMemo(() => matrixTotals(weights), [weights]);
  const shares = useMemo(() => matrixWinShare(weights), [weights]);
  const alive = MATRIX_OPTIONS.map((o) => !gate || o.scores[PRIVACY] >= 3);
  const leaderIndex = totals.reduce((best, v, i) => (alive[i] && (best < 0 || v > totals[best]) ? i : best), -1);

  const setWeight = (i: number, v: number) => setWeights(weights.map((w, k) => (k === i ? v : w)));
  const barMax = 5;
  const innerW = W - LEFT - 70;

  const rows = MATRIX_CRITERIA.map((c, i) => [c, weights[i], ...MATRIX_OPTIONS.map((o) => o.scores[i])]);
  const status = MATRIX_OPTIONS.map((o, i) => `${o.name}: ${totals[i].toFixed(3)}, wins ${(shares[i] * 100).toFixed(1)}%${alive[i] ? '' : ' (removed by gate)'}`).join('; ');

  return (
    <VizPanel
      title="Decision matrix: score, then test the weights"
      hint="Defaults reproduce the chapter: scores 3.571, 3.429 and 3.143, and the first option wins 72.2% of 2,000 draws in which every weight is scaled by a random factor between 0.5 and 1.5. Turn on the privacy gate and the leader is removed: an average should never hide a hard requirement."
      legend={MATRIX_OPTIONS.map((o, i) => ({label: o.name, color: seriesColor(i, dark)}))}
      table={{columns: ['criterion', 'weight', ...MATRIX_OPTIONS.map((o) => o.name)], rows}}
      controls={
        <>
          {MATRIX_CRITERIA.map((c, i) => (
            <label key={c} className={s.control}>
              {c}
              <input type="range" min={1} max={10} step={1} value={weights[i]} onChange={(e) => setWeight(i, Number(e.target.value))} />
              <span className={s.value}>{weights[i]}</span>
            </label>
          ))}
          <label className={s.control}>
            privacy gate
            <select className={s.select} value={gate ? 'on' : 'off'} onChange={(e) => setGate(e.target.value === 'on')}>
              <option value="off">none</option>
              <option value="on">privacy score of at least 3 to stay in</option>
            </select>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Weighted scores. ${status}`}>
        <text className={s.axisLabel} x={LEFT} y={16}>
          weighted score (0 to 5)
        </text>
        <text className={s.axisLabel} x={LEFT} y={H - 4}>
          thin bar: share of weight draws won
        </text>
        {MATRIX_OPTIONS.map((o, i) => {
          const y0 = 30 + i * ROW;
          const color = seriesColor(i, dark);
          return (
            <g key={o.name} opacity={alive[i] ? 1 : 0.35}>
              <text className={s.dataLabel} x={LEFT - 10} y={y0 + 22} textAnchor="end">
                {o.name}
              </text>
              <rect x={LEFT} y={y0 + 6} width={(totals[i] / barMax) * innerW} height={22} fill={color} rx={3} />
              <text className={s.dataLabel} x={LEFT + (totals[i] / barMax) * innerW + 6} y={y0 + 22}>
                {totals[i].toFixed(3)}{i === leaderIndex ? '  leads' : ''}
              </text>
              <rect x={LEFT} y={y0 + 34} width={shares[i] * innerW} height={8} fill={color} opacity={0.55} rx={2} />
              <text className={s.tick} x={LEFT + shares[i] * innerW + 6} y={y0 + 42}>
                {(shares[i] * 100).toFixed(1)}%
              </text>
            </g>
          );
        })}
      </svg>
    </VizPanel>
  );
}
