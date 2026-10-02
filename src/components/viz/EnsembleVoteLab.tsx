import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 290;
const VOTER_CHOICES = [1, 3, 5, 7, 9, 11, 15, 25, 51, 101];
const CURVE_SIZES = Array.from({length: 51}, (_, i) => 2 * i + 1);

const LOG_FACT: number[] = [0];
for (let i = 1; i <= 101; i += 1) LOG_FACT.push(LOG_FACT[i - 1] + Math.log(i));

export function binomialPmf(n: number, p: number): number[] {
  const out: number[] = [];
  for (let k = 0; k <= n; k += 1) {
    const logChoose = LOG_FACT[n] - LOG_FACT[k] - LOG_FACT[n - k];
    const term =
      (k === 0 ? 0 : k * Math.log(p)) + (n - k === 0 ? 0 : (n - k) * Math.log(1 - p));
    out.push(Math.exp(logChoose + term));
  }
  return out;
}

export function mixturePmf(n: number, p: number, rho: number): number[] {
  const base = binomialPmf(n, p);
  return base.map((mass, k) => {
    const shared = k === n ? p : k === 0 ? 1 - p : 0;
    return (1 - rho) * mass + rho * shared;
  });
}

export function voteAccuracy(n: number, p: number, rho: number): number {
  const need = Math.floor(n / 2) + 1;
  const independent = binomialPmf(n, p)
    .slice(need)
    .reduce((a, b) => a + b, 0);
  return rho * p + (1 - rho) * independent;
}

export default function EnsembleVoteLab() {
  const dark = useDarkViz();
  const [p, setP] = useState(0.7);
  const [n, setN] = useState(3);
  const [rho, setRho] = useState(0);

  const pmf = useMemo(() => mixturePmf(n, p, rho), [n, p, rho]);
  const accuracy = voteAccuracy(n, p, rho);
  const curve = useMemo(() => CURVE_SIZES.map((m) => voteAccuracy(m, p, rho)), [p, rho]);

  const need = Math.floor(n / 2) + 1;
  const winColor = DIVERGING[dark ? 'dark' : 'light'].positive;
  const loseColor = DIVERGING[dark ? 'dark' : 'light'].mid;
  const lineColor = seriesColor(0, dark);

  const leftX0 = 44;
  const leftW = 250;
  const rightX0 = 360;
  const rightW = 250;
  const top = 36;
  const bottom = H - 52;
  const plotH = bottom - top;

  const pmfMax = Math.max(...pmf, 0.0001);
  const barW = Math.min(26, leftW / (n + 1) - 2);
  const barX = (k: number) => leftX0 + ((k + 0.5) / (n + 1)) * leftW - barW / 2;
  const curveX = (m: number) => rightX0 + ((m - 1) / 100) * rightW;
  const curveY = (a: number) => bottom - ((a - 0.4) / 0.6) * plotH;

  const rows = [1, 3, 5, 11, 25].map((m) => [m, voteAccuracy(m, p, rho).toFixed(3)]);
  const labelEvery = Math.max(1, Math.ceil(n / 12));

  return (
    <VizPanel
      title="Majority-vote accuracy"
      hint="Set each model's accuracy and the number of voters. With independent errors the vote climbs fast; raise the error correlation and the gain disappears, because voters that share a mistake outvote nobody."
      legend={[
        {label: 'majority of voters right', color: winColor},
        {label: 'not a majority', color: loseColor},
        {label: 'ensemble accuracy against number of voters', color: lineColor},
      ]}
      table={{columns: ['voters', 'ensemble accuracy'], rows}}
      controls={
        <>
          <label className={s.control}>
            model accuracy p
            <input
              type="range"
              min={0.5}
              max={0.95}
              step={0.01}
              value={p}
              onChange={(e) => setP(Number(e.target.value))}
            />
            <span className={s.value}>{p.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            voters
            <select className={s.select} value={n} onChange={(e) => setN(Number(e.target.value))}>
              {VOTER_CHOICES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            error correlation
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={rho}
              onChange={(e) => setRho(Number(e.target.value))}
            />
            <span className={s.value}>{rho.toFixed(2)}</span>
          </label>
          <span className={s.value} aria-live="polite">
            ensemble accuracy {accuracy.toFixed(3)}
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Majority vote of ${n} voters, each ${(p * 100).toFixed(0)} percent accurate, is ${(accuracy * 100).toFixed(1)} percent accurate`}>
        <text className={s.axisLabel} x={leftX0 + leftW / 2} y={18} textAnchor="middle">
          how many voters are right
        </text>
        <line className={s.axis} x1={leftX0} y1={bottom} x2={leftX0 + leftW} y2={bottom} />
        {pmf.map((mass, k) => {
          const h = (mass / pmfMax) * (plotH - 8);
          return (
            <g key={k}>
              <rect
                x={barX(k)}
                y={bottom - h}
                width={Math.max(barW, 1)}
                height={h}
                fill={k >= need ? winColor : loseColor}
                opacity={0.9}
              />
              {k % labelEvery === 0 && (
                <text className={s.tick} x={barX(k) + barW / 2} y={bottom + 14} textAnchor="middle">
                  {k}
                </text>
              )}
            </g>
          );
        })}
        <text className={s.axisLabel} x={leftX0 + leftW / 2} y={H - 8} textAnchor="middle">
          voters right (k) out of {n}
        </text>

        <text className={s.axisLabel} x={rightX0 + rightW / 2} y={18} textAnchor="middle">
          accuracy against number of voters
        </text>
        <line className={s.axis} x1={rightX0} y1={bottom} x2={rightX0 + rightW} y2={bottom} />
        <line className={s.axis} x1={rightX0} y1={top} x2={rightX0} y2={bottom} />
        {[0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line className={s.grid} x1={rightX0} y1={curveY(tick)} x2={rightX0 + rightW} y2={curveY(tick)} />
            <text className={s.tick} x={rightX0 - 6} y={curveY(tick) + 3} textAnchor="end">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        {[1, 25, 51, 101].map((tick) => (
          <text key={tick} className={s.tick} x={curveX(tick)} y={bottom + 14} textAnchor="middle">
            {tick}
          </text>
        ))}
        <path
          d={CURVE_SIZES.map((m, i) => `${i ? 'L' : 'M'}${curveX(m).toFixed(1)},${curveY(curve[i]).toFixed(1)}`).join(' ')}
          fill="none"
          stroke={lineColor}
          strokeWidth={2.4}
        />
        <circle
          cx={curveX(n)}
          cy={curveY(accuracy)}
          r={5}
          fill={lineColor}
          stroke="var(--surface-raised)"
          strokeWidth={2}
        />
        <text className={s.dataLabel} x={Math.min(curveX(n) + 8, rightX0 + rightW - 34)} y={curveY(accuracy) - 10}>
          {accuracy.toFixed(3)}
        </text>
        <text className={s.axisLabel} x={rightX0 + rightW / 2} y={H - 8} textAnchor="middle">
          number of voters
        </text>
      </svg>
    </VizPanel>
  );
}
