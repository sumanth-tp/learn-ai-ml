import {useState} from 'react';

import {normalCdf} from './evalMath';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const PAD = {top: 20, right: 24, bottom: 40, left: 52};

const A = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
const B = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
const C = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
const D = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];

export function normalQuantile(p: number): number {
  const low = 0.02425;
  if (p < low) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1);
  }
  if (p > 1 - low) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return ((((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q) / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1);
}

export function usersPerArm(p0: number, mde: number, alpha: number, power: number, rho = 0): number {
  const p1 = p0 + mde;
  const pbar = (p0 + p1) / 2;
  const za = normalQuantile(1 - alpha / 2);
  const zb = normalQuantile(power);
  const top = za * Math.sqrt(2 * pbar * (1 - pbar)) + zb * Math.sqrt(p0 * (1 - p0) + p1 * (1 - p1));
  return Math.ceil(((top * top) / (mde * mde)) * (1 - rho * rho));
}

export function powerAt(p0: number, mde: number, alpha: number, n: number, rho = 0): number {
  const p1 = p0 + mde;
  const pbar = (p0 + p1) / 2;
  const za = normalQuantile(1 - alpha / 2);
  const effective = n / (1 - rho * rho);
  const z = (mde * Math.sqrt(effective) - za * Math.sqrt(2 * pbar * (1 - pbar))) / Math.sqrt(p0 * (1 - p0) + p1 * (1 - p1));
  return normalCdf(z);
}

const fmt = (v: number) => v.toLocaleString('en-GB');
const X_MIN = 500;
const X_MAX = 400000;

export default function AbTestPowerLab() {
  const dark = useDarkViz();
  const [p0, setP0] = useState(0.1);
  const [mde, setMde] = useState(0.01);
  const [alpha, setAlpha] = useState(0.05);
  const [power, setPower] = useState(0.8);
  const [rho, setRho] = useState(0);
  const [daily, setDaily] = useState(5000);

  const needed = usersPerArm(p0, mde, alpha, power, rho);
  const days = Math.ceil((2 * needed) / daily);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const lx = (v: number) => PAD.left + ((Math.log(v) - Math.log(X_MIN)) / (Math.log(X_MAX) - Math.log(X_MIN))) * innerW;
  const y = (v: number) => PAD.top + innerH - v * innerH;
  const grid = Array.from({length: 61}, (_, i) => Math.exp(Math.log(X_MIN) + ((Math.log(X_MAX) - Math.log(X_MIN)) * i) / 60));
  const path = grid.map((n, i) => `${i === 0 ? 'M' : 'L'}${lx(n).toFixed(1)},${y(powerAt(p0, mde, alpha, n, rho)).toFixed(1)}`).join(' ');
  const baseline = grid.map((n, i) => `${i === 0 ? 'M' : 'L'}${lx(n).toFixed(1)},${y(powerAt(p0, mde, alpha, n, 0)).toFixed(1)}`).join(' ');

  const line = seriesColor(0, dark);
  const second = seriesColor(1, dark);
  const neutral = dark ? '#848c99' : '#9aa0a6';
  const markerX = Math.min(Math.max(needed, X_MIN), X_MAX);

  const rows = [2, 1, 0.5].map((factor) => {
    const m = mde * factor;
    const plain = usersPerArm(p0, m, alpha, power, 0);
    const adjusted = usersPerArm(p0, m, alpha, power, rho);
    return [`x${factor} (${m.toFixed(4)})`, fmt(plain), fmt(adjusted), fmt(2 * adjusted), Math.ceil((2 * adjusted) / daily)];
  });

  const ticks = [1000, 10000, 100000];

  return (
    <VizPanel
      title="How many users an A/B test needs"
      hint="Halve the effect you want to detect and the sample quadruples. Raise the CUPED correlation and the same power arrives with fewer users. Defaults match the chapter's block 1: 14,751 users per arm for a 1 point lift on a 10% baseline, with 3,841 at a 2 point lift and 57,763 at 0.5 points."
      legend={[
        {label: 'power with current settings', color: line},
        {label: 'power without CUPED', color: second},
        {label: 'target power', color: neutral},
      ]}
      table={{columns: ['effect (MDE)', 'per arm, plain', 'per arm, CUPED', 'total, CUPED', 'days'], rows}}
      controls={
        <>
          <label className={s.control}>
            baseline rate
            <input type="range" min={0.01} max={0.5} step={0.01} value={p0} onChange={(e) => setP0(Number(e.target.value))} />
            <span className={s.value}>{p0.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            MDE (absolute)
            <input type="range" min={0.002} max={0.05} step={0.001} value={mde} onChange={(e) => setMde(Number(e.target.value))} />
            <span className={s.value}>{mde.toFixed(3)}</span>
          </label>
          <label className={s.control}>
            alpha
            <select className={s.select} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))}>
              {[0.01, 0.05, 0.1].map((v) => (
                <option key={v} value={v}>{v.toFixed(2)}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            power
            <select className={s.select} value={power} onChange={(e) => setPower(Number(e.target.value))}>
              {[0.7, 0.8, 0.9].map((v) => (
                <option key={v} value={v}>{v.toFixed(2)}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            CUPED rho
            <input type="range" min={0} max={0.95} step={0.05} value={rho} onChange={(e) => setRho(Number(e.target.value))} />
            <span className={s.value}>{rho.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            users per day
            <select className={s.select} value={daily} onChange={(e) => setDaily(Number(e.target.value))}>
              {[2000, 5000, 20000, 100000].map((v) => (
                <option key={v} value={v}>{fmt(v)}</option>
              ))}
            </select>
          </label>
          <span className={s.value} aria-live="polite">
            {fmt(needed)} per arm, {fmt(2 * needed)} in total, {days} {days === 1 ? 'day' : 'days'}
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Power against users per arm. ${fmt(needed)} users per arm reach power ${power}.`}>
        <line className={s.axis} x1={PAD.left} y1={y(0)} x2={W - PAD.right} y2={y(0)} />
        <line className={s.axis} x1={PAD.left} y1={y(0)} x2={PAD.left} y2={y(1)} />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} className={s.tick} x={PAD.left - 8} y={y(t) + 4} textAnchor="end">
            {t.toFixed(2)}
          </text>
        ))}
        {ticks.map((t) => (
          <text key={t} className={s.tick} x={lx(t)} y={H - 18} textAnchor="middle">
            {fmt(t)}
          </text>
        ))}
        <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 2} textAnchor="middle">users per arm (log scale)</text>
        <line x1={PAD.left} y1={y(power)} x2={W - PAD.right} y2={y(power)} stroke={neutral} strokeDasharray="5 4" />
        {rho > 0 && <path d={baseline} fill="none" stroke={second} strokeWidth={2} strokeDasharray="6 3" />}
        <path d={path} fill="none" stroke={line} strokeWidth={2.5} />
        <line x1={lx(markerX)} y1={PAD.top} x2={lx(markerX)} y2={y(0)} stroke={line} strokeDasharray="3 3" />
        <circle cx={lx(markerX)} cy={y(power)} r={5} fill={line} stroke="var(--surface-raised)" strokeWidth={2} />
        <text className={s.dataLabel} x={Math.min(lx(markerX) + 8, W - PAD.right - 70)} y={y(power) - 10}>
          {fmt(needed)}
        </text>
      </svg>
    </VizPanel>
  );
}
