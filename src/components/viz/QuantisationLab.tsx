import {useMemo, useState} from 'react';

import {QuantScheme, WEIGHT_SLICE, quantiseSlice} from './inferenceMath';
import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 330;
const PAD = {left: 46, right: 14, top: 22};
const GROUPS = [256, 128, 64, 32, 16];
const BITS = [8, 7, 6, 5, 4, 3, 2];
const CLIP = 0.5;

export default function QuantisationLab() {
  const dark = useDarkViz();
  const [bits, setBits] = useState(4);
  const [group, setGroup] = useState(64);
  const [scheme, setScheme] = useState<QuantScheme>('absmax');
  const [keep, setKeep] = useState(false);

  const result = useMemo(() => quantiseSlice(bits, group, scheme, keep), [bits, group, scheme, keep]);
  const rows = useMemo(
    () =>
      BITS.map((b) => [
        b,
        ...GROUPS.map((g) => quantiseSlice(b, g, scheme, keep).relativeError.toFixed(4)),
      ]),
    [scheme, keep],
  );

  const original = DIVERGING.light.mid;
  const stem = dark ? '#848c99' : '#9aa0a6';
  const recon = seriesColor(0, dark);
  const errColor = seriesColor(1, dark);
  const n = WEIGHT_SLICE.length;
  const innerW = W - PAD.left - PAD.right;
  const x = (i: number) => PAD.left + ((i + 0.5) / n) * innerW;
  const topH = 150;
  const y = (v: number) => PAD.top + topH / 2 - (Math.max(-CLIP, Math.min(CLIP, v)) / CLIP) * (topH / 2 - 4);
  const errTop = PAD.top + topH + 36;
  const errH = 70;
  const maxErr = Math.max(0.0001, ...result.recon.map((r, i) => (Math.abs(WEIGHT_SLICE[i]) > CLIP ? 0 : Math.abs(r - WEIGHT_SLICE[i]))));
  const ey = (v: number) => errTop + errH - (Math.min(v, maxErr) / maxErr) * errH;
  const outlierIndex = WEIGHT_SLICE.reduce((best, v, i) => (Math.abs(v) > Math.abs(WEIGHT_SLICE[best]) ? i : best), 0);

  return (
    <VizPanel
      title="Quantising one real row of SmolLM2 weights"
      hint="One weight, -3.22, is fifteen standard deviations out. With one scale for the whole row it forces a coarse grid on everything else and almost every ordinary weight rounds to zero. Smaller groups confine the damage to one group, and keeping the outlier in 16 bits removes it. Defaults match the chapter: 4 bits, groups of 64, relative error 0.1767."
      legend={[
        {label: 'original weight', color: stem},
        {label: 'dequantised weight', color: recon},
        {label: 'absolute error', color: errColor},
      ]}
      table={{columns: ['bits', 'one scale', 'groups of 128', 'groups of 64', 'groups of 32', 'groups of 16'], rows}}
      controls={
        <>
          <label className={s.control}>
            bits
            <input type="range" min={2} max={8} step={1} value={bits} onChange={(e) => setBits(Number(e.target.value))} />
            <span className={s.value}>{bits}</span>
          </label>
          <label className={s.control}>
            group size
            <select className={s.select} value={group} onChange={(e) => setGroup(Number(e.target.value))}>
              {GROUPS.map((g) => (
                <option key={g} value={g}>{g === 256 ? '256 (one scale)' : g}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            scheme
            <select className={s.select} value={scheme} onChange={(e) => setScheme(e.target.value as QuantScheme)}>
              <option value="absmax">absmax, symmetric</option>
              <option value="zeropoint">zero point, asymmetric</option>
            </select>
          </label>
          <label className={s.control}>
            outlier
            <select className={s.select} value={keep ? 'keep' : 'quantise'} onChange={(e) => setKeep(e.target.value === 'keep')}>
              <option value="quantise">quantise everything</option>
              <option value="keep">keep largest weight in 16 bits</option>
            </select>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Quantising 256 weights to ${bits} bits in groups of ${group}: relative error ${result.relativeError.toFixed(4)}, ${result.bitsPerWeight.toFixed(2)} bits per weight including scales.`}>
        <text className={s.axisLabel} x={PAD.left} y={14}>weights and their dequantised values (scale clipped to plus or minus {CLIP})</text>
        <rect x={PAD.left} y={PAD.top} width={innerW} height={topH} fill="none" stroke="var(--border-strong)" />
        <line className={s.axis} x1={PAD.left} y1={y(0)} x2={PAD.left + innerW} y2={y(0)} />
        {[-CLIP, 0, CLIP].map((t) => (
          <text key={t} className={s.tick} x={PAD.left - 6} y={y(t) + 3} textAnchor="end">{t}</text>
        ))}
        {group < n &&
          Array.from({length: n / group - 1}, (_, k) => (k + 1) * group).map((g) => (
            <line key={g} x1={PAD.left + (g / n) * innerW} y1={PAD.top} x2={PAD.left + (g / n) * innerW} y2={PAD.top + topH}
                  stroke="var(--border-strong)" strokeDasharray="2 3" opacity={0.6} />
          ))}
        {WEIGHT_SLICE.map((v, i) => (
          <g key={i}>
            <line x1={x(i)} y1={y(0)} x2={x(i)} y2={y(v)} stroke={stem} strokeWidth={1} opacity={0.7} />
            <circle cx={x(i)} cy={y(result.recon[i])} r={1.8} fill={recon} />
          </g>
        ))}
        <text className={s.dataLabel} x={x(outlierIndex) + 6} y={PAD.top + topH - 6}>
          weight {outlierIndex}: {WEIGHT_SLICE[outlierIndex]} (off scale) becomes {result.recon[outlierIndex].toFixed(3)}
        </text>
        <text className={s.axisLabel} x={PAD.left} y={errTop - 8}>absolute error per weight (outlier excluded from the scale)</text>
        <rect x={PAD.left} y={errTop} width={innerW} height={errH} fill="none" stroke="var(--border-strong)" />
        {WEIGHT_SLICE.map((v, i) => {
          const err = Math.abs(result.recon[i] - v);
          if (Math.abs(v) > CLIP) return null;
          return <line key={i} x1={x(i)} y1={errTop + errH} x2={x(i)} y2={ey(err)} stroke={errColor} strokeWidth={1.2} />;
        })}
        <text className={s.tick} x={PAD.left - 6} y={errTop + 8} textAnchor="end">{maxErr.toFixed(3)}</text>
        <text className={s.tick} x={PAD.left - 6} y={errTop + errH + 3} textAnchor="end">0</text>
        <text className={s.tick} x={PAD.left} y={errTop + errH + 16}>0</text>
        <text className={s.tick} x={PAD.left + innerW} y={errTop + errH + 16} textAnchor="end">255</text>
      </svg>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.4rem'}}>
        <span aria-live="polite">
          {bits} bits, groups of {group}, {scheme === 'absmax' ? 'absmax' : 'zero point'}{keep ? ', largest weight kept' : ''}: relative
          error {result.relativeError.toFixed(4)}, mean squared error {result.mse.toFixed(5)}, largest error {result.maxError.toFixed(3)},
          {' '}{result.bitsPerWeight.toFixed(2)} bits per weight including 16-bit scales.
        </span>
      </div>
    </VizPanel>
  );
}
