import {useState} from 'react';

import {fmt} from './craftMath';
import {LATENESS, windowSummary} from './dmProjectsMath';
import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 260;
const PAD = {left: 56, right: 20, top: 36, bottom: 48};

export default function RestatementWindowLab() {
  const dark = useDarkViz();
  const [windowDays, setWindowDays] = useState(3);
  const summary = windowSummary(windowDays);
  const maxNet = Math.max(...LATENESS.slice(1).map((r) => r.netMinor));
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const slot = innerW / (LATENESS.length - 1);
  const held = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;
  const status = `window ${windowDays} days: ${fmt(summary.heldPayments)} payments held back, ${fmt(summary.heldNet)} minor units, ${(summary.heldShare * 100).toFixed(2)}% of the net`;

  return (
    <VizPanel
      title="How long should a day stay open?"
      hint="The default window of 3 days reproduces the chapter: 475 payments and 2,610,892 minor units held back. Move it to 6 and nothing is held back, but every day stays open for a week before finance can rely on it."
      legend={[
        {label: 'counted (restated if late)', color: seriesColor(0, dark)},
        {label: 'held back for approval', color: held},
      ]}
      table={{
        columns: ['days late', 'payments', 'net minor units', 'verdict'],
        rows: LATENESS.map((r) => [r.days, fmt(r.payments), fmt(r.netMinor), r.days > windowDays ? 'held back' : r.days === 0 ? 'counted at once' : 'counted, day restated']),
      }}
      controls={
        <>
          <label className={s.control}>
            restatement window (days)
            <input type="range" min={0} max={6} step={1} value={windowDays} onChange={(e) => setWindowDays(Number(e.target.value))} />
            <span className={s.value}>{windowDays}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Net money by days late. ${status}`}>
        {LATENESS.slice(1).map((r, i) => {
          const h = (r.netMinor / maxNet) * innerH;
          const x = PAD.left + i * slot + slot * 0.15;
          const isHeld = r.days > windowDays;
          return (
            <g key={r.days}>
              <rect x={x} y={PAD.top + innerH - h} width={slot * 0.7} height={h} fill={isHeld ? held : seriesColor(0, dark)} opacity={0.9} />
              <text className={s.tick} x={x + slot * 0.35} y={H - PAD.bottom + 18} textAnchor="middle">
                {r.days} d
              </text>
              <text className={s.dataLabel} x={x + slot * 0.35} y={PAD.top + innerH - h - 6} textAnchor="middle">
                {fmt(r.netMinor / 1000)}k
              </text>
            </g>
          );
        })}
        <text className={s.axisLabel} x={PAD.left} y={18}>
          net minor units that arrived late, by days late (day 0 is {fmt(LATENESS[0].netMinor / 1e6, 1)}M and is not drawn)
        </text>
        <text className={s.tick} x={PAD.left} y={H - 6}>
          restated after first publication: {fmt(summary.restatedPayments)} payments, {(summary.restatedShare * 100).toFixed(2)}% of the net
        </text>
      </svg>
    </VizPanel>
  );
}
