import {useState} from 'react';

import {BURN_RULES, budgetEvent, fmt} from './craftMath';
import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 220;
const PAD = {left: 24, right: 24};
const SLOS = [0.99, 0.995, 0.999, 0.9995, 0.9999];
const SHARES = [0.01, 0.03, 0.08, 0.15, 1];

type EventState = {share: number; hours: number};

export default function ErrorBudgetLab() {
  const dark = useDarkViz();
  const [slo, setSlo] = useState(0.99);
  const [days, setDays] = useState(30);
  const [events, setEvents] = useState<EventState[]>([
    {share: 0.08, hours: 6},
    {share: 0.03, hours: 20},
    {share: 1, hours: 0.5},
  ]);

  const update = (i: number, patch: Partial<EventState>) => setEvents(events.map((e, k) => (k === i ? {...e, ...patch} : e)));
  const results = events.map((e) => budgetEvent(slo, days, e.share, e.hours));
  const total = results.reduce((a, r) => a + r.used, 0);
  const budgetMinutes = (1 - slo) * days * 24 * 60;
  const innerW = W - PAD.left - PAD.right;
  const scale = Math.max(1, total);
  const danger = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;

  let cursor = PAD.left;
  const segments = results.map((r, i) => {
    const w = (r.used / scale) * innerW;
    const seg = {x: cursor, w, i};
    cursor += w;
    return seg;
  });

  const rows = results.map((r, i) => [
    `event ${i + 1}: ${(events[i].share * 100).toFixed(0)}% bad for ${events[i].hours} h`,
    `${fmt(r.burn, 1)}x`,
    `${(r.used * 100).toFixed(1)}%`,
    r.fired.length ? r.fired.join(', ') : 'none',
  ]);
  const status = `budget ${fmt(budgetMinutes, 1)} full-outage minutes; used ${(total * 100).toFixed(1)}%; ${
    total > 1 ? 'budget exhausted' : `${((1 - total) * 100).toFixed(1)}% left`
  }`;

  return (
    <VizPanel
      title="Error budget: what each event costs and which alerts see it"
      hint="Defaults reproduce the chapter: 6.7%, 8.3% and 6.9% of a 99% quality budget, 21.9% together. The 8% regression burns at 8x and trips the 6x page; the 3% regression burns at 3x for 20 hours and trips nothing, though it costs more budget than either of the others. Stretch it to 36 hours and the ticket rule finally fires."
      legend={events.map((_, i) => ({label: `event ${i + 1}`, color: seriesColor(i, dark)}))}
      table={{columns: ['event', 'burn rate', 'budget used', 'rules that fire'], rows}}
      controls={
        <>
          <label className={s.control}>
            SLO
            <select className={s.select} value={slo} onChange={(e) => setSlo(Number(e.target.value))}>
              {SLOS.map((v) => (
                <option key={v} value={v}>
                  {(v * 100).toFixed(2)}%
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            window
            <select className={s.select} value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {[7, 28, 30].map((d) => (
                <option key={d} value={d}>
                  {d} days
                </option>
              ))}
            </select>
          </label>
          {events.map((e, i) => (
            <span key={i} className={s.control}>
              <label>
                event {i + 1} bad share
                <select className={s.select} value={e.share} onChange={(ev) => update(i, {share: Number(ev.target.value)})}>
                  {SHARES.map((v) => (
                    <option key={v} value={v}>
                      {v * 100}%
                    </option>
                  ))}
                </select>
              </label>
              <label>
                hours
                <input type="range" min={0.5} max={48} step={0.5} value={e.hours} onChange={(ev) => update(i, {hours: Number(ev.target.value)})} />
                <span className={s.value}>{e.hours}</span>
              </label>
            </span>
          ))}
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Stacked error budget use. ${status}`}>
        <rect x={PAD.left} y={50} width={innerW / scale} height={40} fill="none" stroke="var(--border-strong)" />
        {segments.map((seg) => (
          <rect key={seg.i} x={seg.x} y={50} width={seg.w} height={40} fill={seriesColor(seg.i, dark)} opacity={0.9} />
        ))}
        <line x1={PAD.left + innerW / scale} y1={40} x2={PAD.left + innerW / scale} y2={100} stroke={total > 1 ? danger : 'var(--text-strong)'} strokeWidth={2} strokeDasharray="4 3" />
        <text className={s.dataLabel} x={PAD.left + innerW / scale} y={32} textAnchor="end">
          100% of the budget
        </text>
        <text className={s.axisLabel} x={PAD.left} y={122}>
          used {(total * 100).toFixed(1)}% of {fmt(budgetMinutes, 1)} full-outage minutes over {days} days
        </text>
        {results.map((r, i) => (
          <text key={i} className={s.tick} x={PAD.left} y={148 + i * 20}>
            event {i + 1}: burn {fmt(r.burn, 1)}x, {(r.used * 100).toFixed(1)}% of budget, rules: {r.fired.length ? r.fired.join(' + ') : 'none'}
          </text>
        ))}
        <text className={s.tick} x={PAD.left} y={H - 6}>
          rules: {BURN_RULES.map((r) => r.name).join(', ')}
        </text>
      </svg>
    </VizPanel>
  );
}
