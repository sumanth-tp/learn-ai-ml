import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const N = 2000;
const AGENT_COST = 0.05;
const W = 640;
const H = 280;

export function mulberry32(seed: number) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type Ticket = {conf: number; correct: boolean; risky: boolean};

export function makeTickets(n = N, seed = 7): Ticket[] {
  const rnd = mulberry32(seed);
  const out: Ticket[] = [];
  for (let i = 0; i < n; i += 1) {
    const d = rnd();
    const r = rnd();
    const noise = rnd();
    const luck = rnd();
    const risky = r < 0.15;
    let p = 0.97 - 0.85 * d * d;
    const conf = Math.min(1, Math.max(0, p + (noise - 0.5) * 0.3));
    if (risky) p *= 0.6;
    out.push({conf, correct: luck < p, risky});
  }
  return out;
}

export function run(data: Ticket[], tau: number, escalateRisky: boolean, human: number, wrong: number) {
  let automated = 0;
  let right = 0;
  for (const t of data) {
    if (t.conf >= tau && !(escalateRisky && t.risky)) {
      automated += 1;
      if (t.correct) right += 1;
    }
  }
  const wrongN = automated - right;
  const escalated = data.length - automated;
  const cost = (data.length * AGENT_COST + (escalated + wrongN) * human + wrongN * wrong) / data.length;
  return {contained: automated / data.length, resolved: right / data.length, wrong: wrongN, cost};
}

const PAD = {top: 18, right: 14, bottom: 40, left: 40};
const PANEL_W = 290;

export default function EscalationPolicyLab() {
  const dark = useDarkViz();
  const [tau, setTau] = useState(0.6);
  const [escalateRisky, setEscalateRisky] = useState(true);
  const [human, setHuman] = useState(4);
  const [wrong, setWrong] = useState(15);

  const data = useMemo(() => makeTickets(), []);
  const taus = useMemo(() => Array.from({length: 21}, (_, i) => i / 20), []);
  const curve = useMemo(
    () => taus.map((t) => run(data, t, escalateRisky, human, wrong)),
    [data, taus, escalateRisky, human, wrong],
  );
  const now = run(data, tau, escalateRisky, human, wrong);

  const blue = seriesColor(0, dark);
  const orange = seriesColor(1, dark);
  const aqua = seriesColor(2, dark);
  const neg = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;

  const innerH = H - PAD.top - PAD.bottom;
  const innerW = PANEL_W - PAD.left - PAD.right;
  const costMax = Math.max(human * 1.3, ...curve.map((c) => c.cost)) * 1.05;
  const x = (panel: number, v: number) => (panel === 0 ? 0 : 320) + PAD.left + v * innerW;
  const yShare = (v: number) => PAD.top + innerH - v * innerH;
  const yCost = (v: number) => PAD.top + innerH - (v / costMax) * innerH;
  const path = (panel: number, ys: number[], fy: (v: number) => number) =>
    ys.map((v, i) => `${i ? 'L' : 'M'}${x(panel, taus[i]).toFixed(1)},${fy(v).toFixed(1)}`).join(' ');

  const best = curve.reduce((a, c, i) => (c.cost < curve[a].cost ? i : a), 0);

  const rows = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].map((t) => {
    const r = run(data, t, escalateRisky, human, wrong);
    return [t.toFixed(1), r.contained.toFixed(3), r.resolved.toFixed(3), r.wrong, r.cost.toFixed(3)];
  });

  const status = `threshold ${tau.toFixed(2)}: contained ${now.contained.toFixed(3)}, resolved by agent ${now.resolved.toFixed(3)}, ${now.wrong} wrong answers, cost per ticket ${now.cost.toFixed(3)}; cheapest threshold ${taus[best].toFixed(2)} at ${curve[best].cost.toFixed(3)}`;

  return (
    <VizPanel
      title="When should the agent hand over to a person?"
      hint="2,000 synthetic tickets, placeholder costs. Containment rises as the threshold falls, but the cost per ticket has a minimum: automating everything costs more than sending everything to people. Defaults (threshold 0.6, risky intents escalated, human 4, wrong answer 15) give contained 0.557, resolved 0.479, 156 wrong answers and 3.304 per ticket, the chapter's printed values."
      legend={[
        {label: 'contained by the agent', color: blue},
        {label: 'resolved correctly by the agent', color: aqua},
        {label: 'cost per ticket', color: orange},
        {label: 'all tickets to people', color: neg},
      ]}
      table={{columns: ['threshold', 'contained', 'resolved by agent', 'wrong answers', 'cost per ticket'], rows}}
      controls={
        <>
          <label className={s.control}>
            threshold
            <input type="range" min={0} max={1} step={0.05} value={tau} onChange={(e) => setTau(Number(e.target.value))} />
            <span className={s.value}>{tau.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={escalateRisky} onChange={(e) => setEscalateRisky(e.target.checked)} />
            risky intents always to a person
          </label>
          <label className={s.control}>
            human cost
            <input type="range" min={1} max={10} step={0.5} value={human} onChange={(e) => setHuman(Number(e.target.value))} />
            <span className={s.value}>{human.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            wrong answer
            <input type="range" min={0} max={40} step={1} value={wrong} onChange={(e) => setWrong(Number(e.target.value))} />
            <span className={s.value}>{wrong}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Containment and cost per ticket against the confidence threshold">
        {[0, 1].map((panel) => (
          <g key={panel}>
            <line className={s.axis} x1={x(panel, 0)} y1={yShare(0)} x2={x(panel, 1)} y2={yShare(0)} />
            <line className={s.axis} x1={x(panel, 0)} y1={yShare(0)} x2={x(panel, 0)} y2={yShare(1)} />
            {[0, 0.5, 1].map((v) => (
              <text key={v} className={s.tick} x={x(panel, v)} y={H - 22} textAnchor="middle">
                {v}
              </text>
            ))}
            <text className={s.axisLabel} x={x(panel, 0.5)} y={H - 6} textAnchor="middle">
              confidence threshold
            </text>
            <line x1={x(panel, tau)} y1={yShare(0)} x2={x(panel, tau)} y2={yShare(1)} stroke="var(--text-faint)" strokeDasharray="3 3" />
          </g>
        ))}
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line className={s.grid} x1={x(0, 0)} y1={yShare(v)} x2={x(0, 1)} y2={yShare(v)} />
            <text className={s.tick} x={x(0, 0) - 5} y={yShare(v) + 3} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {[0, 0.5, 1].map((f) => (
          <text key={f} className={s.tick} x={x(1, 0) - 5} y={yCost(f * costMax) + 3} textAnchor="end">
            {(f * costMax).toFixed(1)}
          </text>
        ))}
        <path d={path(0, curve.map((c) => c.contained), yShare)} fill="none" stroke={blue} strokeWidth={2.4} />
        <path d={path(0, curve.map((c) => c.resolved), yShare)} fill="none" stroke={aqua} strokeWidth={2.4} />
        <path d={path(1, curve.map((c) => c.cost), yCost)} fill="none" stroke={orange} strokeWidth={2.4} />
        <line x1={x(1, 0)} y1={yCost(human)} x2={x(1, 1)} y2={yCost(human)} stroke={neg} strokeDasharray="5 3" strokeWidth={1.6} />
        <circle cx={x(0, tau)} cy={yShare(now.contained)} r={4.5} fill={blue} stroke="var(--surface-raised)" strokeWidth={2} />
        <circle cx={x(0, tau)} cy={yShare(now.resolved)} r={4.5} fill={aqua} stroke="var(--surface-raised)" strokeWidth={2} />
        <circle cx={x(1, tau)} cy={yCost(now.cost)} r={4.5} fill={orange} stroke="var(--surface-raised)" strokeWidth={2} />
        <circle cx={x(1, taus[best])} cy={yCost(curve[best].cost)} r={3} fill="none" stroke="var(--text-strong)" strokeWidth={1.5} />
        <text className={s.axisLabel} x={x(0, 0)} y={12}>
          share of tickets
        </text>
        <text className={s.axisLabel} x={x(1, 0)} y={12}>
          cost per ticket (ring = cheapest)
        </text>
      </svg>
    </VizPanel>
  );
}
