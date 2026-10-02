import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const METHODS: Record<string, number> = {'kill switch': 2, rollback: 15, hotfix: 120};
const PHASES = ['detect', 'acknowledge', 'decide', 'contain'];

const W = 640;
const H = 210;
const LEFT = 24;
const RIGHT = 24;

export default function IncidentTimelineLab() {
  const dark = useDarkViz();
  const [detect, setDetect] = useState(47);
  const [ack, setAck] = useState(5);
  const [decide, setDecide] = useState(18);
  const [method, setMethod] = useState('rollback');
  const [rate, setRate] = useState(12);

  const contain = METHODS[method];
  const parts = [detect, ack, decide, contain];
  const total = parts.reduce((a, b) => a + b, 0);
  const alt = detect + ack + decide + METHODS['kill switch'];
  const scaleMax = Math.max(total, alt, 60);
  const innerW = W - LEFT - RIGHT;
  const colours = [0, 1, 2, 3].map((i) => seriesColor(i, dark));

  const bar = (values: number[], y: number, h: number) => {
    let x = LEFT;
    return values.map((v, i) => {
      const w = (v / scaleMax) * innerW;
      const node = (
        <g key={PHASES[i]}>
          <rect x={x} y={y} width={Math.max(w, 1)} height={h} fill={colours[i]} />
          {w > 34 && h > 20 && (
            <text className={s.dataLabel} x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" fill="#fff">{v}</text>
          )}
        </g>
      );
      x += w;
      return node;
    });
  };

  const rows = PHASES.map((p, i) => [p, parts[i], `${((parts[i] / total) * 100).toFixed(1)}%`]);
  rows.push(['harm window', total, '100%']);

  return (
    <VizPanel
      title="Where the minutes go in an incident"
      hint="Defaults are the chapter's worked incident: 47 minutes to detect, 5 to acknowledge, 18 to decide and 15 to roll back, a harm window of 85 minutes and 1,020 harmful responses at 12 per minute. Containment minutes are model parameters: a kill switch saves only 13 minutes here, and detection is the larger share."
      legend={PHASES.map((p, i) => ({label: p, color: colours[i]}))}
      table={{columns: ['phase', 'minutes', 'share'], rows}}
      controls={
        <>
          <label className={s.control}>
            detect
            <input type="range" min={1} max={240} step={1} value={detect} onChange={(e) => setDetect(Number(e.target.value))} />
            <span className={s.value}>{detect}</span>
          </label>
          <label className={s.control}>
            acknowledge
            <input type="range" min={1} max={60} step={1} value={ack} onChange={(e) => setAck(Number(e.target.value))} />
            <span className={s.value}>{ack}</span>
          </label>
          <label className={s.control}>
            decide
            <input type="range" min={1} max={120} step={1} value={decide} onChange={(e) => setDecide(Number(e.target.value))} />
            <span className={s.value}>{decide}</span>
          </label>
          <label className={s.control}>
            containment
            <select className={s.select} value={method} onChange={(e) => setMethod(e.target.value)}>
              {Object.keys(METHODS).map((m) => (
                <option key={m} value={m}>{m} ({METHODS[m]} min)</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            harmful per minute
            <input type="range" min={1} max={60} step={1} value={rate} onChange={(e) => setRate(Number(e.target.value))} />
            <span className={s.value}>{rate}</span>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Incident timeline: harm window ${total} minutes, ${rate * total} harmful responses`}>
        <text className={s.axisLabel} x={LEFT} y={20}>this incident</text>
        {bar(parts, 30, 40)}
        <text className={s.dataLabel} x={LEFT} y={92}>
          harm window {total} min, {(rate * total).toLocaleString('en-GB')} harmful responses
        </text>
        <text className={s.axisLabel} x={LEFT} y={128}>same detection and decision, kill switch</text>
        {bar([detect, ack, decide, METHODS['kill switch']], 138, 22)}
        <text className={s.dataLabel} x={LEFT} y={184}>
          harm window {alt} min, {(rate * alt).toLocaleString('en-GB')} harmful responses
        </text>
      </svg>
    </VizPanel>
  );
}
