import {useState} from 'react';
import VizPanel from './VizPanel';
import {LinePlot, Slider} from './CourseLabShared';
import {forgettingSeries} from './courseSimulations';
import c from './CourseLab.module.css';

export default function ForgettingCurveLab() {
  const [halfLife, setHalfLife] = useState(24);
  const [boost, setBoost] = useState(0.3);
  const [threshold, setThreshold] = useState(0.1);
  const profile = forgettingSeries(halfLife, boost, threshold, true);
  const chat = forgettingSeries(halfLife, boost, threshold, false);
  const hours = profile.points.map(p => p.hour);
  return <VizPanel title="Forgetting: decay, reinforcement and pruning"
    hint="Strength halves every half-life. Access at hours 24 and 60 adds a boost, capped at 1. This hourly simulation permanently prunes a memory once its strength drops below the threshold; a later access cannot recover deleted content. The pinned constraint is exempt."
    controls={<><Slider label="Half-life (hours)" value={halfLife} min={6} max={96} step={6} onChange={setHalfLife} />
      <Slider label="Boost on access" value={boost} min={0} max={0.5} step={0.05} onChange={setBoost} />
      <Slider label="Pruning threshold" value={threshold} min={0.05} max={0.3} step={0.05} onChange={setThreshold} /></>}
    table={{columns: ['Hour', 'Pinned constraint', 'Profile fact', 'Chit-chat', 'Profile access scheduled?'], rows: hours.map((h, i) => [h, '1.000', profile.points[i].strength.toFixed(3), chat.points[i].strength.toFixed(3), profile.points[i].access ? 'Yes' : 'No'])}}>
    <LinePlot xValues={hours} xLabel="Hours elapsed" yLabel="Memory strength" yMax={1}
      series={[{label: 'A · pinned constraint', values: hours.map(() => 1)}, {label: 'B · profile fact, accessed twice', values: profile.points.map(p => p.strength)}, {label: 'C · unused chit-chat', values: chat.points.map(p => p.strength), dashed: true}]} />
    <div className={c.stack} aria-live="polite">
      <div className={c.chip}>A · pinned constraint: retained throughout</div>
      <div className={c.chip}>B · profile fact: {profile.prunedAt === null ? 'retained through hour 120' : `pruned at hour ${profile.prunedAt}`}</div>
      <div className={c.chip}>C · unused chit-chat: {chat.prunedAt === null ? 'retained through hour 120' : `pruned at hour ${chat.prunedAt}`}</div>
    </div>
    <p className={c.note}>Pruning threshold: {threshold.toFixed(2)}. Scheduled reinforcement: hours 24 and 60.</p>
  </VizPanel>;
}
