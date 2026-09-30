import {useState} from 'react';
import VizPanel, {vizStyles as s} from './VizPanel';
import {LinePlot, Slider} from './CourseLabShared';
import {hpaSimulation} from './courseSimulations';
import c from './CourseLab.module.css';

export default function HPALab() {
  const [users, setUsers] = useState(20);
  const [duration, setDuration] = useState(8);
  const rows = hpaSimulation(users, duration);
  const peakDesired = Math.max(...rows.map(row => row.desired));
  const peakPending = Math.max(...rows.map(row => row.pending));
  const peakFailure = Math.max(...rows.map(row => row.failures));
  return <VizPanel title="Autoscaling: desired pods versus serving pods"
    hint="Illustrative CPU-only simulation, not measured load-test data or a complete Kubernetes HPA implementation. Assumptions: 70% CPU target; 2–6 replicas; at most 4 serving pods fit; new pods take one minute. Scale up by at most 2 per minute, then wait five minutes below target before removing at most 1 per minute. Real behaviour also depends on memory, readiness and node capacity."
    controls={<><Slider label="Concurrent users" value={users} min={0} max={60} onChange={setUsers} />
      <div className={c.row}>{[10, 20, 50].map(value => <button key={value} type="button" className={s.button} aria-pressed={users === value} onClick={() => setUsers(value)}>{value} users</button>)}</div>
      <Slider label="Load duration (minutes)" value={duration} min={1} max={10} onChange={setDuration} /></>}
    table={{columns: ['Minute', 'Users', 'Desired pods', 'Serving pods', 'Pending pods', 'CPU per serving pod (%)', 'Illustrative failures (%)'], rows: rows.map(row => [row.minute, row.users, row.desired, row.running, row.pending, row.cpu.toFixed(1), (row.failures * 100).toFixed(1)])}}>
    <div className={c.chips} aria-live="polite">
      <span className={c.chip}>Peak desired: {peakDesired}</span><span className={c.chip}>Peak Pending: {peakPending}</span>
      <span className={c.chip}>Peak simulated failures: {(peakFailure * 100).toFixed(1)}%</span>
    </div>
    <LinePlot xValues={rows.map(row => row.minute)} xLabel="Minutes" yLabel="API pods" yMax={6} marker={duration}
      series={[{label: 'Desired replicas', values: rows.map(row => row.desired), step: true}, {label: 'Serving replicas', values: rows.map(row => row.running), step: true, dashed: true}, {label: 'Pending replicas', values: rows.map(row => row.pending), step: true}]} />
    <LinePlot xValues={rows.map(row => row.minute)} xLabel="Minutes" yLabel="CPU per serving pod (%)" yMax={Math.max(100, Math.ceil(Math.max(...rows.map(row => row.cpu)) / 50) * 50)} marker={duration}
      series={[{label: 'CPU utilisation', values: rows.map(row => row.cpu)}, {label: 'HPA target: 70%', values: rows.map(() => 70), dashed: true}]} />
    <p className={c.note}>CPU = users × 8.3% ÷ serving pods. Failure fraction = max(0, 1 − 100 ÷ CPU%). The vertical line marks the end of load.</p>
  </VizPanel>;
}
