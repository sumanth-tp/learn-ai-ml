import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Pod = {cpu: number; gpu: number; tolerates: boolean; kind: 'prep' | 'gpu'};
type NodeState = {name: string; cpu: number; gpu: number; usedCpu: number; usedGpu: number; pods: Pod[]};

const NODES = [
  {name: 'cpu-1', cpu: 8, gpu: 0},
  {name: 'gpu-a', cpu: 16, gpu: 2},
  {name: 'gpu-b', cpu: 16, gpu: 4},
];

export const PREP: Pod = {cpu: 3, gpu: 0, tolerates: false, kind: 'prep'};
export const TRAINER: Pod = {cpu: 4, gpu: 1, tolerates: true, kind: 'gpu'};

export type Strategy = 'spread' | 'pack';

export function place(prep: number, gpuPods: number, strategy: Strategy, tainted: boolean) {
  const nodes: NodeState[] = NODES.map((n) => ({...n, usedCpu: 0, usedGpu: 0, pods: []}));
  const pending: Pod[] = [];
  const queue: Pod[] = [...Array(prep).fill(PREP), ...Array(gpuPods).fill(TRAINER)];
  for (const pod of queue) {
    const candidates = nodes.filter((n) => {
      const room = n.cpu - n.usedCpu >= pod.cpu && n.gpu - n.usedGpu >= pod.gpu;
      const allowed = pod.tolerates || !(tainted && n.gpu > 0);
      return room && allowed;
    });
    if (candidates.length === 0) {
      pending.push(pod);
      continue;
    }
    const sign = strategy === 'pack' ? -1 : 1;
    const score = (n: NodeState) => (sign * (n.usedCpu + pod.cpu)) / n.cpu;
    candidates.sort((a, b) => score(a) - score(b) || a.name.localeCompare(b.name));
    const best = candidates[0];
    best.usedCpu += pod.cpu;
    best.usedGpu += pod.gpu;
    best.pods.push(pod);
  }
  return {nodes, pending};
}

export function desiredReplicas(ready: number, utilisationPct: number, targetPct: number, lo = 2, hi = 12, tolerance = 0.1) {
  const ratio = utilisationPct / targetPct;
  if (Math.abs(ratio - 1) <= tolerance) return ready;
  return Math.min(hi, Math.max(lo, Math.ceil(ready * ratio)));
}

const W = 640;
const H = 300;

export default function ReplicaSchedulerLab() {
  const dark = useDarkViz();
  const [prep, setPrep] = useState(5);
  const [gpuPods, setGpuPods] = useState(6);
  const [strategy, setStrategy] = useState<Strategy>('spread');
  const [tainted, setTainted] = useState(false);
  const [ready, setReady] = useState(2);
  const [util, setUtil] = useState(315);
  const [target, setTarget] = useState(70);

  const {nodes, pending} = useMemo(() => place(prep, gpuPods, strategy, tainted), [prep, gpuPods, strategy, tainted]);
  const placedGpu = nodes.reduce((a, n) => a + n.pods.filter((p) => p.kind === 'gpu').length, 0);
  const placedPrep = nodes.reduce((a, n) => a + n.pods.filter((p) => p.kind === 'prep').length, 0);
  const idle = nodes.reduce((a, n) => a + n.gpu - n.usedGpu, 0);
  const pendingGpu = pending.filter((p) => p.kind === 'gpu').length;
  const pendingPrep = pending.length - pendingGpu;
  const want = desiredReplicas(ready, util, target);

  const prepColor = seriesColor(0, dark);
  const gpuColor = seriesColor(1, dark);
  const track = dark ? '#2b3340' : '#e3e6ea';
  const colW = (W - 40) / 3;

  const status = `GPU pods placed ${placedGpu} of ${gpuPods}, Pending ${pendingGpu}, GPUs idle ${idle}; preprocessing placed ${placedPrep} of ${prep}, Pending ${pendingPrep}`;

  return (
    <VizPanel
      title="Where replicas land, and what the autoscaler asks for"
      hint="Put preprocessing pods and GPU pods on a small cluster. Without a taint, 3 CPU pods wander onto the GPU nodes and use up the CPU that GPU pods also need, so 2 GPUs sit idle while 2 pods stay Pending. Taint the GPU nodes and all 6 GPU pods fit, but 3 preprocessing pods wait for CPU nodes. Defaults match block 2 and block 3 of the chapter."
      legend={[
        {label: 'preprocessing pod (3 CPU)', color: prepColor},
        {label: 'GPU pod (4 CPU, 1 GPU)', color: gpuColor},
      ]}
      table={{
        columns: ['node', 'CPU used / capacity', 'GPU used / capacity', 'pods'],
        rows: nodes.map((n) => [n.name, `${n.usedCpu} / ${n.cpu}`, `${n.usedGpu} / ${n.gpu}`, n.pods.length]),
      }}
      controls={
        <>
          <label className={s.control}>
            preprocessing pods
            <input type="range" min={0} max={8} step={1} value={prep} onChange={(e) => setPrep(Number(e.target.value))} />
            <span className={s.value}>{prep}</span>
          </label>
          <label className={s.control}>
            GPU pods
            <input type="range" min={0} max={10} step={1} value={gpuPods} onChange={(e) => setGpuPods(Number(e.target.value))} />
            <span className={s.value}>{gpuPods}</span>
          </label>
          <label className={s.control}>
            strategy
            <select className={s.select} value={strategy} onChange={(e) => setStrategy(e.target.value as Strategy)}>
              <option value="spread">spread (lowest CPU utilisation first)</option>
              <option value="pack">bin-pack (highest first)</option>
            </select>
          </label>
          <label className={s.control}>
            GPU nodes tainted
            <select className={s.select} value={tainted ? 'yes' : 'no'} onChange={(e) => setTainted(e.target.value === 'yes')}>
              <option value="no">no</option>
              <option value="yes">yes</option>
            </select>
          </label>
          <label className={s.control}>
            ready replicas
            <input type="range" min={1} max={12} step={1} value={ready} onChange={(e) => setReady(Number(e.target.value))} />
            <span className={s.value}>{ready}</span>
          </label>
          <label className={s.control}>
            CPU utilisation %
            <input type="range" min={10} max={400} step={5} value={util} onChange={(e) => setUtil(Number(e.target.value))} />
            <span className={s.value}>{util}</span>
          </label>
          <label className={s.control}>
            target %
            <input type="range" min={30} max={90} step={5} value={target} onChange={(e) => setTarget(Number(e.target.value))} />
            <span className={s.value}>{target}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {status}. Autoscaler asks for {want} replicas (min 2, max 12).
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={status}>
        {nodes.map((n, i) => {
          const x0 = 20 + i * colW;
          const bar = colW - 28;
          return (
            <g key={n.name}>
              <rect x={x0} y={14} width={colW - 12} height={196} rx={8} fill="none" stroke="var(--text-muted, #888)" strokeDasharray="5 4" />
              <text className={s.dataLabel} x={x0 + 10} y={34}>
                {n.name}
              </text>
              <rect x={x0 + 10} y={44} width={bar} height={8} rx={4} fill={track} />
              <rect x={x0 + 10} y={44} width={(bar * n.usedCpu) / n.cpu} height={8} rx={4} fill={prepColor} />
              <text className={s.tick} x={x0 + 10} y={68}>
                CPU {n.usedCpu}/{n.cpu}
              </text>
              {n.gpu > 0 && (
                <>
                  <rect x={x0 + 10} y={76} width={bar} height={8} rx={4} fill={track} />
                  <rect x={x0 + 10} y={76} width={(bar * n.usedGpu) / n.gpu} height={8} rx={4} fill={gpuColor} />
                </>
              )}
              <text className={s.tick} x={x0 + 10} y={100}>
                GPU {n.usedGpu}/{n.gpu}
              </text>
              {n.pods.map((p, k) => (
                <rect
                  key={k}
                  x={x0 + 10 + (k % 8) * 24}
                  y={112 + Math.floor(k / 8) * 24}
                  width={20}
                  height={20}
                  rx={4}
                  fill={p.kind === 'prep' ? prepColor : gpuColor}
                />
              ))}
            </g>
          );
        })}
        <text className={s.dataLabel} x={20} y={236}>
          Pending: {pending.length}
        </text>
        {pending.map((p, k) => (
          <rect
            key={k}
            x={20 + (k % 20) * 24}
            y={246 + Math.floor(k / 20) * 24}
            width={20}
            height={20}
            rx={4}
            fill="none"
            stroke={p.kind === 'prep' ? prepColor : gpuColor}
            strokeWidth={2}
            strokeDasharray="3 2"
          />
        ))}
      </svg>
    </VizPanel>
  );
}
