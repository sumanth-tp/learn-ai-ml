import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Attrs = Record<string, string | number | boolean>;
type Resource = {type: string; attrs: Attrs; ignore?: string[]; keep?: boolean};
type Action = {kind: 'create' | 'update' | 'destroy' | 'replace'; addr: string; changed: [string, string, string][]};

const IMMUTABLE: Record<string, string[]> = {network: ['cidr'], bucket: ['name'], cluster: [], endpoint: [], alarm: []};

const STATE: Record<string, Resource> = {
  'network.main': {type: 'network', attrs: {cidr: '10.0.0.0/16'}},
  'bucket.artifacts': {type: 'bucket', attrs: {name: 'ml-artifacts-prod', versioning: true}},
  'cluster.gpu': {type: 'cluster', attrs: {network: '10.0.0.0/16', node_count: 2, machine_type: 'gpu-small'}},
  'endpoint.ranker': {
    type: 'endpoint',
    attrs: {cluster: 'gpu-small', bucket_name: 'ml-artifacts-prod', replicas: 3, image: 'ranker:1.0'},
  },
};

export function plan(want: Record<string, Resource>, state: Record<string, Resource>): Action[] {
  const actions: Action[] = [];
  for (const [addr, res] of Object.entries(want)) {
    const current = state[addr];
    if (!current) {
      actions.push({kind: 'create', addr, changed: []});
      continue;
    }
    const changed: [string, string, string][] = [];
    for (const [key, value] of Object.entries(res.attrs)) {
      if ((res.ignore ?? []).includes(key)) continue;
      if (current.attrs[key] !== value) changed.push([key, String(current.attrs[key]), String(value)]);
    }
    if (changed.length === 0) continue;
    const forced = changed.some(([key]) => IMMUTABLE[res.type].includes(key));
    if (forced && res.keep) throw new Error(`${addr}: prevent_destroy blocks the replacement`);
    actions.push({kind: forced ? 'replace' : 'update', addr, changed});
  }
  for (const addr of Object.keys(state)) {
    if (!want[addr]) actions.push({kind: 'destroy', addr, changed: []});
  }
  return actions;
}

export function summary(actions: Action[]): string {
  if (actions.length === 0) return 'No changes.';
  const add = actions.filter((a) => a.kind === 'create' || a.kind === 'replace').length;
  const change = actions.filter((a) => a.kind === 'update').length;
  const destroy = actions.filter((a) => a.kind === 'destroy' || a.kind === 'replace').length;
  return `Plan: ${add} to add, ${change} to change, ${destroy} to destroy.`;
}

const SYMBOL = {create: '+', update: '~', destroy: '-', replace: '-/+'};
const W = 640;
const ROW = 30;

export default function PlanDiffLab() {
  const dark = useDarkViz();
  const [replicas, setReplicas] = useState(5);
  const [image, setImage] = useState('ranker:1.1');
  const [bucket, setBucket] = useState('ml-artifacts-prod-eu');
  const [alarm, setAlarm] = useState(true);
  const [keep, setKeep] = useState(false);
  const [drift, setDrift] = useState(false);
  const [ignore, setIgnore] = useState(false);

  const result = useMemo(() => {
    const state: Record<string, Resource> = JSON.parse(JSON.stringify(STATE));
    if (drift) state['cluster.gpu'].attrs.node_count = 4;
    const want: Record<string, Resource> = {
      'network.main': {type: 'network', attrs: {cidr: '10.0.0.0/16'}},
      'bucket.artifacts': {type: 'bucket', attrs: {name: bucket, versioning: true}, keep},
      'cluster.gpu': {
        type: 'cluster',
        attrs: {network: '10.0.0.0/16', node_count: 2, machine_type: 'gpu-small'},
        ignore: ignore ? ['node_count'] : [],
      },
      'endpoint.ranker': {
        type: 'endpoint',
        attrs: {cluster: 'gpu-small', bucket_name: bucket, replicas, image},
      },
    };
    if (alarm) want['alarm.latency'] = {type: 'alarm', attrs: {endpoint: image, threshold_ms: 200}};
    try {
      return {actions: plan(want, state), error: null as string | null};
    } catch (e) {
      return {actions: [] as Action[], error: (e as Error).message};
    }
  }, [replicas, image, bucket, alarm, keep, drift, ignore]);

  const colour = (kind: Action['kind']) =>
    kind === 'create' ? DIVERGING[dark ? 'dark' : 'light'].positive
    : kind === 'destroy' ? DIVERGING[dark ? 'dark' : 'light'].negative
    : kind === 'replace' ? seriesColor(3, dark)
    : seriesColor(4, dark);

  const line = result.error ?? summary(result.actions);
  const lines = result.actions.flatMap((a) =>
    a.changed.length === 0
      ? [{kind: a.kind, addr: a.addr, text: '', first: true}]
      : a.changed.map(([k, f, t], i) => ({kind: a.kind, addr: a.addr, text: `${k}: ${f} to ${t}`, first: i === 0})),
  );
  const height = Math.max(120, 50 + lines.length * ROW + 40);

  const rows = result.actions.flatMap((a) =>
    a.changed.length === 0
      ? [[a.kind, a.addr, '-', '-', '-']]
      : a.changed.map(([key, from, to]) => [a.kind, a.addr, key, from, to]),
  );

  return (
    <VizPanel
      title="Plan: the difference between code and state"
      hint="Change the code and watch the plan. Renaming the bucket forces a replacement (-/+) because the name is immutable and also updates the endpoint that points at it. Tick the console edit to see drift: the plan wants to put node_count back to 2, unless ignore_changes is on. Defaults match block 1 of the chapter: Plan: 2 to add, 1 to change, 1 to destroy."
      legend={[
        {label: '+ add', color: DIVERGING[dark ? 'dark' : 'light'].positive},
        {label: '~ change', color: seriesColor(4, dark)},
        {label: '-/+ replace', color: seriesColor(3, dark)},
        {label: '- destroy', color: DIVERGING[dark ? 'dark' : 'light'].negative},
      ]}
      table={{columns: ['action', 'address', 'attribute', 'from', 'to'], rows}}
      controls={
        <>
          <label className={s.control}>
            endpoint replicas
            <input type="range" min={1} max={8} step={1} value={replicas} onChange={(e) => setReplicas(Number(e.target.value))} />
            <span className={s.value}>{replicas}</span>
          </label>
          <label className={s.control}>
            image
            <select className={s.select} value={image} onChange={(e) => setImage(e.target.value)}>
              <option value="ranker:1.0">ranker:1.0</option>
              <option value="ranker:1.1">ranker:1.1</option>
            </select>
          </label>
          <label className={s.control}>
            bucket name
            <select className={s.select} value={bucket} onChange={(e) => setBucket(e.target.value)}>
              <option value="ml-artifacts-prod">ml-artifacts-prod</option>
              <option value="ml-artifacts-prod-eu">ml-artifacts-prod-eu</option>
              <option value="ml-artifacts-prod-us">ml-artifacts-prod-us</option>
            </select>
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={alarm} onChange={(e) => setAlarm(e.target.checked)} /> add latency alarm
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={keep} onChange={(e) => setKeep(e.target.checked)} /> prevent_destroy on bucket
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={drift} onChange={(e) => setDrift(e.target.checked)} /> node_count set to 4 in console
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={ignore} onChange={(e) => setIgnore(e.target.checked)} /> ignore_changes on node_count
          </label>
          <span className={s.value} aria-live="polite">
            {line}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`Terraform-style plan. ${line}`}>
        {lines.map((l, i) => (
          <g key={`${l.addr}-${i}`}>
            {l.first && (
              <>
                <text className={s.dataLabel} x={20} y={34 + i * ROW} fill={colour(l.kind)}>
                  {SYMBOL[l.kind]}
                </text>
                <text className={s.tick} x={64} y={34 + i * ROW}>
                  {l.addr}
                </text>
              </>
            )}
            <text className={s.tick} x={200} y={34 + i * ROW}>
              {l.text}
            </text>
          </g>
        ))}
        <text className={s.dataLabel} x={20} y={height - 20} fill={result.error ? DIVERGING[dark ? 'dark' : 'light'].negative : undefined}>
          {result.error ? `Error: ${line}` : line}
        </text>
      </svg>
    </VizPanel>
  );
}
