import {useState} from 'react';

import {Aggregator, TINY_EDGES, TINY_START, adjacency, propagate} from './gnnMath';
import {SliderControl} from './labParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const A = adjacency(TINY_EDGES, 5);
const POSITIONS: [number, number][] = [[90, 70], [90, 200], [230, 135], [370, 135], [500, 135]];
const MODES: {value: Aggregator; label: string}[] = [
  {value: 'gcn', label: 'GCN: normalised, with self-loop'},
  {value: 'sum', label: 'sum of neighbours'},
  {value: 'mean', label: 'mean of neighbours'},
  {value: 'max', label: 'max of neighbours'},
];

export default function MessagePassingLab() {
  const dark = useDarkViz();
  const [mode, setMode] = useState<Aggregator>('gcn');
  const [layers, setLayers] = useState(1);
  const history = propagate(A, TINY_START, mode, layers);
  const values = history[layers];
  const spread = Math.max(...values) - Math.min(...values);
  const colour = seriesColor(0, dark);
  const edge = dark ? '#8a93a6' : '#8a93a6';

  const rows: (string | number)[][] = history.map((h, k) => [`after ${k} layer${k === 1 ? '' : 's'}`, h.map((v) => v.toFixed(3)).join(', ')]);

  return (
    <div data-testid="mp-lab">
      <VizPanel
        title="Message passing on a five-node graph"
        hint="The defaults are block 1: starting values 1 to 5 and one GCN step give 1.866, 1.866, 2.771, 4.241, 4.133. Add layers and watch the values draw together."
        table={{columns: ['Step', 'Node values (node 0 to node 4)'], rows}}
        controls={
          <>
            <label className={s.control}>
              Aggregation
              <select className={s.select} value={mode} onChange={(e) => setMode(e.target.value as Aggregator)} aria-label="Aggregation">
                {MODES.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </label>
            <SliderControl label="Layers" value={layers} min={0} max={8} step={1} onChange={setLayers} digits={0} />
            <span className={s.value} aria-live="polite" data-testid="mp-summary">
              values {values.map((v) => v.toFixed(3)).join(', ')}; spread {spread.toFixed(3)}
            </span>
          </>
        }>
        <svg className={s.svg} viewBox="0 0 600 270" role="img" aria-label={`Five nodes with values ${values.map((v) => v.toFixed(2)).join(', ')}`}>
          {TINY_EDGES.map(([i, j]) => (
            <line key={`${i}-${j}`} x1={POSITIONS[i][0]} y1={POSITIONS[i][1]} x2={POSITIONS[j][0]} y2={POSITIONS[j][1]} stroke={edge} strokeWidth={2} />
          ))}
          {values.map((v, i) => (
            <g key={i}>
              <circle cx={POSITIONS[i][0]} cy={POSITIONS[i][1]} r={30} fill={colour} opacity={0.25 + 0.75 * Math.min(1, Math.abs(v) / 8)} stroke={colour} strokeWidth={2} />
              <text className={s.dataLabel} x={POSITIONS[i][0]} y={POSITIONS[i][1] + 5} textAnchor="middle">{v.toFixed(2)}</text>
              <text className={s.tick} x={POSITIONS[i][0]} y={POSITIONS[i][1] + 50} textAnchor="middle">node {i}</text>
            </g>
          ))}
        </svg>
      </VizPanel>
    </div>
  );
}
