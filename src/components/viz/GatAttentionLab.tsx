import {useState} from 'react';

import {NODE2_GCN_WEIGHTS, gatWeights} from './gnnMath';
import {HorizontalBars, SliderControl} from './labParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const NAMES = ['itself', 'node 0', 'node 1', 'node 3'];

export default function GatAttentionLab() {
  const dark = useDarkViz();
  const [scores, setScores] = useState([2, 0.5, -1, 1]);
  const [slope, setSlope] = useState(0.2);
  const alpha = gatWeights(scores, slope);
  const set = (i: number) => (v: number) => setScores(scores.map((old, k) => (k === i ? v : old)));
  const largest = alpha.indexOf(Math.max(...alpha));

  const rows: (string | number)[][] = NAMES.map((name, i) => [
    name,
    scores[i].toFixed(2),
    alpha[i].toFixed(4),
    NODE2_GCN_WEIGHTS[i].toFixed(3),
    i === 0 ? 'own matrix' : (1 / 3).toFixed(3),
  ]);

  return (
    <div data-testid="gat-lab">
      <VizPanel
        title="One node deciding how much to listen to each input"
        hint="The defaults are block 3: raw scores 2.0, 0.5, -1.0 and 1.0 give attention weights 0.5876, 0.1311, 0.0651 and 0.2162. GCN would use fixed weights 0.25 and 0.289; GraphSAGE would give each neighbour one third."
        table={{columns: ['Input', 'Raw score', 'GAT weight', 'GCN weight', 'GraphSAGE weight'], rows}}
        legend={[{label: 'GAT attention weight', color: seriesColor(0, dark)}]}
        controls={
          <>
            {NAMES.map((name, i) => (
              <SliderControl key={name} label={`Raw score for ${name}`} value={scores[i]} min={-3} max={3} step={0.5} onChange={set(i)} digits={1} />
            ))}
            <SliderControl label="LeakyReLU slope" value={slope} min={0} max={1} step={0.05} onChange={setSlope} />
            <span className={s.value} aria-live="polite" data-testid="gat-summary">
              weights {alpha.map((v) => v.toFixed(4)).join(', ')}; most attention goes to {NAMES[largest]}
            </span>
          </>
        }>
        <HorizontalBars
          caption="attention weight (they always sum to 1)"
          min={0}
          max={1}
          bars={NAMES.map((name, i) => ({label: name, value: alpha[i], color: seriesColor(0, dark)}))}
          digits={4}
        />
      </VizPanel>
    </div>
  );
}
