import {useState} from 'react';

import {samplingNodes} from './gnnMath';
import {HorizontalBars, SliderControl} from './labParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const TOTAL = 100000;

function formatCount(v: number) {
  return Math.round(v).toLocaleString('en-GB');
}

export default function NeighbourSamplingLab() {
  const dark = useDarkViz();
  const [batch, setBatch] = useState(64);
  const [layers, setLayers] = useState(2);
  const [f1, setF1] = useState(10);
  const [f2, setF2] = useState(5);
  const [f3, setF3] = useState(5);
  const [degree, setDegree] = useState(10);
  const [width, setWidth] = useState(16);
  const fanouts = [f1, f2, f3].slice(0, layers);
  const full = samplingNodes(batch, fanouts, degree, TOTAL, false);
  const sampled = samplingNodes(batch, fanouts, degree, TOTAL, true);
  const megabytes = (n: number) => (n * width * 4) / 1e6;

  const rows: (string | number)[][] = [
    ['Nodes loaded, all neighbours', formatCount(full)],
    ['Nodes loaded, sampled', formatCount(sampled)],
    ['Reduction', `${(full / sampled).toFixed(2)} times`],
    ['Feature memory, all neighbours (MB)', megabytes(full).toFixed(2)],
    ['Feature memory, sampled (MB)', megabytes(sampled).toFixed(2)],
    ['Share of the 100,000-node graph, sampled', `${((sampled / TOTAL) * 100).toFixed(1)}%`],
  ];

  return (
    <div data-testid="sampling-lab">
      <VizPanel
        title="How many nodes does one mini-batch drag in?"
        hint="The defaults are block 4: 64 target nodes, mean degree 10, two layers with fan-outs 10 and 5. Sampling loads 3,904 nodes where taking all neighbours would load 7,104 by the same formula. Real graphs with hubs load more, as the chapter shows."
        table={{columns: ['Quantity', 'Value'], rows}}
        legend={[
          {label: 'all neighbours', color: seriesColor(1, dark)},
          {label: 'sampled', color: seriesColor(0, dark)},
        ]}
        controls={
          <>
            <SliderControl label="Target nodes per batch" value={batch} min={1} max={512} step={1} onChange={setBatch} digits={0} />
            <SliderControl label="Layers" value={layers} min={1} max={3} step={1} onChange={setLayers} digits={0} />
            <SliderControl label="Fan-out, first hop" value={f1} min={1} max={50} step={1} onChange={setF1} digits={0} />
            <SliderControl label="Fan-out, second hop" value={f2} min={1} max={50} step={1} onChange={setF2} digits={0} />
            <SliderControl label="Fan-out, third hop" value={f3} min={1} max={50} step={1} onChange={setF3} digits={0} />
            <SliderControl label="Mean degree of the graph" value={degree} min={2} max={50} step={1} onChange={setDegree} digits={0} />
            <SliderControl label="Feature values per node" value={width} min={4} max={1024} step={4} onChange={setWidth} digits={0} />
            <span className={s.value} aria-live="polite" data-testid="sampling-summary">
              all neighbours {formatCount(full)} nodes, sampled {formatCount(sampled)} nodes
            </span>
          </>
        }>
        <HorizontalBars
          caption="nodes loaded for one batch (capped at the 100,000 nodes in the graph)"
          min={0}
          max={TOTAL}
          digits={0}
          bars={[
            {label: 'all neighbours', value: full, color: seriesColor(1, dark)},
            {label: 'sampled', value: sampled, color: seriesColor(0, dark)},
          ]}
        />
      </VizPanel>
    </div>
  );
}
