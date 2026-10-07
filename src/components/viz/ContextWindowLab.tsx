import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const SERIES = {"mask|2|10000":{"size":[1000,1814,2250,1847,1645,1896,1974,2186,1722,1347,1391,2713,3504,2813,2459,2186,1933,2301,3026,3635,3258,2295,1742,2775],"hit":[0,1000,27,49,71,102,124,147,179,202,224,246,268,290,324,347,369,392,414,436,464,487,510,533],"kept":4},"batch|2|10000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,1722,2123,2874,4575,6095,7071,2459,3140,4007,5034,6604,3635,4805,5427,6021,7654],"hit":[0,1000,1814,3208,3597,4758,5376,6592,27,1722,2123,2874,4575,6095,202,2459,3140,4007,5034,347,3635,4805,5427,6021],"kept":4},"summary|2|10000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,1577,1978,2729,4430,5950,6926,2185,2866,3733,4760,6330,7938,2854,3476,4070,5703],"hit":[0,1000,1814,3208,3597,4758,5376,6592,20,1577,1978,2729,4430,5950,49,2185,2866,3733,4760,6330,65,2854,3476,4070],"kept":4},"mask|4|10000":{"size":[1000,1814,3208,3597,3800,3626,3479,3910,3501,3316,2874,3799,4612,5209,5623,4637,4007,4080,4530,5479,5804,5427,4474,4522],"hit":[0,1000,1814,3208,27,49,71,102,124,147,179,202,224,246,268,290,324,347,369,392,414,436,464,487],"kept":4},"batch|4|10000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,3501,3902,4653,6354,7874,5209,6352,7033,7900,4080,5650,7258,5804,6426,7020,4522],"hit":[0,1000,1814,3208,3597,4758,5376,6592,27,3501,3902,4653,6354,147,5209,6352,7033,268,4080,5650,369,5804,6426,436],"kept":4},"summary|4|10000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,3396,3797,4548,6249,7769,4998,6141,6822,7689,3784,5354,6962,5441,6063,6657,4095],"hit":[0,1000,1814,3208,3597,4758,5376,6592,20,3396,3797,4548,6249,34,4998,6141,6822,49,3784,5354,71,5441,6063,65],"kept":4},"mask|6|10000":{"size":[1000,1814,3208,3597,4758,5376,5634,5640,5006,5040,4653,5768,6095,6295,6731,7033,7171,6531,6604,7258,7308,7271,7020,7654],"hit":[0,1000,1814,3208,3597,4758,27,49,71,102,124,147,179,202,224,246,268,290,324,347,369,392,414,436],"kept":4},"batch|6|10000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,5006,5407,6158,7859,6095,7071,6731,7412,7171,6531,6604,7258,7308,7930,7020,7654],"hit":[0,1000,1814,3208,3597,4758,5376,6592,27,5006,5407,6158,102,6095,202,6731,246,290,324,347,369,7308,392,436],"kept":4},"summary|6|10000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,4946,5347,6098,7799,5950,6926,6542,7223,6938,7965,6330,7938,6989,7611,6657,7273],"hit":[0,1000,1814,3208,3597,4758,5376,6592,20,4946,5347,6098,34,5950,55,6542,55,6938,49,6330,71,6989,71,65],"kept":4},"mask|8|10000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,7161,6770,6158,7492,7874,8264,8214,8119,8279,8927,9768,9709,9382,9050,8524,9498],"hit":[0,1000,1814,3208,3597,4758,5376,6592,27,49,71,102,124,147,179,202,224,246,268,290,324,347,369,392],"kept":4},"batch|8|10000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,7161,7562,6158,7859,7874,8264,8214,8119,8279,8927,9768,9709,9382,9050,8524,9498],"hit":[0,1000,1814,3208,3597,4758,5376,6592,27,7161,49,6158,102,147,179,202,224,246,268,290,324,347,369,392],"kept":4},"summary|8|10000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,7142,7543,6098,7799,7769,8142,8069,7952,8090,8716,9535,9458,9108,8754,8205,9157],"hit":[0,1000,1814,3208,3597,4758,5376,6592,20,7142,22,6098,40,34,55,55,55,55,55,49,71,71,71,71],"kept":4},"slide|0|6000":{"size":[1000,1814,3208,3597,4758,5376,5612,5596,4931,5332,5694,5616,5920,5369,5360,4340,5207,4714,5308,5773,5395,4990,5584,5647],"hit":[0,1000,1814,3208,3597,4758,25,25,25,4931,25,25,25,25,25,25,4340,25,25,25,25,25,4990,25],"kept":0},"slide|0|10000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,8119,8520,9271,9992,9304,9891,9873,9936,9587,9816,9505,9412,9062,9684,9302,9792],"hit":[0,1000,1814,3208,3597,4758,5376,6592,7390,8119,8520,25,25,25,25,25,25,25,25,25,25,9062,25,25],"kept":1},"slide|0|16000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,8119,8520,9271,10972,12492,13468,14611,15292,15179,15392,15568,15626,14962,15584,15380,15883],"hit":[0,1000,1814,3208,3597,4758,5376,6592,7390,8119,8520,9271,10972,12492,13468,14611,25,25,25,25,25,14962,25,25],"kept":3},"full|0|10000":{"size":[1000,1814,3208,3597,4758,5376,6592,7390,8119,8520,9271,10972,12492,13468,14611,15292,16159,17186,18756,20364,21534,22156,22750,24383],"hit":[0,1000,1814,3208,3597,4758,5376,6592,7390,8119,8520,9271,10972,12492,13468,14611,15292,16159,17186,18756,20364,21534,22156,22750],"kept":6},"clock|0|10000":{"size":[1017,1831,3225,3614,4775,5393,6609,7407,8136,8537,9288,10989,12509,13485,14628,15309,16176,17203,18773,20381,21551,22173,22767,24400],"hit":[0,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13],"kept":6}} as Record<string, {size: number[]; hit: number[]; kept: number}>;

type Strategy = 'full' | 'clock' | 'slide' | 'mask' | 'batch' | 'summary';

const LABELS: Record<Strategy, string> = {
  full: 'Keep everything (append only)',
  clock: 'Keep everything, clock at the top',
  slide: 'Sliding window (drop oldest steps)',
  mask: 'Mask old tool results, every step',
  batch: 'Mask old tool results, in batches',
  summary: 'Summarise old steps, in batches',
};

const W = 640;
const H = 300;
const LEFT = 56;
const RIGHT = 14;
const TOP = 26;
const BOTTOM = 40;
const Y_MAX = 26000;
const STEPS = 24;

const keyFor = (strategy: Strategy, keep: number, windowSize: number) => {
  if (strategy === 'full' || strategy === 'clock') return `${strategy}|0|10000`;
  if (strategy === 'slide') return `slide|0|${windowSize}`;
  return `${strategy}|${keep}|10000`;
};

export default function ContextWindowLab() {
  const dark = useDarkViz();
  const [strategy, setStrategy] = useState<Strategy>('batch');
  const [keep, setKeep] = useState(4);
  const [windowSize, setWindowSize] = useState(10000);
  const [read, setRead] = useState(0.1);
  const [write, setWrite] = useState(1.25);

  const run = SERIES[keyFor(strategy, keep, windowSize)];
  const peak = Math.max(...run.size);
  const billed = run.size.reduce((a, b) => a + b, 0);
  const hits = run.hit.reduce((a, b) => a + b, 0);
  const over = run.size.filter((v) => v > windowSize).length;
  const cost = run.size.reduce((acc, size, i) => acc + run.hit[i] * read + (size - run.hit[i]) * write, 0);

  const x = (i: number) => LEFT + ((i + 0.5) / STEPS) * (W - LEFT - RIGHT);
  const barW = ((W - LEFT - RIGHT) / STEPS) * 0.72;
  const y = (v: number) => H - BOTTOM - (Math.min(v, Y_MAX) / Y_MAX) * (H - TOP - BOTTOM);
  const hitColour = seriesColor(0, dark);
  const newColour = seriesColor(1, dark);
  const limit = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;

  const rows = run.size.map((size, i) => [i + 1, size, run.hit[i], size - run.hit[i], size > windowSize ? 'over' : '-']);
  const usesKeep = strategy === 'mask' || strategy === 'batch' || strategy === 'summary';

  return (
    <VizPanel
      title="One 24-step agent run, six ways to manage the window"
      hint="Each bar is one request: the blue part was found in the prefix cache, the orange part had to be written. The default, batch masking with the last 4 results kept, reproduces the printed row: peak 7,900 tokens, 127,273 billed, 4 of 6 facts, 68.0% cache hits, cost 59,552 units. Switch to the clock layout to see the cache collapse, or to the sliding window to see facts disappear."
      legend={[
        {label: 'tokens read from the prefix cache', color: hitColour},
        {label: 'tokens written fresh', color: newColour},
        {label: 'window limit', color: limit},
      ]}
      table={{columns: ['step', 'tokens sent', 'from cache', 'written', 'vs window'], rows}}
      controls={
        <>
          <label className={s.control}>
            strategy
            <select className={s.select} value={strategy} onChange={(e) => setStrategy(e.target.value as Strategy)}>
              {(Object.keys(LABELS) as Strategy[]).map((key) => (
                <option key={key} value={key}>
                  {LABELS[key]}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            recent steps kept
            <select className={s.select} value={keep} disabled={!usesKeep} onChange={(e) => setKeep(Number(e.target.value))}>
              {[2, 4, 6, 8].map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            window
            <select className={s.select} value={windowSize} onChange={(e) => setWindowSize(Number(e.target.value))}>
              {[6000, 10000, 16000].map((w) => (
                <option key={w} value={w}>
                  {w.toLocaleString('en-GB')}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            cache read price
            <input type="range" min={0.05} max={0.5} step={0.05} value={read} onChange={(e) => setRead(Number(e.target.value))} />
            <span className={s.value}>{read.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            cache write price
            <input type="range" min={1} max={2} step={0.05} value={write} onChange={(e) => setWrite(Number(e.target.value))} />
            <span className={s.value}>{write.toFixed(2)}</span>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Tokens sent in each of 24 requests under ${LABELS[strategy]}. Peak ${peak}, billed ${billed}, cache hit share ${((hits / billed) * 100).toFixed(1)} percent.`}>
        {[0, 6500, 13000, 19500, 26000].map((tick) => (
          <g key={tick}>
            <line className={s.grid} x1={LEFT} x2={W - RIGHT} y1={y(tick)} y2={y(tick)} />
            <text className={s.tick} x={LEFT - 6} y={y(tick) + 3} textAnchor="end">
              {tick.toLocaleString('en-GB')}
            </text>
          </g>
        ))}
        {run.size.map((size, i) => (
          <g key={i}>
            <rect x={x(i) - barW / 2} y={y(run.hit[i])} width={barW} height={Math.max(0, y(0) - y(run.hit[i]))} fill={hitColour} />
            <rect x={x(i) - barW / 2} y={y(size)} width={barW} height={Math.max(0, y(run.hit[i]) - y(size))} fill={newColour} />
            {(i === 0 || (i + 1) % 4 === 0) && (
              <text className={s.tick} x={x(i)} y={H - BOTTOM + 14} textAnchor="middle">
                {i + 1}
              </text>
            )}
          </g>
        ))}
        <line x1={LEFT} x2={W - RIGHT} y1={y(windowSize)} y2={y(windowSize)} stroke={limit} strokeWidth={1.6} strokeDasharray="6 4" />
        <text className={s.tick} x={W - RIGHT} y={y(windowSize) - 4} textAnchor="end" fill={limit}>
          window {windowSize.toLocaleString('en-GB')}
        </text>
        <text className={s.axisLabel} x={LEFT + (W - LEFT - RIGHT) / 2} y={H - 6} textAnchor="middle">
          request number (agent step)
        </text>
        <text className={s.axisLabel} x={LEFT} y={14}>
          tokens sent
        </text>
      </svg>
      <p className={s.hint} style={{padding: '0.4rem 0 0'}} aria-live="polite">
        Peak {peak.toLocaleString('en-GB')} tokens. Billed in total {billed.toLocaleString('en-GB')}. Requests over the window: {over}. Cache hits{' '}
        {((hits / billed) * 100).toFixed(1)}%. Cost {Math.round(cost).toLocaleString('en-GB')} units ({(cost / billed).toFixed(2)} per token billed). Facts still in the last request: {run.kept} of 6.
      </p>
    </VizPanel>
  );
}
