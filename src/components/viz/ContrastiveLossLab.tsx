import {useState} from 'react';

import {sequentialColor, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 340;

export const QUERIES: string[] = ["I cannot log in to the compass", "I cannot log in to the mailbox", "I cannot log in to the harbour", "I cannot log in to the pebble", "I cannot log in to the lighthouse", "I cannot log in to the meadow", "I cannot log in to the hammock", "I cannot log in to the sunbed"];

export const MATRICES: Record<string, number[][]> = {"base":[[0.2330457,0.2733271,0.1823043,0.3198406,0.2682874,0.1343111,0.1721012,0.2617068],[0.1346011,0.2083922,0.1579336,0.282573,0.2072987,0.1019136,0.1386324,0.1794758],[0.2640447,0.1513087,0.1471647,0.2674483,0.2460857,0.1485673,0.0968622,0.1929853],[0.2197455,0.1389109,0.1481193,0.2353644,0.1796187,0.1516166,0.2437095,0.226302],[0.2366409,0.2014354,0.1500124,0.224837,0.2435011,0.1672036,0.1539335,0.2348532],[0.2231379,0.2304054,0.1368166,0.1764417,0.2292345,0.1578623,0.0834502,0.2766484],[0.3398062,0.1982147,0.1821385,0.2571935,0.21566,0.0855613,0.1639875,0.1911684],[0.1963188,0.1731792,0.169491,0.2327104,0.2330885,0.1165375,0.1907239,0.1863358]],"tuned":[[0.4107186,0.3655965,0.3299122,0.3988348,0.3431722,0.2730911,0.3290304,0.3439098],[0.2188907,0.3825334,0.2906497,0.3316626,0.2720412,0.2287804,0.2848541,0.293447],[0.3464802,0.2642067,0.3887567,0.3372567,0.3212892,0.3089723,0.2623422,0.2738774],[0.3278665,0.250869,0.3130838,0.391193,0.2684579,0.3093823,0.3652574,0.281378],[0.3406459,0.3079186,0.3245414,0.3010804,0.3641599,0.304356,0.2997864,0.3093979],[0.3184144,0.3314579,0.3171195,0.278032,0.3249103,0.364372,0.251163,0.321183],[0.3478664,0.2995006,0.3101386,0.3165428,0.2686414,0.2199422,0.3729266,0.2589169],[0.2919262,0.3037119,0.3085297,0.3167761,0.2920953,0.2488817,0.3455524,0.3588629]],"hard":[[0.3975024,0.3582379,0.3555364,0.3761197,0.3649519,0.3097592,0.3738306,0.3001425],[0.2075063,0.3817458,0.3063083,0.3114931,0.2991798,0.2600535,0.326386,0.2676301],[0.3360823,0.2680249,0.3830023,0.3119535,0.3366189,0.3277464,0.3031714,0.2271969],[0.3137975,0.2594964,0.3262694,0.3692742,0.2956345,0.3326345,0.3999367,0.2506412],[0.3255025,0.3024296,0.3323335,0.2756133,0.3556329,0.3242618,0.3373409,0.2555748],[0.306087,0.3262182,0.33409,0.2657132,0.3414309,0.3720706,0.2952379,0.2716154],[0.3231295,0.2970785,0.3161119,0.294122,0.2910333,0.2434348,0.3872112,0.2243791],[0.2738777,0.2938808,0.3152804,0.2954108,0.3147571,0.2788417,0.3707621,0.296673]]};

export const MODELS: {key: string; label: string}[] = [
  {key: 'base', label: 'base all-MiniLM-L6-v2'},
  {key: 'tuned', label: 'tuned, in-batch negatives'},
  {key: 'hard', label: 'tuned, plus hard negatives'},
];

export function rowLosses(matrix: number[][], temperature: number, size: number): number[] {
  return matrix.slice(0, size).map((row, i) => {
    const logits = row.slice(0, size).map((v) => v / temperature);
    const top = Math.max(...logits);
    const sum = logits.reduce((a, v) => a + Math.exp(v - top), 0);
    return -(logits[i] - top - Math.log(sum));
  });
}

export function probabilityMatrix(matrix: number[][], temperature: number, size: number): number[][] {
  return matrix.slice(0, size).map((row) => {
    const logits = row.slice(0, size).map((v) => v / temperature);
    const top = Math.max(...logits);
    const e = logits.map((v) => Math.exp(v - top));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / sum);
  });
}

export default function ContrastiveLossLab() {
  const dark = useDarkViz();
  const [model, setModel] = useState('base');
  const [temperature, setTemperature] = useState(0.05);
  const [negatives, setNegatives] = useState(7);

  const matrix = MATRICES[model];
  const size = negatives + 1;
  const losses = rowLosses(matrix, temperature, size);
  const mean = losses.reduce((a, b) => a + b, 0) / losses.length;
  const probMatrix = probabilityMatrix(matrix, temperature, size);
  const probs = probMatrix.map((row, i) => row[i]);

  const CELL = 30;
  const GRID = {x: 30, y: 52};
  const BAR = {x: GRID.x + 8 * CELL + 70, w: 190};
  const maxLoss = Math.max(8.5, ...losses);
  const barColor = seriesColor(1, dark);

  const rows = QUERIES.slice(0, size).map((q, i) => {
    const wrong = matrix[i].slice(0, size).filter((_, j) => j !== i);
    return [
      q.replace('I cannot log in to the ', ''),
      matrix[i][i].toFixed(3),
      wrong.length ? Math.max(...wrong).toFixed(3) : '-',
      probs[i].toFixed(3),
      losses[i].toFixed(3),
    ];
  });

  return (
    <VizPanel
      title="Contrastive loss, temperature and negatives"
      hint="Each row is a query. The loss asks the right document to win a softmax over the batch. Lower the temperature and the softmax sharpens: wrong documents with a high cosine are punished hard. Add negatives and the task gets harder. The base model cannot tell the nicknames apart (mean cosine to the right and wrong documents is 0.197 and 0.196), the tuned model can. Defaults match the chapter: base model, temperature 0.05, 7 negatives gives 2.5865; the tuned models give 1.0621 and 1.3016."
      legend={[
        {label: 'probability on the right document', color: sequentialColor(0.85, dark)},
        {label: 'row loss', color: barColor},
      ]}
      table={{columns: ['query (nickname)', 'cosine, right', 'cosine, best wrong', 'probability', 'row loss'], rows}}
      controls={
        <>
          <label className={s.control}>
            model
            <select className={s.select} value={model} onChange={(e) => setModel(e.target.value)}>
              {MODELS.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            temperature
            <input type="range" min={0.01} max={0.5} step={0.01} value={temperature}
                   onChange={(e) => setTemperature(Math.round(Number(e.target.value) * 100) / 100)} />
            <span className={s.value}>{temperature.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            in-batch negatives
            <input type="range" min={1} max={7} step={1} value={negatives}
                   onChange={(e) => setNegatives(Number(e.target.value))} />
            <span className={s.value}>{negatives}</span>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Softmax probability matrix for ${size} queries, mean loss ${mean.toFixed(4)}`}>
        <text className={s.axisLabel} x={GRID.x} y={GRID.y - 22}>
          softmax over documents, rows are queries
        </text>
        {QUERIES.map((_, i) => (
          <g key={i}>
            <text className={s.tick} x={GRID.x - 6} y={GRID.y + i * CELL + CELL / 2 + 3} textAnchor="end">q{i + 1}</text>
            <text className={s.tick} x={GRID.x + i * CELL + CELL / 2} y={GRID.y - 6} textAnchor="middle">d{i + 1}</text>
          </g>
        ))}
        {QUERIES.map((_, i) =>
          QUERIES.map((__, j) => {
            const used = i < size && j < size;
            const probability = used ? probMatrix[i][j] : 0;
            return (
              <g key={`${i}-${j}`}>
                <rect x={GRID.x + j * CELL} y={GRID.y + i * CELL} width={CELL - 1} height={CELL - 1}
                      fill={used ? sequentialColor(probability, dark) : 'var(--surface-1)'}
                      stroke={i === j ? 'var(--text-strong)' : 'none'} strokeWidth={i === j ? 2 : 0}
                      opacity={used ? 1 : 0.4} />
                {used && (
                  <text className={s.tick} x={GRID.x + j * CELL + CELL / 2} y={GRID.y + i * CELL + CELL / 2 + 3}
                        textAnchor="middle" fill={probability > 0.45 ? '#fff' : 'var(--text-strong)'}>
                    {probability.toFixed(2).replace('0.', '.')}
                  </text>
                )}
              </g>
            );
          }),
        )}
        <text className={s.axisLabel} x={BAR.x} y={GRID.y - 22}>row loss</text>
        {QUERIES.map((_, i) => (
          <g key={i}>
            {i < size && (
              <>
                <rect x={BAR.x} y={GRID.y + i * CELL + 4} width={(losses[i] / maxLoss) * BAR.w} height={CELL - 9} fill={barColor} rx={3} />
                <text className={s.dataLabel} x={BAR.x + (losses[i] / maxLoss) * BAR.w + 5} y={GRID.y + i * CELL + CELL / 2 + 3}>
                  {losses[i].toFixed(2)}
                </text>
              </>
            )}
          </g>
        ))}
        <text className={s.dataLabel} x={GRID.x} y={GRID.y + 8 * CELL + 28}>
          mean loss {mean.toFixed(4)} over {size} queries, {negatives} negative{negatives === 1 ? '' : 's'} each, temperature {temperature.toFixed(2)} (scale {(1 / temperature).toFixed(0)})
        </text>
      </svg>
    </VizPanel>
  );
}
