import {useState} from 'react';

import {summarise} from './capstoneMath';
import {CHUNK_DOC_NAME, CHUNK_DOC_TEXT, CHUNK_SPLITS} from './chunkingData';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const SIZES = [200, 400, 600, 800, 1000];
const OVERLAPS = [0, 50, 100, 200];
const W = 640;
const ROW_H = 22;
const PAD_X = 20;
const BAND = W - PAD_X * 2;

export default function ChunkSplitLab() {
  const dark = useDarkViz();
  const [size, setSize] = useState(1000);
  const [overlap, setOverlap] = useState(200);
  const [selected, setSelected] = useState(0);

  const usable = OVERLAPS.filter((o) => o < size);
  const safeOverlap = usable.includes(overlap) ? overlap : usable[usable.length - 1];
  const spans = CHUNK_SPLITS[`${size}/${safeOverlap}`];
  const stats = summarise(spans, CHUNK_DOC_TEXT.length);
  const current = Math.min(selected, spans.length - 1);
  const [start, end] = spans[current];
  const previousEnd = current > 0 ? spans[current - 1][1] : 0;
  const nextStart = current < spans.length - 1 ? spans[current + 1][0] : CHUNK_DOC_TEXT.length;
  const x = (position: number) => PAD_X + (position / CHUNK_DOC_TEXT.length) * BAND;
  const height = 28 + spans.length * ROW_H + 24;
  const base = seriesColor(0, dark);
  const accent = seriesColor(1, dark);

  const rows = SIZES.flatMap((sz) =>
    OVERLAPS.filter((o) => o < sz).map((o) => {
      const sp = CHUNK_SPLITS[`${sz}/${o}`];
      const st = summarise(sp, CHUNK_DOC_TEXT.length);
      return [`${sz}`, `${o}`, `${st.count}`, `${st.average.toFixed(1)}`, `${st.inflation.toFixed(2)}`];
    }),
  );

  const headOverlap = Math.max(0, previousEnd - start);
  const tailOverlap = Math.max(0, end - nextStart);
  const text = CHUNK_DOC_TEXT.slice(start, end);

  return (
    <VizPanel
      title={`How ${CHUNK_DOC_NAME} is cut into chunks`}
      hint="Each bar is one chunk, drawn over the position it covers in the document. Where two bars overlap, the same text is stored twice. Bigger chunks keep more context together; smaller chunks make retrieval more precise but cost more to store."
      legend={[
        {label: 'chunk', color: base},
        {label: 'selected chunk', color: accent},
      ]}
      table={{columns: ['chunk size', 'overlap', 'chunks', 'average length', 'stored / original'], rows}}
      controls={
        <>
          <label className={s.control}>
            chunk size
            <select
              className={s.select}
              value={size}
              onChange={(e) => {
                setSize(Number(e.target.value));
                setSelected(0);
              }}>
              {SIZES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            overlap
            <select
              className={s.select}
              value={safeOverlap}
              onChange={(e) => {
                setOverlap(Number(e.target.value));
                setSelected(0);
              }}>
              {usable.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            show chunk
            <input
              type="range"
              min={1}
              max={spans.length}
              step={1}
              value={current + 1}
              onChange={(e) => setSelected(Number(e.target.value) - 1)}
            />
            <span className={s.value}>
              {current + 1} of {spans.length}
            </span>
          </label>
          <span className={s.value} aria-live="polite">
            {stats.count} chunks · average {stats.average.toFixed(1)} characters · stored {stats.inflation.toFixed(2)}x the original
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${height}`}
        role="img"
        aria-label={`The document of ${CHUNK_DOC_TEXT.length} characters cut into ${stats.count} chunks of up to ${size} characters with ${safeOverlap} characters of overlap`}>
        <text className={s.axisLabel} x={PAD_X} y={16}>
          position in the document (0 to {CHUNK_DOC_TEXT.length} characters)
        </text>
        {spans.map(([a, b], i) => (
          <g key={`${a}-${b}`}>
            <rect
              x={x(a)}
              y={28 + i * ROW_H}
              width={Math.max(x(b) - x(a), 2)}
              height={ROW_H - 5}
              rx={3}
              fill={i === current ? accent : base}
              opacity={i === current ? 1 : 0.55}
              onClick={() => setSelected(i)}
              style={{cursor: 'pointer'}}
            />
            <text className={s.tick} x={x(a) + 4} y={28 + i * ROW_H + 11} fill="#fff">
              {i + 1}
            </text>
          </g>
        ))}
        <line className={s.axis} x1={PAD_X} y1={height - 18} x2={PAD_X + BAND} y2={height - 18} />
        <text className={s.tick} x={PAD_X} y={height - 4}>
          0
        </text>
        <text className={s.tick} x={PAD_X + BAND} y={height - 4} textAnchor="end">
          {CHUNK_DOC_TEXT.length}
        </text>
      </svg>
      <p style={{margin: '0.5rem 1rem 0', fontSize: '0.82rem'}}>
        <strong>Chunk {current + 1}</strong> covers characters {start} to {end}
        {headOverlap > 0 ? `; its first ${headOverlap} characters repeat the end of chunk ${current}` : ''}
        {tailOverlap > 0 ? `; its last ${tailOverlap} characters are repeated at the start of chunk ${current + 2}` : ''}.
      </p>
      <pre
        style={{margin: '0.4rem 1rem 1rem', padding: '0.6rem', whiteSpace: 'pre-wrap', fontSize: '0.78rem', maxHeight: '11rem', overflow: 'auto'}}
        aria-label={`Text of chunk ${current + 1}`}>
        {text}
      </pre>
    </VizPanel>
  );
}
