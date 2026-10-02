import {useState} from 'react';

import {seriesColor} from './palette';
import {FUNNEL_DOCS, FUNNEL_QUERIES, FUNNEL_RANKS} from './retrievalFunnelData';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 210;
const CUTS = [1, 5, 10, 20, 50];
const RETRIEVERS: {key: string; label: string}[] = [
  {key: 'bm25', label: 'BM25'},
  {key: 'dense', label: 'dense (MiniLM)'},
  {key: 'hybrid', label: 'hybrid (RRF, k = 60)'},
];

const share = (ranks: number[], limit: number, zeroMeansMiss: boolean) =>
  ranks.filter((r) => (zeroMeansMiss ? r > 0 && r <= limit : r <= limit)).length / ranks.length;

export default function RetrievalFunnelLab() {
  const dark = useDarkViz();
  const [retriever, setRetriever] = useState('hybrid');
  const [depth, setDepth] = useState(20);
  const [rerank, setRerank] = useState(true);
  const [chunks, setChunks] = useState(5);
  const [chunkTokens, setChunkTokens] = useState(350);

  const canRerank = retriever === 'hybrid' && depth <= 20;
  const useRerank = rerank && canRerank;
  const sent = Math.min(chunks, depth);
  const ceiling = share(FUNNEL_RANKS[retriever], depth, false);
  const final = useRerank
    ? share(FUNNEL_RANKS[depth === 10 ? 'rerank10' : 'rerank20'], sent, true)
    : share(FUNNEL_RANKS[retriever], sent, false);
  const tokens = sent * chunkTokens;

  const bars = [
    {label: `all documents: ${FUNNEL_DOCS.toLocaleString()}`, value: 1, shown: 'every query has its evidence somewhere'},
    {label: `retrieved: ${depth}`, value: ceiling, shown: `${ceiling.toFixed(2)} of queries keep their evidence`},
    {
      label: `in the prompt: ${sent}${useRerank ? ' after rerank' : ''}`,
      value: final,
      shown: `${final.toFixed(2)} of queries keep their evidence, ${tokens.toLocaleString()} prompt tokens`,
    },
  ];

  const barX = 20;
  const barW = W - 2 * barX;
  const rows = RETRIEVERS.map((r) => [
    r.label,
    ...CUTS.map((k) => share(FUNNEL_RANKS[r.key], k, false).toFixed(2)),
  ]);

  return (
    <VizPanel
      title="Retrieval funnel on 100 SciFact questions"
      hint={`Real ranks from the chapter code over ${FUNNEL_QUERIES} questions and ${FUNNEL_DOCS} documents. Each bar is the share of questions whose evidence is still inside after that stage. Defaults reproduce 0.89 (hybrid, depth 20, rerank, 5 chunks); switch the rerank off to see 0.92.`}
      legend={[{label: 'share of questions with the evidence still in', color: seriesColor(0, dark)}]}
      table={{columns: ['retriever', 'top 1', 'top 5', 'top 10', 'top 20', 'top 50'], rows}}
      controls={
        <>
          <label className={s.control}>
            retriever
            <select className={s.select} value={retriever} onChange={(e) => setRetriever(e.target.value)}>
              {RETRIEVERS.map((r) => (
                <option key={r.key} value={r.key}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            retrieval depth
            <select className={s.select} value={depth} onChange={(e) => setDepth(Number(e.target.value))}>
              {[10, 20, 50].map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            <input
              type="checkbox"
              checked={useRerank}
              disabled={!canRerank}
              onChange={(e) => setRerank(e.target.checked)}
            />
            cross-encoder rerank {canRerank ? '' : '(hybrid, depth 10 or 20 only)'}
          </label>
          <label className={s.control}>
            chunks in the prompt
            <select className={s.select} value={chunks} onChange={(e) => setChunks(Number(e.target.value))}>
              {[1, 3, 5, 10].map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            tokens per chunk
            <input
              type="range"
              min={100}
              max={800}
              step={50}
              value={chunkTokens}
              onChange={(e) => setChunkTokens(Number(e.target.value))}
            />
            <span className={s.value}>{chunkTokens}</span>
          </label>
          <span className={s.value} aria-live="polite">
            evidence in the prompt for {final.toFixed(2)} of questions, {tokens.toLocaleString()} tokens
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Funnel: ${bars.map((b) => `${b.label}, ${b.shown}`).join('; ')}`}>
        {bars.map((b, i) => {
          const y = 14 + i * 64;
          return (
            <g key={b.label}>
              <text className={s.dataLabel} x={barX} y={y + 12}>
                {b.label}
              </text>
              <rect x={barX} y={y + 20} width={barW} height={18} rx={4} fill="none" stroke="var(--border, #aaa)" />
              <rect
                x={barX}
                y={y + 20}
                width={Math.max(2, barW * b.value)}
                height={18}
                rx={4}
                fill={seriesColor(i === 2 ? 1 : 0, dark)}
              />
              <text className={s.tick} x={barX} y={y + 54}>
                {b.shown}
              </text>
            </g>
          );
        })}
      </svg>
    </VizPanel>
  );
}
