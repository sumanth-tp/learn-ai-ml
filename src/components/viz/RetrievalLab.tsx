import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const CHUNKS = [
  {id: 'shipping.md#1', text: 'Shipping costs are refunded only when the order arrived damaged.'},
  {id: 'refunds.md#0', text: 'Refunds are issued to the original payment method within 14 days.'},
  {id: 'refunds.md#1', text: 'Damaged goods qualify for a full refund including delivery charges.'},
  {id: 'shipping.md#0', text: 'Standard delivery takes 3 to 5 business days; express is next day.'},
  {id: 'accounts.md#0', text: 'You can reset a password from the login page at any time.'},
  {id: 'returns.md#0', text: 'Unwanted items may be returned unused within 30 days of delivery.'},
];

const QUERIES = [
  'do I get shipping costs back if my order arrived damaged',
  'how long do refunds take',
  'can I send something back I do not want',
];

// A real lexical index drops stop words; in a corpus this small they otherwise
// carry high IDF and swamp the terms that matter.
const STOP = new Set(['a', 'an', 'the', 'i', 'you', 'can', 'do', 'does', 'not', 'is',
  'are', 'was', 'were', 'be', 'to', 'of', 'in', 'on', 'at', 'for', 'from', 'my',
  'it', 'this', 'that', 'and', 'or', 'if', 'how', 'what', 'may', 'within', 'any',
  'something', 'get', 'take', 'takes', 'long', 'want', 'send', 'back', 'so']);

const tokens = (t: string) =>
  (t.toLowerCase().match(/[a-z]+/g) ?? []).filter((w) => !STOP.has(w));

/** Lexical scoring: exact term overlap with IDF weighting. */
function bm25Scores(query: string) {
  const df = new Map<string, number>();
  CHUNKS.forEach((c) => new Set(tokens(c.text)).forEach((t) => df.set(t, (df.get(t) ?? 0) + 1)));
  const avg = CHUNKS.reduce((a, c) => a + tokens(c.text).length, 0) / CHUNKS.length;
  return CHUNKS.map((c) => {
    const counts = new Map<string, number>();
    tokens(c.text).forEach((t) => counts.set(t, (counts.get(t) ?? 0) + 1));
    const dl = tokens(c.text).length;
    let score = 0;
    for (const term of tokens(query)) {
      const tf = counts.get(term);
      if (!tf) continue;
      const n = df.get(term) ?? 0;
      const idf = Math.log(1 + (CHUNKS.length - n + 0.5) / (n + 0.5));
      score += (idf * tf * 2.5) / (tf + 1.5 * (1 - 0.75 + (0.75 * dl) / avg));
    }
    return score;
  });
}

/** Semantic scoring: a small hand-built synonym map stands in for an encoder. */
const SYNONYMS: Record<string, string[]> = {
  back: ['refund', 'refunds', 'returned', 'return'],
  send: ['returned', 'return'],
  want: ['unwanted'],
  long: ['days', 'time'],
  take: ['days'],
  costs: ['charges', 'delivery'],
  shipping: ['delivery'],
  damaged: ['damaged'],
};

function denseScores(query: string) {
  // Semantic matching works from the raw query — stop words still carry intent
  // ("send back" → returns), which is exactly what a dense encoder captures.
  const raw = query.toLowerCase().match(/[a-z]+/g) ?? [];
  const expanded = new Set(tokens(query));
  for (const term of raw) (SYNONYMS[term] ?? []).forEach((syn) => expanded.add(syn));
  return CHUNKS.map((c) => {
    const words = new Set(tokens(c.text));
    let overlap = 0;
    expanded.forEach((t) => {
      if (words.has(t)) overlap += 1;
    });
    return overlap / Math.sqrt(words.size);
  });
}

function rankOf(scores: number[]) {
  return scores
    .map((score, index) => ({score, index}))
    .sort((a, b) => b.score - a.score)
    .map((x) => x.index);
}

/** Reciprocal rank fusion — the standard way to merge two rankings. */
function fuse(a: number[], b: number[], k = 3) {
  const points = new Map<number, number>();
  [a, b].forEach((ranking) =>
    ranking.forEach((index, rank) =>
      points.set(index, (points.get(index) ?? 0) + 1 / (k + rank + 1)),
    ),
  );
  return points;
}

export default function RetrievalLab() {
  const dark = useDarkViz();
  const [query, setQuery] = useState(QUERIES[0]);

  const {lexical, dense, fusedRank, points} = useMemo(() => {
    const lexicalScores = bm25Scores(query);
    const denseScoresOut = denseScores(query);
    const points = fuse(rankOf(lexicalScores), rankOf(denseScoresOut));
    const fusedRank = [...points.entries()].sort((a, b) => b[1] - a[1]).map(([i]) => i);
    return {lexical: lexicalScores, dense: denseScoresOut, fusedRank, points};
  }, [query]);

  const lexRank = rankOf(lexical);
  const denseRank = rankOf(dense);

  const colLex = seriesColor(0, dark);
  const colDense = seriesColor(1, dark);
  const colFused = seriesColor(2, dark);

  const column = (title: string, order: number[], color: string, score: (i: number) => string) => (
    <div style={{flex: 1, minWidth: 0}}>
      <div style={{fontSize: '0.75rem', fontWeight: 700, color, marginBottom: '0.35rem'}}>
        {title}
      </div>
      <ol style={{margin: 0, paddingLeft: '1.1rem', fontSize: '0.78rem', lineHeight: 1.5}}>
        {order.slice(0, 4).map((index) => (
          <li key={index} style={{color: 'var(--text-muted)'}}>
            <code style={{fontSize: '0.72rem'}}>{CHUNKS[index].id}</code>{' '}
            <span style={{color: 'var(--text-faint)'}}>{score(index)}</span>
          </li>
        ))}
      </ol>
    </div>
  );

  return (
    <VizPanel
      title="Hybrid retrieval: lexical, dense and fused"
      hint="Lexical matching locks onto exact words; the semantic side catches paraphrase. Fusion takes both rankings and merges them — which is why production RAG runs both rather than choosing."
      legend={[
        {label: 'BM25 (lexical)', color: colLex},
        {label: 'semantic (dense)', color: colDense},
        {label: 'fused (RRF)', color: colFused},
      ]}
      table={{
        columns: ['chunk', 'BM25', 'dense', 'RRF points'],
        rows: CHUNKS.map((c, i) => [
          c.id,
          lexical[i].toFixed(3),
          dense[i].toFixed(3),
          (points.get(i) ?? 0).toFixed(4),
        ]),
      }}
      controls={
        <label className={s.control}>
          query
          <select className={s.select} value={query} onChange={(e) => setQuery(e.target.value)}>
            {QUERIES.map((q) => (
              <option key={q} value={q}>
                {q}
              </option>
            ))}
          </select>
        </label>
      }>
      <div style={{display: 'flex', gap: '1.25rem', flexWrap: 'wrap'}}>
        {column('BM25 top 4', lexRank, colLex, (i) => lexical[i].toFixed(2))}
        {column('dense top 4', denseRank, colDense, (i) => dense[i].toFixed(2))}
        {column('fused top 4', fusedRank, colFused, (i) => (points.get(i) ?? 0).toFixed(3))}
      </div>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem'}}>
        <span>
          top result — BM25: <code>{CHUNKS[lexRank[0]].id}</code>, dense:{' '}
          <code>{CHUNKS[denseRank[0]].id}</code>, fused: <code>{CHUNKS[fusedRank[0]].id}</code>
        </span>
      </div>
    </VizPanel>
  );
}
