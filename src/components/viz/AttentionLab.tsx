import {useMemo, useState} from 'react';

import {sequentialColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const SENTENCES: Record<string, {tokens: string[]; links: [number, number, number][]}> = {
  'The animal did not cross the street because it was too tired': {
    tokens: 'The animal did not cross the street because it was too tired'.split(' '),
    // [query index, key index, strength] — the structure a trained head learns
    links: [[8, 1, 3.4], [8, 6, 1.1], [4, 1, 2.0], [4, 6, 2.2], [11, 1, 1.6], [3, 4, 2.4]],
  },
  'The trophy did not fit in the suitcase because it was too small': {
    tokens: 'The trophy did not fit in the suitcase because it was too small'.split(' '),
    links: [[9, 7, 3.2], [9, 1, 1.0], [4, 1, 1.8], [4, 7, 2.4], [12, 7, 1.7], [3, 4, 2.2]],
  },
};

function softmax(row: number[], temperature: number) {
  const scaled = row.map((v) => v / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / total);
}

export default function AttentionLab() {
  const dark = useDarkViz();
  const [sentenceKey, setSentenceKey] = useState(Object.keys(SENTENCES)[0]);
  const [temperature, setTemperature] = useState(1);
  const [causal, setCausal] = useState(false);
  const [hover, setHover] = useState<{q: number; k: number} | null>(null);

  const {tokens, links} = SENTENCES[sentenceKey];

  const weights = useMemo(() => {
    const n = tokens.length;
    const raw: number[][] = Array.from({length: n}, (_, q) =>
      Array.from({length: n}, (_, k) => {
        if (q === k) return 1.2;                       // mild self-attention
        const link = links.find(([lq, lk]) => lq === q && lk === k);
        return link ? link[2] : 0.15;
      }),
    );
    return raw.map((row, q) =>
      softmax(
        row.map((v, k) => (causal && k > q ? -1e9 : v)),
        temperature,
      ),
    );
  }, [tokens, links, temperature, causal]);

  const cell = Math.min(34, 560 / tokens.length);
  const left = 92;
  const top = 78;
  const W = left + cell * tokens.length + 12;
  const H = top + cell * tokens.length + 12;

  const focusRow = hover ? hover.q : null;

  return (
    <VizPanel
      title="Self-attention: every token re-weights every other"
      hint="Each row is one token asking 'which tokens matter to me?' — the row sums to 1. Hover a cell to read the weight. Notice which word 'it' attends to, and how the answer changes with the sentence."
      legend={[
        {label: 'low attention', color: sequentialColor(0.08, dark)},
        {label: 'high attention', color: sequentialColor(0.95, dark)},
      ]}
      table={{
        columns: ['query token', 'attends most to', 'weight'],
        rows: tokens.map((t, q) => {
          const row = weights[q];
          let best = 0;
          row.forEach((v, k) => {
            if (v > row[best]) best = k;
          });
          return [`${q}: ${t}`, `${best}: ${tokens[best]}`, row[best].toFixed(3)];
        }),
      }}
      controls={
        <>
          <label className={s.control}>
            sentence
            <select className={s.select} value={sentenceKey}
                    onChange={(e) => setSentenceKey(e.target.value)}>
              {Object.keys(SENTENCES).map((key) => (
                <option key={key} value={key}>
                  {key.slice(0, 34)}…
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            temperature
            <input type="range" min={0.4} max={3} step={0.1} value={temperature}
                   onChange={(e) => setTemperature(Number(e.target.value))} />
            <span className={s.value}>{temperature.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={causal}
                   onChange={(e) => setCausal(e.target.checked)} />
            causal mask (decoder)
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Attention weight matrix over the sentence tokens">
        {tokens.map((t, k) => (
          <text key={`col-${k}`} className={s.tick}
                x={left + k * cell + cell / 2} y={top - 8}
                textAnchor="start"
                transform={`rotate(-55 ${left + k * cell + cell / 2} ${top - 8})`}>
            {t}
          </text>
        ))}
        {tokens.map((t, q) => (
          <text key={`row-${q}`} className={s.tick} x={left - 8} y={top + q * cell + cell / 2 + 3}
                textAnchor="end"
                style={{fontWeight: focusRow === q ? 700 : 400}}>
            {t}
          </text>
        ))}
        {weights.map((row, q) =>
          row.map((value, k) => {
            const masked = causal && k > q;
            return (
              <rect
                key={`${q}-${k}`}
                x={left + k * cell}
                y={top + q * cell}
                width={cell - 1.5}
                height={cell - 1.5}
                rx={3}
                fill={masked ? 'var(--surface-2)' : sequentialColor(Math.min(value * 1.6, 1), dark)}
                opacity={focusRow === null || focusRow === q ? 1 : 0.35}
                onMouseEnter={() => setHover({q, k})}
                onMouseLeave={() => setHover(null)}>
                <title>
                  {masked
                    ? `masked: "${tokens[q]}" cannot see "${tokens[k]}"`
                    : `"${tokens[q]}" → "${tokens[k]}": ${value.toFixed(3)}`}
                </title>
              </rect>
            );
          }),
        )}
      </svg>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0}}>
        {hover ? (
          <span>
            <strong>{tokens[hover.q]}</strong> → <strong>{tokens[hover.k]}</strong>:{' '}
            {weights[hover.q][hover.k].toFixed(3)}
          </span>
        ) : (
          <span>hover a cell to read one weight; every row sums to 1.000</span>
        )}
        <span>{causal ? 'causal mask on — a decoder cannot look ahead' : 'full attention — an encoder sees everything'}</span>
      </div>
    </VizPanel>
  );
}
