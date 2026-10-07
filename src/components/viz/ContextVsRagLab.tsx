import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const ROW = 38;
const WINDOW = 1_000_000;
const QUESTION = 100;
const ANSWER = 300;
const CACHE_SECONDS = 300;
const CACHE_WRITE_RATIO = 1.25;
const CACHE_READ_RATIO = 0.1;
const LABEL_W = 190;
const BAR_W = 330;

const money = (tokens: number, price: number) => (tokens * price) / 1e6;
const dollars = (value: number) => (value >= 100 ? value.toFixed(0) : value >= 1 ? value.toFixed(2) : value.toFixed(4));

export default function ContextVsRagLab() {
  const dark = useDarkViz();
  const [corpus, setCorpus] = useState(500_000);
  const [perHour, setPerHour] = useState(12);
  const [ragContext, setRagContext] = useState(4_000);
  const [ragShare, setRagShare] = useState(80);
  const [priceIn, setPriceIn] = useState(2);
  const [priceOut, setPriceOut] = useState(10);

  const fits = corpus + QUESTION + ANSWER <= WINDOW;
  const tail = money(QUESTION, priceIn) + money(ANSWER, priceOut);
  const cold = money(corpus, priceIn) + tail;
  const write = money(corpus, priceIn * CACHE_WRITE_RATIO) + tail;
  const warm = money(corpus, priceIn * CACHE_READ_RATIO) + tail;
  const rag = money(ragContext + QUESTION, priceIn) + money(ANSWER, priceOut);
  const hybrid = rag + (1 - ragShare / 100) * cold;

  const gap = 3600 / perHour;
  const cachedHour = gap <= CACHE_SECONDS ? write + (perHour - 1) * warm : perHour * write;

  const items = [
    {label: 'long context, no cache', query: cold, hour: perHour * cold, colour: seriesColor(1, dark), usable: fits},
    {label: 'long context, with cache', query: gap <= CACHE_SECONDS ? warm : write, hour: cachedHour, colour: seriesColor(3, dark), usable: fits},
    {label: 'RAG', query: rag, hour: perHour * rag, colour: seriesColor(2, dark), usable: true},
    {label: `hybrid (${ragShare}% by RAG)`, query: hybrid, hour: perHour * hybrid, colour: seriesColor(0, dark), usable: fits},
  ];
  const max = Math.max(...items.filter((i) => i.usable).map((i) => i.hour), 1e-9);
  const unavailable = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;
  const height = 24 + items.length * ROW;

  const rows = items.map((i) => [i.label, i.usable ? dollars(i.query) : 'does not fit', i.usable ? dollars(i.hour) : 'does not fit']);

  return (
    <VizPanel
      title="Long context against RAG: what a day of questions costs"
      hint="Bars show cost per hour at the chosen traffic, using the same formulas as block 1. Prices default to the Claude Sonnet 5.5 list prices checked on 7 October 2026 (2.00 in, 10.00 out per million tokens; cache read 0.1 times, 5-minute cache write 1.25 times the input price). The cache stays warm only if the gap between queries is under 5 minutes. At the defaults the row for 12 queries per hour in block 1 is reproduced."
      legend={items.map((i) => ({label: i.label, color: i.colour}))}
      table={{columns: ['strategy', 'dollars per query', 'dollars per hour'], rows}}
      controls={
        <>
          <label className={s.control}>
            corpus tokens
            <input type="range" min={10_000} max={1_200_000} step={10_000} value={corpus} onChange={(e) => setCorpus(Number(e.target.value))} />
            <span className={s.value}>{corpus.toLocaleString('en-GB')}</span>
          </label>
          <label className={s.control}>
            queries per hour
            <select className={s.select} value={perHour} onChange={(e) => setPerHour(Number(e.target.value))}>
              {[1, 6, 12, 60, 600].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            RAG context tokens
            <input type="range" min={1000} max={16000} step={500} value={ragContext} onChange={(e) => setRagContext(Number(e.target.value))} />
            <span className={s.value}>{ragContext.toLocaleString('en-GB')}</span>
          </label>
          <label className={s.control}>
            answered by RAG
            <input type="range" min={0} max={100} step={5} value={ragShare} onChange={(e) => setRagShare(Number(e.target.value))} />
            <span className={s.value}>{ragShare}%</span>
          </label>
          <label className={s.control}>
            price in
            <input type="range" min={0.5} max={10} step={0.5} value={priceIn} onChange={(e) => setPriceIn(Number(e.target.value))} />
            <span className={s.value}>{priceIn.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            price out
            <input type="range" min={1} max={50} step={1} value={priceOut} onChange={(e) => setPriceOut(Number(e.target.value))} />
            <span className={s.value}>{priceOut.toFixed(2)}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {fits ? `long context costs ${(cold / rag).toFixed(0)} times RAG per query; cache is ${gap <= CACHE_SECONDS ? 'warm' : 'cold (gap over 5 minutes)'}` : 'the corpus does not fit in a 1,000,000-token window'}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`Cost per hour at ${perHour} queries per hour: ${items.map((i) => `${i.label} ${i.usable ? dollars(i.hour) : 'does not fit'}`).join(', ')}`}>
        <text className={s.axisLabel} x={LABEL_W} y={14}>
          dollars per hour
        </text>
        {items.map((item, i) => {
          const y = 24 + i * ROW;
          const width = item.usable ? Math.max(3, (item.hour / max) * BAR_W) : BAR_W;
          return (
            <g key={item.label}>
              <text className={s.tick} x={LABEL_W - 8} y={y + 16} textAnchor="end">
                {item.label}
              </text>
              <rect x={LABEL_W} y={y} width={width} height={ROW - 12} rx={3} fill={item.usable ? item.colour : unavailable} opacity={item.usable ? 0.92 : 0.25} />
              <text className={s.dataLabel} x={LABEL_W + width + 6} y={y + 16}>
                {item.usable ? dollars(item.hour) : 'does not fit'}
              </text>
            </g>
          );
        })}
      </svg>
    </VizPanel>
  );
}
