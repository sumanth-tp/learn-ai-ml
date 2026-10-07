import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import {SCHEMA_LINK} from './schemaLinkData';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Mode = 'name' | 'text';

const W = 640;
const ROW = 17;
const LABEL_W = 190;
const BAR_W = 330;
const { columns, questions } = SCHEMA_LINK;

const keyColumns = (table: string) =>
  columns.map((c, i) => ({c, i})).filter(({c}) => c.startsWith(`${table}.`) && c.endsWith('_id')).map(({i}) => i);

const selectColumns = (qi: number, mode: Mode, k: number, addKeys: boolean): Set<number> => {
  const scores = mode === 'name' ? questions[qi].name : questions[qi].text_scores;
  const top = new Set(
    scores
      .map((v, i) => ({v, i}))
      .sort((a, b) => b.v - a.v || a.i - b.i)
      .slice(0, k)
      .map(({i}) => i),
  );
  if (addKeys) {
    const tables = new Set([...top].map((i) => columns[i].split('.')[0]));
    tables.forEach((t) => keyColumns(t).forEach((i) => top.add(i)));
  }
  return top;
};

const recallOf = (qi: number, chosen: Set<number>) => {
  const need = questions[qi].need;
  return need.filter((i) => chosen.has(i)).length / need.length;
};

export default function SchemaLinkingLab() {
  const dark = useDarkViz();
  const [question, setQuestion] = useState(7);
  const [mode, setMode] = useState<Mode>('text');
  const [k, setK] = useState(8);
  const [addKeys, setAddKeys] = useState(true);

  const chosen = useMemo(() => selectColumns(question, mode, k, addKeys), [question, mode, k, addKeys]);
  const overall = useMemo(() => {
    const per = questions.map((_, qi) => {
      const sel = selectColumns(qi, mode, k, addKeys);
      return {recall: recallOf(qi, sel), size: sel.size};
    });
    return {
      recall: per.reduce((a, p) => a + p.recall, 0) / per.length,
      complete: per.filter((p) => p.recall === 1).length,
      size: per.reduce((a, p) => a + p.size, 0) / per.length,
    };
  }, [mode, k, addKeys]);

  const q = questions[question];
  const scores = mode === 'name' ? q.name : q.text_scores;
  const sorted = scores.map((v, i) => ({v, i})).sort((a, b) => b.v - a.v || a.i - b.i);
  const max = sorted[0].v;
  const min = sorted[sorted.length - 1].v;
  const missing = q.need.filter((i) => !chosen.has(i));
  const hit = seriesColor(2, dark);
  const extra = seriesColor(0, dark);
  const miss = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;
  const plain = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;
  const height = 14 + sorted.length * ROW;

  const rows = sorted.map(({v, i}, rank) => [
    rank + 1,
    columns[i],
    v.toFixed(3),
    chosen.has(i) ? 'shown to the model' : '-',
    q.need.includes(i) ? 'needed' : '-',
  ]);

  return (
    <VizPanel
      title="Schema linking: which columns reach the prompt"
      hint="Columns are ranked by similarity to the question. Choose how many the model sees and whether key columns come along. A needed column that is not shown can never appear in a correct query. With description text, k = 8 and key columns on, the summary reproduces block 2 (11 of 12 complete)."
      legend={[
        {label: 'shown and needed', color: hit},
        {label: 'shown, not needed', color: extra},
        {label: 'needed but not shown', color: miss},
        {label: 'not shown', color: plain},
      ]}
      table={{columns: ['rank', 'column', 'similarity', 'shown', 'needed'], rows}}
      controls={
        <>
          <label className={s.control}>
            question
            <select className={s.select} value={question} onChange={(e) => setQuestion(Number(e.target.value))}>
              {questions.map((item, i) => (
                <option key={item.text} value={i}>
                  {`Q${i + 1}: ${item.text.length > 48 ? `${item.text.slice(0, 46)}...` : item.text}`}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            column text
            <select className={s.select} value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              <option value="name">name only</option>
              <option value="text">name and description</option>
            </select>
          </label>
          <label className={s.control}>
            k
            <input type="range" min={3} max={12} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} />
            <span className={s.value}>{k}</span>
          </label>
          <label className={s.control}>
            <input type="checkbox" checked={addKeys} onChange={(e) => setAddKeys(e.target.checked)} />
            add key columns
          </label>
          <span className={s.value} aria-live="polite">
            this question: {chosen.size} columns shown, {missing.length === 0 ? 'every needed column present' : `missing ${missing.map((i) => columns[i]).join(', ')}`}. All 12 questions: mean recall {overall.recall.toFixed(3)}, {overall.complete} of 12 complete, {overall.size.toFixed(1)} columns on average.
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`Ranked columns for the question ${q.text}. ${chosen.size} shown, ${missing.length} needed columns missing.`}>
        {sorted.map(({v, i}, rank) => {
          const shown = chosen.has(i);
          const needed = q.need.includes(i);
          const colour = shown && needed ? hit : shown ? extra : needed ? miss : plain;
          const width = 6 + ((v - min) / (max - min || 1)) * (BAR_W - 6);
          const y = 8 + rank * ROW;
          return (
            <g key={columns[i]}>
              <text className={s.tick} x={LABEL_W - 6} y={y + 10} textAnchor="end">
                {columns[i]}
              </text>
              <rect x={LABEL_W} y={y} width={width} height={ROW - 5} rx={2} fill={colour} opacity={shown || needed ? 0.92 : 0.3} />
              <text className={s.tick} x={LABEL_W + width + 6} y={y + 10}>
                {shown ? 'shown' : ''}
                {shown && needed ? ' + ' : ''}
                {needed ? 'needed' : ''}
              </text>
            </g>
          );
        })}
      </svg>
    </VizPanel>
  );
}
