import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Point = [number, number];
const POINTS: Point[] = [[0.1, 0.2], [0.2, 0.1], [0.85, 0.8], [0.75, 0.9]];
const INITIAL: Point[] = [[0.1, 0.2], [0.25, 0.1]];
const distance = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export default function DocumentGroupingLab() {
  const dark = useDarkViz();
  const [mode, setMode] = useState<'classification' | 'clustering'>('classification');
  const [sport, setSport] = useState(0.7);
  const [politics, setPolitics] = useState(0.4);
  const [centres, setCentres] = useState<Point[]>(INITIAL);
  const [steps, setSteps] = useState(0);
  const assignments = POINTS.map((point) => distance(point, centres[0]) <= distance(point, centres[1]) ? 0 : 1);
  const iterate = () => {
    const next = centres.map((centre, group): Point => {
      const members = POINTS.filter((_, index) => assignments[index] === group);
      return members.length ? [members.reduce((sum, point) => sum + point[0], 0) / members.length,
        members.reduce((sum, point) => sum + point[1], 0) / members.length] : centre;
    });
    setCentres(next);
    setSteps((current) => current + 1);
  };
  const reset = () => { setCentres(INITIAL); setSteps(0); };
  const table = mode === 'classification'
    ? {columns: ['category', 'cosine', 'assigned'], rows: [['sport', sport.toFixed(2), sport >= politics ? 'yes' : 'no'], ['politics', politics.toFixed(2), politics > sport ? 'yes' : 'no']]}
    : {columns: ['document', 'x', 'y', 'cluster'], rows: POINTS.map((point, index) => [`D${index + 1}`, point[0], point[1], assignments[index] + 1])};

  return (
    <VizPanel title="Classify or cluster the documents"
      hint="Classification compares a document with labelled class centroids. Clustering repeats assignment and centroid updates without class labels."
      table={table}
      controls={<div className={s.controls}>
        <label className={s.control}>task
          <select className={s.select} value={mode} onChange={(event) => setMode(event.target.value as typeof mode)} aria-label="Document grouping task">
            <option value="classification">classification</option><option value="clustering">clustering</option>
          </select>
        </label>
        {mode === 'classification' ? <>
          <label className={s.control}>sport cosine: {sport.toFixed(2)}
            <input type="range" min="0" max="1" step="0.05" value={sport} aria-label="Sport centroid similarity"
              onChange={(event) => setSport(Number(event.target.value))} />
          </label>
          <label className={s.control}>politics cosine: {politics.toFixed(2)}
            <input type="range" min="0" max="1" step="0.05" value={politics} aria-label="Politics centroid similarity"
              onChange={(event) => setPolitics(Number(event.target.value))} />
          </label>
        </> : <>
          <button type="button" onClick={iterate}>Run one k-means step</button>
          <button type="button" onClick={reset}>Reset centres</button>
        </>}
      </div>}>
      {mode === 'classification' ? <div style={{display: 'grid', gap: '0.7rem'}}>
        {[['sport', sport], ['politics', politics]].map(([name, value], index) => (
          <div key={name} style={{display: 'grid', gridTemplateColumns: '5rem minmax(0, 1fr) 3rem', gap: '0.5rem', alignItems: 'center'}}>
            <span>{name}</span><div style={{background: 'var(--ifm-color-emphasis-200)', height: '1.2rem', borderRadius: '0.25rem'}}>
              <div style={{width: `${Number(value) * 100}%`, height: '100%', background: seriesColor(index, dark), borderRadius: '0.25rem'}} />
            </div><strong>{Number(value).toFixed(2)}</strong>
          </div>
        ))}
        <div aria-live="polite">assigned category: <strong>{sport >= politics ? 'sport' : 'politics'}</strong></div>
      </div> : <div>
        <svg viewBox="0 0 320 250" role="img" aria-label="Four document points and two moving k-means centres" style={{width: '100%', maxWidth: '480px'}}>
          <rect x="12" y="12" width="296" height="226" fill="none" stroke="var(--ifm-color-emphasis-400)" />
          {POINTS.map((point, index) => <g key={index}>
            <circle cx={22 + point[0] * 270} cy={228 - point[1] * 205} r="8" fill={seriesColor(assignments[index], dark)} />
            <text x={29 + point[0] * 270} y={228 - point[1] * 205} fill="currentColor" fontSize="11">D{index + 1}</text>
          </g>)}
          {centres.map((point, index) => <g key={index}>
            <rect x={16 + point[0] * 270} y={222 - point[1] * 205} width="12" height="12" fill={seriesColor(index, dark)} stroke="currentColor" />
            <text x={32 + point[0] * 270} y={222 - point[1] * 205} fill="currentColor" fontSize="11">C{index + 1}</text>
          </g>)}
        </svg>
        <div aria-live="polite">step {steps}; assignments: {assignments.map((group) => group + 1).join(', ')}</div>
      </div>}
    </VizPanel>
  );
}
