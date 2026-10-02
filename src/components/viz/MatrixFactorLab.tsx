import {useState} from 'react';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';
import {seriesColor} from './palette';

const items = [{name: 'A', vector: [2, 1]}, {name: 'B', vector: [0, 2]}];
export default function MatrixFactorLab() {
  const dark = useDarkViz();
  const [first, setFirst] = useState(1);
  const scores = items.map(item => ({...item, parts: [first * item.vector[0], item.vector[1]], score: first * item.vector[0] + item.vector[1]}));
  return <VizPanel title="Score a tiny factor space"
    hint="These vectors are hand-set to show the dot product; a trained matrix factorisation model would learn them from interactions."
    controls={<label className={s.control}>User's first factor: {first.toFixed(1)}<input aria-label="User first factor" type="range" min="0" max="3" step="0.5" value={first} onChange={event => setFirst(Number(event.target.value))} /></label>}
    table={{columns: ['Item', 'Factors', 'Dimension products', 'Score'], rows: scores.map(item => [item.name, item.vector.join(', '), item.parts.join(' + '), item.score])}}>
    <div style={{display: 'grid', gap: '0.8rem'}}>
      {scores.map((item, index) => <div key={item.name}>
        <div>Item {item.name}: {item.score.toFixed(1)}</div>
        <div style={{height: '1rem', background: dark ? '#293142' : '#e7eaf0', borderRadius: '0.2rem'}}>
          <div style={{height: '100%', width: `${item.score / 7 * 100}%`, background: seriesColor(index, dark), borderRadius: '0.2rem'}} />
        </div>
      </div>)}
    </div>
  </VizPanel>;
}
