import {useState} from 'react';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';
import {seriesColor} from './palette';

const items = [{id: 'A', topic: 'X', score: 0.9}, {id: 'B', topic: 'X', score: 0.8}, {id: 'C', topic: 'Y', score: 0.7}];
export default function SlateDiversityLab() {
  const dark = useDarkViz();
  const [bonus, setBonus] = useState(0.2);
  const second = bonus > 0.1 ? items[2] : items[1];
  const baseline = items.slice(0, 2);
  const revised = [items[0], second];
  return <VizPanel title="Rerank a two-item slate"
    hint="A simple bonus for a new topic changes the second selection. It is an illustrative rule, not a measured user benefit."
    controls={<label className={s.control}>New-topic bonus: {bonus.toFixed(1)}<input aria-label="New topic bonus" type="range" min="0" max="0.4" step="0.1" value={bonus} onChange={event => setBonus(Number(event.target.value))} /></label>}
    table={{columns: ['Item', 'Topic', 'Base score', 'Second-position adjusted score', 'Selected'], rows: items.map(item => [item.id, item.topic, item.score.toFixed(1), (item.score + (item.topic === 'Y' ? bonus : 0)).toFixed(1), revised.some(chosen => chosen.id === item.id) ? 'yes' : 'no'])}}>
    <div style={{display: 'grid', gap: '0.6rem'}}>
      <div>Score-first: {baseline.map(item => item.id).join(', ')} · {new Set(baseline.map(item => item.topic)).size} topic</div>
      <div>Reranked: {revised.map(item => item.id).join(', ')} · {new Set(revised.map(item => item.topic)).size} {new Set(revised.map(item => item.topic)).size === 1 ? 'topic' : 'topics'}</div>
      <div style={{display: 'flex', gap: '0.5rem', flexWrap: 'wrap'}}>
        {revised.map((item, index) => <span key={item.id} style={{padding: '0.5rem 0.8rem', borderLeft: `0.35rem solid ${seriesColor(index, dark)}`, background: 'var(--ifm-color-emphasis-100)'}}>{item.id} · topic {item.topic}</span>)}
      </div>
    </div>
  </VizPanel>;
}
