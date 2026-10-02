import {useState} from 'react';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';
import {seriesColor} from './palette';

export default function FeedbackConfidenceLab() {
  const dark = useDarkViz();
  const [count, setCount] = useState(2);
  const [alpha, setAlpha] = useState(2);
  const preference = Number(count > 0);
  const confidence = 1 + alpha * count;
  const rows = Array.from({length: 6}, (_, n) => [n, Number(n > 0), 1 + alpha * n, n === 0 ? 'unknown preference' : 'observed interaction']);
  return <VizPanel title="Separate preference from confidence"
    hint="An interaction is evidence of interest. An unseen item has unknown preference, not a recorded dislike."
    controls={<div className={s.controls}>
      <label className={s.control}>Interaction count: {count}<input aria-label="Interaction count" type="range" min="0" max="5" step="1" value={count} onChange={event => setCount(Number(event.target.value))} /></label>
      <label className={s.control}>Confidence alpha: {alpha}<input aria-label="Confidence alpha" type="range" min="1" max="5" step="1" value={alpha} onChange={event => setAlpha(Number(event.target.value))} /></label>
    </div>}
    table={{columns: ['Count', 'Preference', 'Confidence', 'Meaning'], rows}}>
    <div style={{display: 'grid', gap: '0.7rem'}}>
      <div>Inferred binary preference: <strong>{preference}</strong></div>
      <div>Confidence: <strong>{confidence}</strong></div>
      <div style={{height: '1.4rem', maxWidth: '100%', background: dark ? '#293142' : '#e7eaf0', borderRadius: '0.25rem'}}>
        <div style={{width: `${confidence / 26 * 100}%`, height: '100%', background: seriesColor(0, dark), borderRadius: '0.25rem'}} />
      </div>
    </div>
  </VizPanel>;
}
