import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function CrossModalSimilarityLab() {
  const dark = useDarkViz();
  const [extra, setExtra] = useState(1);
  const query = [1, 0, 1, 0];
  const image = [1, extra, 1, 0];
  const dot = query.reduce((sum, value, index) => sum + value * image[index], 0);
  const qLength = Math.sqrt(query.reduce((sum, value) => sum + value * value, 0));
  const iLength = Math.sqrt(image.reduce((sum, value) => sum + value * value, 0));
  const cosine = dot / (qLength * iLength);
  const table = {columns: ['coordinate', 'text vector', 'image vector'], rows: query.map((value, index) => [index + 1, value, image[index]])};

  return <VizPanel title="Compare a text and image vector"
    hint="These four numbers reproduce the lecture's cosine arithmetic. They are illustrative vectors, not outputs from CLIP or ALIGN."
    table={table}
    controls={<div className={s.controls}>
      <label className={s.control}>image-only extra coordinate: {extra.toFixed(1)}
        <input type="range" min="0" max="3" step="0.1" value={extra} aria-label="Image-only vector coordinate"
          onChange={(event) => setExtra(Number(event.target.value))} />
      </label>
    </div>}>
    <div style={{display: 'grid', gap: '0.75rem'}}>
      <div>text q = [{query.join(', ')}]</div>
      <div>image d = [{image.map((value) => value.toFixed(1)).join(', ')}]</div>
      <div style={{height: '1.4rem', background: 'var(--ifm-color-emphasis-200)', borderRadius: '0.25rem'}}>
        <div style={{height: '100%', width: `${cosine * 100}%`, background: seriesColor(0, dark), borderRadius: '0.25rem'}} />
      </div>
      <div aria-live="polite">dot product {dot.toFixed(0)}; cosine <strong>{cosine.toFixed(3)}</strong></div>
    </div>
  </VizPanel>;
}
