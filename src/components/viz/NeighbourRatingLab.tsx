import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function NeighbourRatingLab() {
  const dark = useDarkViz();
  const [ratings, setRatings] = useState([4, 5]);
  const [similarities, setSimilarities] = useState([0.8, 0.6]);
  const numerator = ratings.reduce((sum, rating, index) => sum + rating * similarities[index], 0);
  const denominator = similarities[0] + similarities[1];
  const prediction = numerator / denominator;
  const setRating = (index: number, value: number) => setRatings((current) => current.map((rating, position) => position === index ? value : rating));
  const setSimilarity = (index: number, value: number) => setSimilarities((current) => current.map((similarity, position) => position === index ? value : similarity));
  const table = {columns: ['neighbour', 'rating', 'similarity', 'weighted contribution'], rows: ratings.map((rating, index) => [
    index + 1, rating, similarities[index].toFixed(1), (rating * similarities[index]).toFixed(2),
  ])};

  return <VizPanel title="Predict from two neighbours"
    hint="A similarity-weighted average predicts a rating from the ratings of two neighbours. It does not handle user or item bias on its own."
    table={table}
    controls={<div className={s.controls}>
      {ratings.map((rating, index) => <div key={index} style={{display: 'flex', flexWrap: 'wrap', gap: '0.8rem'}}>
        <label className={s.control}>neighbour {index + 1} rating: {rating}
          <input type="range" min="1" max="5" step="1" value={rating} aria-label={`Neighbour ${index + 1} rating`}
            onChange={(event) => setRating(index, Number(event.target.value))} />
        </label>
        <label className={s.control}>similarity: {similarities[index].toFixed(1)}
          <input type="range" min="0.1" max="1" step="0.1" value={similarities[index]} aria-label={`Neighbour ${index + 1} similarity`}
            onChange={(event) => setSimilarity(index, Number(event.target.value))} />
        </label>
      </div>)}
    </div>}>
    <div style={{display: 'grid', gap: '0.7rem'}}>
      {ratings.map((rating, index) => <div key={index} style={{display: 'grid', gridTemplateColumns: '6rem minmax(0, 1fr) 4rem', gap: '0.5rem', alignItems: 'center'}}>
        <span>neighbour {index + 1}</span><div style={{height: '1.25rem', background: 'var(--ifm-color-emphasis-200)', borderRadius: '0.25rem'}}>
          <div style={{height: '100%', width: `${rating * similarities[index] / 5 * 100}%`, background: seriesColor(index, dark), borderRadius: '0.25rem'}} />
        </div><span>{(rating * similarities[index]).toFixed(2)}</span>
      </div>)}
      <div aria-live="polite">{numerator.toFixed(2)} / {denominator.toFixed(2)} = predicted rating <strong>{prediction.toFixed(2)}</strong></div>
    </div>
  </VizPanel>;
}
