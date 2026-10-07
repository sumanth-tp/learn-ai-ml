import {useState} from 'react';

import {shopTrace} from './ollamaShopMath';
import VizPanel, {vizStyles as s} from './VizPanel';

export default function OllamaShopLab() {
  const [product, setProduct] = useState('laptop');
  const [years, setYears] = useState(5);
  const [continueAfterInventory, setContinueAfterInventory] = useState(true);
  const result = shopTrace(product, years, continueAfterInventory);

  return (
    <div data-testid="ollama-shop-lab">
      <VizPanel
        title="Follow the shop's tool results"
        hint="This teaching lab shows the required calls and the recorded discount rule. It does not run a language model. Read the result of each Python function before following the next request."
        controls={
          <>
            <label className={s.control}>
              Product
              <select className={s.select} value={product} onChange={(e) => setProduct(e.target.value)}>
                {['laptop', 'monitor', 'keyboard', 'iPhone'].map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            </label>
            <label className={s.control}>
              Customer years: {years}
              <input aria-label="Customer years" type="range" min={0} max={10} step={1} value={years} onChange={(e) => setYears(Number(e.target.value))} />
            </label>
            <label className={s.control}>
              <input type="checkbox" checked={continueAfterInventory} onChange={(e) => setContinueAfterInventory(e.target.checked)} />
              Continue after inventory
            </label>
          </>
        }
      >
        <div style={{padding: '1rem'}}>
          <p aria-live="polite" data-testid="shop-price">
            <strong>Price from an executed discount: </strong>
            {result.computedPrice === null ? 'not available' : result.computedPrice}
          </p>
          <p>Rule: 5% per year, capped at 30%. For {years} years: {result.discountPercent}%.</p>
          <ol>
            {result.steps.map(([request, title, detail], index) => (
              <li key={index} style={{marginBottom: '0.8rem'}}>
                <strong>Request {request}: {title}.</strong> {detail}
              </li>
            ))}
          </ol>
        </div>
      </VizPanel>
    </div>
  );
}
