import {useState} from 'react';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';
import {seriesColor} from './palette';

function plot(values: number[], low: number, high: number): string {
  return values.map((v, i) => `${i ? 'L' : 'M'}${48 + i * 540 / Math.max(1, values.length - 1)},${230 - (v-low)*185/Math.max(1e-9,high-low)}`).join(' ');
}

export default function StaleGradientLab() {
  const dark = useDarkViz();
  const [delay, setDelay] = useState(2);
  const [rate, setRate] = useState(0.4);
  const [steps, setSteps] = useState(20);
  const delayed = [1], fresh = [1];
  for (let i=0; i<steps; i++) {
    delayed.push(delayed[i]-rate*delayed[Math.max(0,i-delay)]);
    fresh.push(fresh[i]*(1-rate));
  }
  const low = Math.min(0,...delayed,...fresh), high = Math.max(1,...delayed,...fresh);
  return <VizPanel title="Accepted updates with stale gradients"
    hint={`Delayed final loss ${(0.5*delayed[steps]**2).toFixed(6)}; fresh final loss ${(0.5*fresh[steps]**2).toFixed(6)}. Unavailable history uses the initial weight.`}
    legend={[{label:'Delayed weight',color:seriesColor(0,dark)},{label:'Fresh weight',color:seriesColor(1,dark)}]}
    table={{columns:['Update','Delayed weight','Delayed loss','Fresh weight','Fresh loss'],rows:delayed.map((v,i)=>[i,v.toFixed(6),(0.5*v*v).toFixed(6),fresh[i].toFixed(6),(0.5*fresh[i]**2).toFixed(6)])}}
    controls={<>
      <label className={s.control}>Delay in updates<input type="range" min={0} max={5} value={delay} onChange={e=>setDelay(Number(e.target.value))}/><span>{delay}</span></label>
      <label className={s.control}>Learning rate<input type="range" min={0.05} max={0.8} step={0.05} value={rate} onChange={e=>setRate(Number(e.target.value))}/><span>{rate.toFixed(2)}</span></label>
      <label className={s.control}>Accepted updates<input type="range" min={5} max={40} value={steps} onChange={e=>setSteps(Number(e.target.value))}/><span>{steps}</span></label>
    </>}>
    <svg className={s.svg} viewBox="0 0 640 280" role="img" aria-label="Model weight against accepted updates">
      <title>Delayed and fresh gradient trajectories</title>
      <path d={plot(Array(steps+1).fill(0),low,high)} stroke="var(--border-strong)" fill="none"/>
      <path d={plot(delayed,low,high)} stroke={seriesColor(0,dark)} strokeWidth={3} fill="none"/>
      <path d={plot(fresh,low,high)} stroke={seriesColor(1,dark)} strokeWidth={3} strokeDasharray="6 4" fill="none"/>
      <text x={48} y={258} className={s.tick}>0 updates</text><text x={588} y={258} textAnchor="end" className={s.tick}>{steps} updates</text>
      <text x={16} y={32} className={s.tick}>{high.toFixed(2)}</text><text x={16} y={230} className={s.tick}>{low.toFixed(2)}</text>
    </svg>
  </VizPanel>;
}
