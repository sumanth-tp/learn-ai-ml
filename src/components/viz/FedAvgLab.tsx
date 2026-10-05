import {useState} from 'react';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';
import {seriesColor} from './palette';

function plot(values: number[], low: number, high: number): string {
  return values.map((v, i) => `${i ? 'L' : 'M'}${48 + i * 540 / Math.max(1, values.length - 1)},${230 - (v-low)*185/Math.max(1e-9,high-low)}`).join(' ');
}

export default function FedAvgLab() {
  const dark=useDarkViz();
  const [n1,setN1]=useState(100),[n2,setN2]=useState(300);
  const [w1,setW1]=useState(0.4),[w2,setW2]=useState(0.8);
  const average=(n1*w1+n2*w2)/(n1+n2),plain=(w1+w2)/2;
  const values=[w1,w2,average],labels=['Client 1','Client 2','FedAvg'];
  return <VizPanel title="Models weighted by client examples"
    hint={`FedAvg ${average.toFixed(6)}; unweighted mean ${plain.toFixed(6)}. This calculator implements one aggregation, not a privacy protocol.`}
    table={{columns:['Client','Examples','Weight','Model','Contribution'],rows:[[1,n1,(n1/(n1+n2)).toFixed(6),w1.toFixed(2),(n1*w1/(n1+n2)).toFixed(6)],[2,n2,(n2/(n1+n2)).toFixed(6),w2.toFixed(2),(n2*w2/(n1+n2)).toFixed(6)]]}}
    controls={<>
      <label className={s.control}>Client 1 examples<input type="range" min={10} max={400} step={10} value={n1} onChange={e=>setN1(Number(e.target.value))}/><span>{n1}</span></label>
      <label className={s.control}>Client 2 examples<input type="range" min={10} max={400} step={10} value={n2} onChange={e=>setN2(Number(e.target.value))}/><span>{n2}</span></label>
      <label className={s.control}>Client 1 model<input type="range" min={-1} max={1} step={0.05} value={w1} onChange={e=>setW1(Number(e.target.value))}/><span>{w1.toFixed(2)}</span></label>
      <label className={s.control}>Client 2 model<input type="range" min={-1} max={1} step={0.05} value={w2} onChange={e=>setW2(Number(e.target.value))}/><span>{w2.toFixed(2)}</span></label>
    </>}>
    <svg className={s.svg} viewBox="0 0 640 280" role="img" aria-label="Client model values and the weighted average"><title>FedAvg on a shared number line</title>
      {values.map((v,i)=><g key={i}><line x1={100} x2={600} y1={60+i*70} y2={60+i*70} stroke="var(--border-strong)"/><circle cx={100+(v+1)*250} cy={60+i*70} r={8} fill={seriesColor(i,dark)}/><text x={10} y={64+i*70} className={s.tick}>{labels[i]}</text><text x={100+(v+1)*250} y={44+i*70} textAnchor="middle" className={s.tick}>{v.toFixed(3)}</text></g>)}
      <text x={100} y={250} className={s.tick}>-1</text><text x={350} y={250} className={s.tick}>0</text><text x={600} y={250} textAnchor="end" className={s.tick}>1</text>
    </svg>
  </VizPanel>;
}
