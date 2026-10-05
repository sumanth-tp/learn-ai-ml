import {useState} from 'react';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';
import {seriesColor} from './palette';

function plot(values: number[], low: number, high: number): string {
  return values.map((v, i) => `${i ? 'L' : 'M'}${48 + i * 540 / Math.max(1, values.length - 1)},${230 - (v-low)*185/Math.max(1e-9,high-low)}`).join(' ');
}

const gradient = [0.12,-0.7,1.5,-2,0.01,0.8,-0.4,0.3];
export default function GradientCompressionLab() {
  const dark=useDarkViz();
  const [bits,setBits]=useState(8), [steps,setSteps]=useState(6), [feedback,setFeedback]=useState(true);
  let residual=gradient.map(()=>0), total=gradient.map(()=>0), firstError=0;
  for(let i=0;i<steps;i++) {
    const corrected=gradient.map((v,j)=>v+(feedback?residual[j]:0));
    const scale=Math.max(...corrected.map(Math.abs))/(2**(bits-1)-1);
    const sent=corrected.map(v=>scale===0?0:Math.sign(v)*Math.floor(Math.abs(v)/scale+0.5)*scale);
    residual=corrected.map((v,j)=>v-sent[j]);
    total=total.map((v,j)=>v+sent[j]);
    if(i===0) firstError=Math.max(...sent.map((v,j)=>Math.abs(v-gradient[j])));
  }
  const desired=gradient.map(v=>steps*v), low=Math.min(...desired,...total,0),high=Math.max(...desired,...total,1);
  const error=Math.sqrt(total.reduce((a,v,j)=>a+(desired[j]-v)**2,0));
  return <VizPanel title="Quantisation and residual feedback"
    hint={`First max error ${firstError.toFixed(6)}; cumulative error norm ${error.toFixed(6)}. Raw 32 bytes; packed payload ${Math.ceil(gradient.length*bits/8)} bytes plus 4 bytes for a scale per transmission.`}
    legend={[{label:'Desired cumulative gradient',color:seriesColor(0,dark)},{label:'Decoded cumulative gradient',color:seriesColor(1,dark)}]}
    table={{columns:['Coordinate','Gradient','Desired sum','Sent sum','Unsent sum'],rows:gradient.map((v,j)=>[j,v.toFixed(2),desired[j].toFixed(6),total[j].toFixed(6),(desired[j]-total[j]).toFixed(6)])}}
    controls={<>
      <label className={s.control}>Signed bits<input type="range" min={2} max={16} value={bits} onChange={e=>setBits(Number(e.target.value))}/><span>{bits}</span></label>
      <label className={s.control}>Transmissions<input type="range" min={1} max={12} value={steps} onChange={e=>setSteps(Number(e.target.value))}/><span>{steps}</span></label>
      <label className={s.control}><input type="checkbox" checked={feedback} onChange={e=>setFeedback(e.target.checked)}/>Error feedback</label>
    </>}>
    <svg className={s.svg} viewBox="0 0 640 280" role="img" aria-label="Desired and decoded cumulative gradients by coordinate"><title>Quantisation error by coordinate</title>
      <path d={plot(desired,low,high)} fill="none" stroke={seriesColor(0,dark)} strokeWidth={3}/>
      <path d={plot(total,low,high)} fill="none" stroke={seriesColor(1,dark)} strokeWidth={2} strokeDasharray="5 4"/>
      <text x={48} y={258} className={s.tick}>coordinate 0</text><text x={588} y={258} textAnchor="end" className={s.tick}>coordinate 7</text>
      <text x={16} y={32} className={s.tick}>{high.toFixed(1)}</text><text x={16} y={230} className={s.tick}>{low.toFixed(1)}</text>
    </svg>
  </VizPanel>;
}
