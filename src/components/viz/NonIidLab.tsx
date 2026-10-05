import {useState} from 'react';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';
import {seriesColor} from './palette';

function plot(values: number[], low: number, high: number): string {
  return values.map((v, i) => `${i ? 'L' : 'M'}${48 + i * 540 / Math.max(1, values.length - 1)},${230 - (v-low)*185/Math.max(1e-9,high-low)}`).join(' ');
}

export default function NonIidLab() {
  const dark=useDarkViz();
  const [period,setPeriod]=useState(4),[rate,setRate]=useState(0.1);
  const [gap,setGap]=useState(1);
  const targets=[-gap,gap], curvature=[1,4];
  const optimum=(targets[0]+4*targets[1])/5;
  let w=0;
  const globals=[0], first=[0], second=[0];
  const rows: (string|number)[][]=[];
  for(let start=0;start<24;start+=period) {
    let local=[w,w];
    for(let i=0;i<Math.min(period,24-start);i++) {
      local=local.map((v,j)=>v-rate*curvature[j]*(v-targets[j]));
      first.push(local[0]);second.push(local[1]);
    }
    w=(local[0]+local[1])/2;
    globals.push(w);
    rows.push([Math.min(start+period,24),local[0].toFixed(6),local[1].toFixed(6),w.toFixed(6),Math.abs(local[1]-local[0]).toFixed(6),(1.25*(w-optimum)**2).toFixed(6)]);
  }
  const drawn=globals, low=Math.min(...drawn,0,optimum),high=Math.max(...drawn,1,optimum);
  return <VizPanel title="Heterogeneity and communication frequency"
    hint={`Rounds ${rows.length}; global ${w.toFixed(6)}; central optimum ${optimum.toFixed(6)}; excess loss ${(1.25*(w-optimum)**2).toFixed(6)}. Synthetic quadratic objectives.`}
    legend={[{label:'Global model',color:seriesColor(0,dark)},{label:'Central optimum',color:seriesColor(2,dark)}]}
    table={{columns:['Local steps','Client 1','Client 2','Average','Client gap','Excess loss'],rows}}
    controls={<>
      <label className={s.control}>Local steps per round<input type="range" min={1} max={8} value={period} onChange={e=>setPeriod(Number(e.target.value))}/><span>{period}</span></label>
      <label className={s.control}>Learning rate<input type="range" min={0.02} max={0.2} step={0.02} value={rate} onChange={e=>setRate(Number(e.target.value))}/><span>{rate.toFixed(2)}</span></label>
      <label className={s.control}>Target heterogeneity<input type="range" min={0} max={3} step={0.25} value={gap} onChange={e=>setGap(Number(e.target.value))}/><span>{gap.toFixed(2)}</span></label>
    </>}>
    <svg className={s.svg} viewBox="0 0 640 280" role="img" aria-label="Local training and global averaging trajectory"><title>Heterogeneity and communication frequency</title>
      <path d={plot(globals,low,high)} fill="none" stroke={seriesColor(0,dark)} strokeWidth={3}/>
      <path d={plot([optimum,optimum],low,high)} fill="none" stroke={seriesColor(2,dark)} strokeDasharray="5 4" strokeWidth={2}/>
      <text x={48} y={258} className={s.tick}>start</text><text x={588} y={258} textAnchor="end" className={s.tick}>{rows.length} communication rounds</text>
      <text x={16} y={32} className={s.tick}>{high.toFixed(1)}</text><text x={16} y={230} className={s.tick}>{low.toFixed(1)}</text>
    </svg>
  </VizPanel>;
}
