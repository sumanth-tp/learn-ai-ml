import {useMemo, useState} from 'react';

import {DIVERGING, sequentialColor, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Lab = 'bert' | 'gpt1' | 'gpt2' | 'gpt3' | 'lora' | 'react' | 'clip'
  | 'llama' | 'grpo' | 'resnet' | 'ddpm';

const bar = (width: number, color: string) => ({
  width: `${Math.max(1, Math.min(100, width))}%`, height: '0.72rem',
  borderRadius: '999px', background: color, transition: 'width 180ms ease',
});

function BertLab() {
  const dark = useDarkViz();
  const [pattern, setPattern] = useState<'mask' | 'random' | 'same'>('mask');
  const words = ['the', 'customer', 'reset', 'the', 'account', 'after', 'the', 'login', 'failed', 'today'];
  const selected = new Set([1, 4, 8]);
  const shown = words.map((word, i) => !selected.has(i) ? word
    : pattern === 'mask' ? '[MASK]' : pattern === 'random' ? ['river', 'green', 'ticket'][[1, 4, 8].indexOf(i)] : word);
  return <VizPanel title="Masked-language modelling: corruption is not the target"
    hint="The original words remain the labels at all selected positions. Change the visible corruption and notice that the prediction targets do not change."
    controls={<label className={s.control}>selected-token treatment
      <select className={s.select} value={pattern} onChange={(e) => setPattern(e.target.value as typeof pattern)}>
        <option value="mask">replace with [MASK] (80%)</option><option value="random">random token (10%)</option><option value="same">leave unchanged (10%)</option>
      </select></label>}>
    <div style={{display: 'grid', gridTemplateColumns: 'repeat(5, minmax(5rem,1fr))', gap: '.55rem'}}>
      {shown.map((word, i) => <div key={i} style={{padding: '.55rem', borderRadius: '.55rem', textAlign: 'center',
        border: `1px solid ${selected.has(i) ? seriesColor(0, dark) : 'var(--border-subtle)'}`}}>
        <div style={{fontWeight: 650}}>{word}</div><small style={{color: 'var(--text-faint)'}}>
          {selected.has(i) ? `target: ${words[i]}` : 'context only'}</small>
      </div>)}
    </div>
  </VizPanel>;
}

function PromptLab({paper}: {paper: 'gpt1' | 'gpt2' | 'gpt3'}) {
  const [examples, setExamples] = useState(paper === 'gpt3' ? 3 : 0);
  const [temperature, setTemperature] = useState(0.8);
  const modes = {
    gpt1: ['Pre-train on text', 'Update weights on labelled task', 'Use the fine-tuned checkpoint'],
    gpt2: ['Pre-train on diverse text', 'Describe task in the prefix', 'Decode a continuation'],
    gpt3: ['Pre-train once at scale', 'Put demonstrations in context', 'Answer with fixed weights'],
  };
  const confidence = Math.min(96, 44 + examples * 11 + (1.4 - temperature) * 15);
  return <VizPanel title={`${paper.toUpperCase()}: what changes when the task changes`}
    hint={paper === 'gpt1' ? 'GPT-1 changes parameters during supervised fine-tuning.' : 'The task lives in the prefix/context; ordinary evaluation does not update parameters.'}
    controls={<><label className={s.control}>demonstrations
      <input type="range" min={0} max={5} value={examples} onChange={(e) => setExamples(Number(e.target.value))}/><span className={s.value}>{examples}</span>
    </label><label className={s.control}>temperature
      <input type="range" min={0.2} max={1.5} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))}/><span className={s.value}>{temperature.toFixed(1)}</span>
    </label></>}>
    <div style={{display: 'flex', gap: '.6rem', flexWrap: 'wrap', alignItems: 'center'}}>
      {modes[paper].map((item, i) => <div key={item} style={{display: 'flex', alignItems: 'center', gap: '.6rem'}}>
        <div style={{padding: '.65rem .8rem', border: '1px solid var(--border-subtle)', borderRadius: '.55rem'}}><strong>{i + 1}</strong> · {item}</div>
        {i < 2 && <span aria-hidden="true">→</span>}
      </div>)}
    </div>
    <div style={{marginTop: '1rem'}}><small>Illustrative task-pattern strength</small>
      <div style={{background: 'var(--surface-2)', borderRadius: '999px'}}><div style={bar(confidence, 'var(--ifm-color-primary)')}/></div>
      <small style={{color: 'var(--text-faint)'}}>More useful demonstrations can clarify a pattern; higher temperature increases output variability. This is an intuition aid, not a benchmark predictor.</small>
    </div>
  </VizPanel>;
}

function LoRALab() {
  const [width, setWidth] = useState(4096);
  const [rank, setRank] = useState(8);
  const base = width * width;
  const adapter = rank * (width + width);
  return <VizPanel title="LoRA parameter and rank calculator"
    hint="Only BA is low-rank. The frozen base matrix remains full-rank and must still be available when an adapter is loaded."
    controls={<><label className={s.control}>matrix width
      <select className={s.select} value={width} onChange={(e) => setWidth(Number(e.target.value))}>
        {[768, 1024, 4096, 8192].map((v) => <option key={v}>{v}</option>)}</select></label>
      <label className={s.control}>rank <input type="range" min={1} max={128} value={rank} onChange={(e) => setRank(Number(e.target.value))}/><span className={s.value}>{rank}</span></label></>}>
    <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(12rem,1fr))', gap: '.7rem'}}>
      <div><strong>{base.toLocaleString()}</strong><br/><small>base parameters (frozen)</small></div>
      <div><strong>{adapter.toLocaleString()}</strong><br/><small>adapter parameters (trainable)</small></div>
      <div><strong>{(100 * adapter / base).toFixed(3)}%</strong><br/><small>trainable fraction for this matrix</small></div>
      <div><strong>≤ {rank}</strong><br/><small>maximum update rank</small></div>
    </div>
  </VizPanel>;
}

const TRACE = [
  ['Thought', 'I need the current refund rule, not a remembered answer.'],
  ['Action', 'search("damaged order shipping refund")'],
  ['Observation', 'refunds.md: damaged goods include delivery charges'],
  ['Thought', 'The evidence directly answers the question.'],
  ['Final', 'Yes—delivery charges are included for damaged goods.'],
];

function ReactLab() {
  const [step, setStep] = useState(1);
  return <VizPanel title="ReAct trace: model text versus environment evidence"
    hint="Only the runner may create an Observation. Advancing the trace shows why planning, tool execution and evidence must remain separate records."
    controls={<><button className={s.button} disabled={step >= TRACE.length} onClick={() => setStep((v) => Math.min(TRACE.length, v + 1))}>next event</button>
      <button className={s.button} onClick={() => setStep(1)}>reset</button></>}>
    <ol style={{margin: 0, paddingLeft: '1.4rem'}}>{TRACE.slice(0, step).map(([type, text], i) =>
      <li key={i} style={{margin: '.45rem 0'}}><strong>{type}:</strong> {text}</li>)}</ol>
  </VizPanel>;
}

function ClipLab() {
  const dark = useDarkViz();
  const [temperature, setTemperature] = useState(0.2);
  const raw = [[.9, .2, .1], [.15, .82, .3], [.1, .25, .88]];
  const probs = raw.map((row) => { const e = row.map((v) => Math.exp(v / temperature)); const z = e.reduce((a,b) => a+b,0); return e.map((v) => v/z); });
  return <VizPanel title="CLIP's all-pairs contrastive batch"
    hint="Diagonal cells are the supplied image–text pairs. Lower temperature sharpens the row probabilities but does not turn similarity into calibrated truth."
    controls={<label className={s.control}>temperature <input type="range" min={0.08} max={1} step={0.02} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))}/><span className={s.value}>{temperature.toFixed(2)}</span></label>}
    table={{columns: ['image', 'cat text', 'bike text', 'bowl text'], rows: probs.map((r,i) => [`image ${i+1}`, ...r.map((v) => v.toFixed(3))])}}>
    <div style={{display:'grid',gridTemplateColumns:'4rem repeat(3,1fr)',gap:'.35rem'}}>
      <span/>{['cat text','bike text','bowl text'].map((x)=><strong key={x} style={{textAlign:'center',fontSize:'.75rem'}}>{x}</strong>)}
      {probs.flatMap((row,i)=>[<strong key={`r${i}`} style={{fontSize:'.75rem'}}>image {i+1}</strong>, ...row.map((p,j)=><div key={`${i}-${j}`} style={{padding:'.75rem',textAlign:'center',borderRadius:'.4rem',background:sequentialColor(p,dark)}}>{p.toFixed(2)}</div>)])}
    </div>
  </VizPanel>;
}

function LlamaLab() {
  const [tokens, setTokens] = useState(4096);
  const [kvHeads, setKvHeads] = useState(32);
  const layers = 32, headDim = 128, bytes = 2;
  const mb = 2 * layers * tokens * kvHeads * headDim * bytes / 1024 / 1024;
  return <VizPanel title="Autoregressive KV-cache cost"
    hint="Caching avoids recomputing keys and values for the prefix, but cache memory grows linearly with sequence length, layers and KV heads."
    controls={<><label className={s.control}>tokens <input type="range" min={512} max={16384} step={512} value={tokens} onChange={(e)=>setTokens(Number(e.target.value))}/><span className={s.value}>{tokens}</span></label>
      <label className={s.control}>KV heads <select className={s.select} value={kvHeads} onChange={(e)=>setKvHeads(Number(e.target.value))}>{[4,8,16,32].map(v=><option key={v}>{v}</option>)}</select></label></>}>
    <p style={{fontSize:'1.25rem',margin:'.2rem 0'}}><strong>{mb >= 1024 ? `${(mb/1024).toFixed(2)} GiB` : `${mb.toFixed(0)} MiB`}</strong> approximate cache per sequence</p>
    <small style={{color:'var(--text-faint)'}}>Illustration: 32 layers, head width 128, FP16/BF16, separate K and V. Real layouts and parallelism change allocation.</small>
  </VizPanel>;
}

function GrpoLab() {
  const dark = useDarkViz();
  const [preset, setPreset] = useState('mixed');
  const rewards = preset === 'mixed' ? [0,1,1,0] : preset === 'equal' ? [1,1,1,1] : [0,.25,.75,1];
  const mean = rewards.reduce((a,b)=>a+b,0)/rewards.length;
  const sd = Math.sqrt(rewards.reduce((a,b)=>a+(b-mean)**2,0)/rewards.length);
  const advantages = rewards.map((r)=>(r-mean)/(sd+1e-8));
  return <VizPanel title="GRPO: compare responses within one prompt"
    hint={sd === 0 ? 'All rewards are equal, so every centered advantage is zero: the group supplies no ranking signal.' : 'Positive advantages encourage above-group responses; negative advantages discourage below-group responses.'}
    controls={<label className={s.control}>reward group <select className={s.select} value={preset} onChange={(e)=>setPreset(e.target.value)}><option value="mixed">[0, 1, 1, 0]</option><option value="graded">[0, .25, .75, 1]</option><option value="equal">[1, 1, 1, 1]</option></select></label>}
    table={{columns:['response','reward','advantage'],rows:rewards.map((r,i)=>[i+1,r.toFixed(2),advantages[i].toFixed(2)])}}>
    <div style={{display:'flex',gap:'.7rem',alignItems:'flex-end',height:'9rem'}}>{advantages.map((a,i)=><div key={i} style={{flex:1,textAlign:'center'}}>
      <div style={{height:`${Math.abs(a)*3+1}rem`,background:a>0?DIVERGING[dark?'dark':'light'].positive:a<0?DIVERGING[dark?'dark':'light'].negative:DIVERGING[dark?'dark':'light'].mid,borderRadius:'.35rem .35rem 0 0'}}/>
      <small>r={rewards[i]}<br/>A={a.toFixed(2)}</small></div>)}</div>
  </VizPanel>;
}

function ResNetLab() {
  const [x, setX] = useState(1);
  const [residual, setResidual] = useState(.2);
  const output = x + residual;
  return <VizPanel title="Residual learning: preserve x, learn the change"
    hint="When the desired mapping is close to identity, the residual branch can stay small. The shortcut also contributes an identity term to the block's Jacobian."
    controls={<><label className={s.control}>input x <input type="range" min={-2} max={2} step={.1} value={x} onChange={(e)=>setX(Number(e.target.value))}/><span className={s.value}>{x.toFixed(1)}</span></label>
      <label className={s.control}>F(x) <input type="range" min={-1} max={1} step={.1} value={residual} onChange={(e)=>setResidual(Number(e.target.value))}/><span className={s.value}>{residual.toFixed(1)}</span></label></>}>
    <div style={{display:'flex',alignItems:'center',justifyContent:'center',gap:'.8rem',fontSize:'1.1rem',flexWrap:'wrap'}}>
      <code>x = {x.toFixed(1)}</code><span>+</span><code>F(x) = {residual.toFixed(1)}</code><span>→</span><strong>H(x) = {output.toFixed(1)}</strong>
    </div>
  </VizPanel>;
}

function DdpmLab() {
  const dark = useDarkViz();
  const [t, setT] = useState(250);
  const alphaBar = Math.exp(-5 * t / 1000);
  const signal = Math.sqrt(alphaBar), noise = Math.sqrt(1-alphaBar);
  return <VizPanel title="Forward diffusion: signal gives way to noise"
    hint="Training samples one timestep and constructs xₜ directly. The coefficients are square roots because alpha-bar describes retained variance."
    controls={<label className={s.control}>timestep t <input type="range" min={0} max={1000} step={10} value={t} onChange={(e)=>setT(Number(e.target.value))}/><span className={s.value}>{t}</span></label>}
    table={{columns:['quantity','coefficient'],rows:[['clean signal √ᾱₜ',signal.toFixed(3)],['noise √(1−ᾱₜ)',noise.toFixed(3)]]}}>
    <div style={{display:'grid',gap:'.75rem'}}><div><small>clean signal coefficient: {signal.toFixed(3)}</small><div style={{background:'var(--surface-2)',borderRadius:'999px'}}><div style={bar(signal*100,seriesColor(0,dark))}/></div></div>
      <div><small>noise coefficient: {noise.toFixed(3)}</small><div style={{background:'var(--surface-2)',borderRadius:'999px'}}><div style={bar(noise*100,seriesColor(1,dark))}/></div></div></div>
  </VizPanel>;
}

export default function ResearchPaperLab({lab}: {lab: Lab}) {
  if (lab === 'bert') return <BertLab/>;
  if (lab === 'gpt1' || lab === 'gpt2' || lab === 'gpt3') return <PromptLab paper={lab}/>;
  if (lab === 'lora') return <LoRALab/>;
  if (lab === 'react') return <ReactLab/>;
  if (lab === 'clip') return <ClipLab/>;
  if (lab === 'llama') return <LlamaLab/>;
  if (lab === 'grpo') return <GrpoLab/>;
  if (lab === 'resnet') return <ResNetLab/>;
  return <DdpmLab/>;
}

