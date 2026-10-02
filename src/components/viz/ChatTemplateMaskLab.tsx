import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Variant = {tokens: string[]; start: number; end_marker: number; nll: (number | null)[]};
type Example = {label: string; ticket: string; variants: {default: Variant; custom: Variant}};

export const EXAMPLES: Example[] = [{"label":"account","ticket":"Hi team, I forgot my password and the reset email never arrives.","variants":{"default":{"tokens":["<|im_start|>","system","\n","You"," are"," a"," helpful"," AI"," assistant"," named"," Sm","ol","LM",","," trained"," by"," H","ugging"," Face","<|im_end|>","\n","<|im_start|>","user","\n","Hi"," team",","," I"," forgot"," my"," password"," and"," the"," reset"," email"," never"," arrives",".","<|im_end|>","\n","<|im_start|>","ass","istant","\n","{\"","queue","\":"," \"","account","\"}","<|im_end|>","\n"],"start":44,"end_marker":50,"nll":[null,6.51437,0.2809,3.57068,0.57131,1.87129,2.12137,2.60187,0.51682,15.32613,9.82469,12.86573,22.14724,1.50046,6.78872,7.81553,8.10808,6.70121,1.2033,12.23333,0.00962,0.03035,2.4576,0.00187,7.41013,13.21937,0.02552,1.87614,10.22961,0.95545,2.09625,2.15673,3.99618,9.01598,3.07411,7.58293,3.31175,0.12602,4.77982,0.00012,0.00091,1e-05,0.0,2e-05,13.28206,10.36635,0.21393,1.19024,7.59225,4.56504,0.3992,0.00096]},"custom":{"tokens":["<|im_start|>","system","\n","Rout","e"," the"," ticket",".","<|im_end|>","\n","<|im_start|>","user","\n","Hi"," team",","," I"," forgot"," my"," password"," and"," the"," reset"," email"," never"," arrives",".","<|im_end|>","\n","<|im_start|>","ass","istant","\n","{\"","queue","\":"," \"","account","\"}","<|im_end|>","\n"],"start":33,"end_marker":39,"nll":[null,6.51437,0.2809,15.08132,0.41309,7.40848,9.93123,5.01916,3.74385,0.0002,0.13046,1.90505,0.00038,5.39503,9.68821,0.13261,1.45749,8.52261,1.99038,2.93682,2.17662,3.86486,8.46449,2.92128,8.60044,3.97386,0.20883,2.67401,2e-05,0.00169,1e-05,0.0,2e-05,16.58596,8.1777,0.31194,1.21869,6.86636,4.5289,2.15039,7e-05]}}},{"label":"billing","ticket":"Hello, I was charged twice for my subscription this month. Thanks.","variants":{"default":{"tokens":["<|im_start|>","system","\n","You"," are"," a"," helpful"," AI"," assistant"," named"," Sm","ol","LM",","," trained"," by"," H","ugging"," Face","<|im_end|>","\n","<|im_start|>","user","\n","Hello",","," I"," was"," charged"," twice"," for"," my"," subscription"," this"," month","."," Thanks",".","<|im_end|>","\n","<|im_start|>","ass","istant","\n","{\"","queue","\":"," \"","b","illing","\"}","<|im_end|>","\n"],"start":44,"end_marker":51,"nll":[null,6.51437,0.2809,3.57068,0.57131,1.87129,2.12137,2.60187,0.51682,15.32613,9.82469,12.86573,22.14724,1.50046,6.78872,7.81553,8.10808,6.70121,1.2033,12.23333,0.00962,0.03035,2.4576,0.00187,6.89989,0.20874,0.41097,3.96423,9.74864,7.13747,0.93091,1.18923,8.978,9.24721,1.41979,0.64485,4.95125,3.28085,1.37941,7e-05,0.00065,1e-05,0.0,1e-05,12.26768,10.1177,0.4899,0.97747,5.16209,3.23968,5.15368,0.32588,0.00074]},"custom":{"tokens":["<|im_start|>","system","\n","Rout","e"," the"," ticket",".","<|im_end|>","\n","<|im_start|>","user","\n","Hello",","," I"," was"," charged"," twice"," for"," my"," subscription"," this"," month","."," Thanks",".","<|im_end|>","\n","<|im_start|>","ass","istant","\n","{\"","queue","\":"," \"","b","illing","\"}","<|im_end|>","\n"],"start":33,"end_marker":40,"nll":[null,6.51437,0.2809,15.08132,0.41309,7.40848,9.93123,5.01916,3.74385,0.0002,0.13046,1.90505,0.00038,6.6489,0.76964,0.25075,4.36602,9.30004,6.77452,0.6464,1.62409,7.32554,7.20438,1.61949,0.69515,6.26796,2.09558,0.6542,1e-05,0.00055,0.0,0.0,1e-05,14.83848,8.16015,0.89723,1.03571,5.03252,3.72293,5.01808,1.40474,8e-05]}}},{"label":"technical","ticket":"The app crashes every time I open the dashboard. Please help.","variants":{"default":{"tokens":["<|im_start|>","system","\n","You"," are"," a"," helpful"," AI"," assistant"," named"," Sm","ol","LM",","," trained"," by"," H","ugging"," Face","<|im_end|>","\n","<|im_start|>","user","\n","The"," app"," crashes"," every"," time"," I"," open"," the"," dashboard","."," Please"," help",".","<|im_end|>","\n","<|im_start|>","ass","istant","\n","{\"","queue","\":"," \"","technical","\"}","<|im_end|>","\n"],"start":43,"end_marker":49,"nll":[null,6.51437,0.2809,3.57068,0.57131,1.87129,2.12137,2.60187,0.51682,15.32613,9.82469,12.86573,22.14724,1.50046,6.78872,7.81553,8.10808,6.70121,1.2033,12.23333,0.00962,0.03035,2.4576,0.00187,6.56912,6.7678,5.55628,2.96152,0.24483,0.41364,3.48082,2.12647,6.67419,0.96065,6.20816,0.64848,2.57676,0.45489,0.00016,0.00073,1e-05,0.0,2e-05,12.96142,9.23501,0.37978,0.93355,9.71417,5.61933,0.33746,0.00095]},"custom":{"tokens":["<|im_start|>","system","\n","Rout","e"," the"," ticket",".","<|im_end|>","\n","<|im_start|>","user","\n","The"," app"," crashes"," every"," time"," I"," open"," the"," dashboard","."," Please"," help",".","<|im_end|>","\n","<|im_start|>","ass","istant","\n","{\"","queue","\":"," \"","technical","\"}","<|im_end|>","\n"],"start":32,"end_marker":38,"nll":[null,6.51437,0.2809,15.08132,0.41309,7.40848,9.93123,5.01916,3.74385,0.0002,0.13046,1.90505,0.00038,3.36671,7.97861,4.89473,2.83708,0.22515,1.09891,4.51112,1.13871,6.02157,0.92549,7.63174,1.22575,1.42928,0.75859,1e-05,0.00078,1e-05,0.0,1e-05,15.80589,7.6613,0.42945,0.93958,9.57595,5.4337,1.53158,0.00024]}}}];

export type Mode = 'all' | 'completion' | 'markers';

export const MODES: {value: Mode; label: string}[] = [
  {value: 'completion', label: 'Completion only (TRL default)'},
  {value: 'markers', label: 'Assistant markers (patched)'},
  {value: 'all', label: 'Every token'},
];

export function trainedFlags(v: Variant, mode: Mode): boolean[] {
  return v.tokens.map((_, i) => {
    if (mode === 'all') return i > 0;
    if (mode === 'completion') return i >= v.start;
    return i >= v.start && i <= v.end_marker;
  });
}

export function meanLoss(v: Variant, flags: boolean[]): number {
  let total = 0;
  let count = 0;
  v.nll.forEach((value, i) => {
    if (flags[i] && value !== null) {
      total += value;
      count += 1;
    }
  });
  return count ? total / count : 0;
}

const show = (t: string) => t.replace(/\n/g, '\\n');

export default function ChatTemplateMaskLab() {
  const dark = useDarkViz();
  const [exampleIndex, setExampleIndex] = useState(0);
  const [system, setSystem] = useState<'default' | 'custom'>('default');
  const [mode, setMode] = useState<Mode>('completion');

  const example = EXAMPLES[exampleIndex];
  const variant = example.variants[system];
  const flags = trainedFlags(variant, mode);
  const trained = flags.filter(Boolean).length;
  const total = variant.tokens.length;
  const lossTrained = meanLoss(variant, flags);
  const lossAll = meanLoss(variant, trainedFlags(variant, 'all'));

  const on = seriesColor(2, dark);
  const status = `${total} tokens, ${trained} carry loss (${Math.round((trained / total) * 100)}%). Mean loss over the trained tokens ${lossTrained.toFixed(4)}; over every token ${lossAll.toFixed(4)}.`;

  const rows = variant.tokens.map((t, i) => [
    i,
    show(t),
    flags[i] ? 'trained' : 'masked',
    variant.nll[i] === null ? '-' : (variant.nll[i] as number).toFixed(3),
  ]);

  return (
    <VizPanel
      title="Chat template and loss mask, token by token"
      hint="Default: the account ticket with the template's own system prompt and completion-only loss gives 52 tokens, 8 trained, loss 4.7013 over the trained tokens and 4.5594 over every token. Highlighted tokens train; faded ones are masked with label -100."
      legend={[
        {label: 'trained (label kept)', color: on},
        {label: 'masked (label -100)', color: dark ? '#848c99' : '#9aa0a6'},
      ]}
      table={{columns: ['position', 'token', 'status', 'loss on this token'], rows}}
      controls={
        <>
          <label className={s.control}>
            ticket
            <select className={s.select} value={exampleIndex} onChange={(e) => setExampleIndex(Number(e.target.value))}>
              {EXAMPLES.map((ex, i) => (
                <option key={ex.label} value={i}>
                  {ex.label}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            system prompt
            <select className={s.select} value={system} onChange={(e) => setSystem(e.target.value as 'default' | 'custom')}>
              <option value="default">template default</option>
              <option value="custom">Route the ticket.</option>
            </select>
          </label>
          <label className={s.control}>
            loss mask
            <select className={s.select} value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              {MODES.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        </>
      }>
      <div
        role="img"
        aria-label={`Token strip. ${status}`}
        style={{display: 'flex', flexWrap: 'wrap', gap: '3px', fontFamily: 'var(--ifm-font-family-monospace)', fontSize: '0.78rem'}}>
        {variant.tokens.map((t, i) => (
          <span
            key={i}
            title={`position ${i}: ${flags[i] ? 'trained' : 'masked'}`}
            style={{
              padding: '2px 5px',
              borderRadius: 4,
              whiteSpace: 'pre',
              border: `1.5px solid ${flags[i] ? on : 'var(--border-strong)'}`,
              background: flags[i] ? `color-mix(in srgb, ${on} 22%, transparent)` : 'transparent',
              opacity: flags[i] ? 1 : 0.55,
            }}>
            {show(t)}
          </span>
        ))}
      </div>
      <p className={s.value} style={{padding: '0.6rem 0 0'}} aria-live="polite">
        {status}
      </p>
    </VizPanel>
  );
}
