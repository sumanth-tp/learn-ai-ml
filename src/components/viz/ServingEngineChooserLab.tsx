import {useMemo, useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Cell = 'yes' | 'no' | 'unknown' | 'backend';
type Need = 'openai' | 'structured' | 'multilora' | 'spec' | 'prefix' | 'tp';
type Hardware = 'nvidia' | 'amd' | 'apple' | 'cpu' | 'intel_gpu';

type Engine = {
  name: string;
  maintenance: boolean;
  hardware: Record<Hardware, Cell>;
  needs: Record<Need, Cell>;
};

const Y: Cell = 'yes';
const N: Cell = 'no';
const U: Cell = 'unknown';
const B: Cell = 'backend';

export const ENGINES: Engine[] = [
  {
    name: 'vLLM',
    maintenance: false,
    hardware: {nvidia: Y, amd: Y, apple: Y, cpu: Y, intel_gpu: Y},
    needs: {openai: Y, structured: Y, multilora: Y, spec: Y, prefix: Y, tp: Y},
  },
  {
    name: 'SGLang',
    maintenance: false,
    hardware: {nvidia: Y, amd: Y, apple: U, cpu: Y, intel_gpu: U},
    needs: {openai: Y, structured: Y, multilora: Y, spec: Y, prefix: Y, tp: Y},
  },
  {
    name: 'TGI',
    maintenance: true,
    hardware: {nvidia: Y, amd: Y, apple: U, cpu: U, intel_gpu: U},
    needs: {openai: Y, structured: Y, multilora: Y, spec: Y, prefix: Y, tp: Y},
  },
  {
    name: 'TensorRT-LLM',
    maintenance: false,
    hardware: {nvidia: Y, amd: N, apple: N, cpu: N, intel_gpu: N},
    needs: {openai: Y, structured: Y, multilora: Y, spec: Y, prefix: Y, tp: Y},
  },
  {
    name: 'llama.cpp',
    maintenance: false,
    hardware: {nvidia: Y, amd: Y, apple: Y, cpu: Y, intel_gpu: Y},
    needs: {openai: Y, structured: Y, multilora: Y, spec: Y, prefix: Y, tp: U},
  },
  {
    name: 'Ollama',
    maintenance: false,
    hardware: {nvidia: Y, amd: Y, apple: Y, cpu: U, intel_gpu: Y},
    needs: {openai: Y, structured: Y, multilora: U, spec: U, prefix: U, tp: U},
  },
  {
    name: 'Triton',
    maintenance: false,
    hardware: {nvidia: U, amd: U, apple: U, cpu: Y, intel_gpu: U},
    needs: {openai: U, structured: B, multilora: B, spec: B, prefix: B, tp: B},
  },
];

const HARDWARE: {key: Hardware; label: string}[] = [
  {key: 'nvidia', label: 'NVIDIA GPU'},
  {key: 'amd', label: 'AMD GPU'},
  {key: 'apple', label: 'Apple silicon'},
  {key: 'cpu', label: 'CPU only'},
  {key: 'intel_gpu', label: 'Intel GPU'},
];

const NEEDS: {key: Need; label: string; short: string}[] = [
  {key: 'openai', label: 'OpenAI-compatible API', short: 'API'},
  {key: 'structured', label: 'structured output', short: 'JSON'},
  {key: 'multilora', label: 'multi-LoRA', short: 'LoRA'},
  {key: 'spec', label: 'speculative decoding', short: 'spec'},
  {key: 'prefix', label: 'prefix reuse', short: 'prefix'},
  {key: 'tp', label: 'multi-GPU serving', short: 'TP'},
];

const SYMBOL: Record<Cell, string> = {yes: '✓', no: '✗', unknown: '?', backend: '~'};

export type Verdict = {name: string; verdict: 'fits' | 'unverified' | 'out'; reason: string};

export function choose(hardware: Hardware, needs: Need[], avoidMaintenance: boolean): Verdict[] {
  return ENGINES.map((e) => {
    const cells: Cell[] = [e.hardware[hardware], ...needs.map((n) => e.needs[n])];
    if (avoidMaintenance && e.maintenance) {
      return {name: e.name, verdict: 'out', reason: 'maintenance mode'};
    }
    if (cells.includes('no')) {
      return {name: e.name, verdict: 'out', reason: 'a needed cell is documented as unsupported'};
    }
    if (cells.every((c) => c === 'yes')) {
      return {name: e.name, verdict: 'fits', reason: 'every needed cell is yes'};
    }
    return {name: e.name, verdict: 'unverified', reason: 'a needed cell is not established or depends on the backend'};
  });
}

const W = 640;
const H = 300;
const LABEL_W = 112;
const CELL_W = 52;
const TOP = 46;
const ROW_H = 30;

export default function ServingEngineChooserLab() {
  const dark = useDarkViz();
  const [hardware, setHardware] = useState<Hardware>('nvidia');
  const [needs, setNeeds] = useState<Need[]>(['openai', 'structured', 'multilora', 'spec', 'prefix', 'tp']);
  const [avoid, setAvoid] = useState(true);

  const verdicts = useMemo(() => choose(hardware, needs, avoid), [hardware, needs, avoid]);
  const fits = verdicts.filter((v) => v.verdict === 'fits');
  const unverified = verdicts.filter((v) => v.verdict === 'unverified');
  const out = verdicts.filter((v) => v.verdict === 'out');

  const colour = (c: Cell) =>
    c === 'yes'
      ? seriesColor(2, dark)
      : c === 'no'
        ? dark
          ? DIVERGING.dark.negative
          : DIVERGING.light.negative
        : c === 'backend'
          ? seriesColor(4, dark)
          : dark
            ? DIVERGING.dark.mid
            : DIVERGING.light.mid;

  const columns: {label: string; get: (e: Engine) => Cell; needed: boolean}[] = [
    {label: 'HW', get: (e) => e.hardware[hardware], needed: true},
    ...NEEDS.map((n) => ({label: n.short, get: (e: Engine) => e.needs[n.key], needed: needs.includes(n.key)})),
  ];

  const toggle = (key: Need) =>
    setNeeds((current) => (current.includes(key) ? current.filter((k) => k !== key) : [...current, key]));

  const verdictColour = (v: Verdict['verdict']) =>
    v === 'fits' ? seriesColor(2, dark) : v === 'unverified' ? seriesColor(4, dark) : dark ? DIVERGING.dark.negative : DIVERGING.light.negative;

  return (
    <VizPanel
      title="Serving engine chooser"
      hint="Pick the hardware and tick what you need. The grid is the feature matrix from each engine's documentation (opened 2026-10-02): an engine fits only when every needed cell is a yes, and a question mark means the pages opened did not establish the cell, not that it is missing. Defaults (NVIDIA, all six needs, maintenance mode excluded) give fits 3, unverified 3, out 1, as the chapter's code prints."
      legend={[
        {label: 'yes, documented', color: seriesColor(2, dark)},
        {label: 'no, documented', color: dark ? DIVERGING.dark.negative : DIVERGING.light.negative},
        {label: 'not established', color: dark ? DIVERGING.dark.mid : DIVERGING.light.mid},
        {label: 'depends on backend', color: seriesColor(4, dark)},
      ]}
      table={{
        columns: ['engine', 'verdict', 'reason'],
        rows: verdicts.map((v) => [v.name, v.verdict, v.reason]),
      }}
      controls={
        <>
          <label className={s.control}>
            hardware
            <select className={s.select} value={hardware} onChange={(e) => setHardware(e.target.value as Hardware)}>
              {HARDWARE.map((h) => (
                <option key={h.key} value={h.key}>
                  {h.label}
                </option>
              ))}
            </select>
          </label>
          {NEEDS.map((n) => (
            <label key={n.key} className={s.control}>
              <input type="checkbox" checked={needs.includes(n.key)} onChange={() => toggle(n.key)} />
              {n.label}
            </label>
          ))}
          <label className={s.control}>
            <input type="checkbox" checked={avoid} onChange={() => setAvoid(!avoid)} />
            leave out maintenance-mode projects
          </label>
          <span className={s.value} aria-live="polite">
            fits {fits.length} | unverified {unverified.length} | out {out.length}
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Feature matrix for seven serving engines. ${fits.length} fit: ${fits.map((v) => v.name).join(', ') || 'none'}.`}>
        {columns.map((c, i) => (
          <text
            key={c.label}
            className={s.tick}
            x={LABEL_W + i * CELL_W + CELL_W / 2}
            y={TOP - 12}
            textAnchor="middle"
            fontWeight={c.needed ? 700 : 400}>
            {c.label}
          </text>
        ))}
        <text className={s.tick} x={LABEL_W + columns.length * CELL_W + 14} y={TOP - 12} fontWeight={700}>
          verdict
        </text>
        {ENGINES.map((e, r) => {
          const y = TOP + r * ROW_H;
          const verdict = verdicts[r];
          return (
            <g key={e.name}>
              <text className={s.dataLabel} x={LABEL_W - 10} y={y + ROW_H / 2 + 4} textAnchor="end">
                {e.name}
              </text>
              {columns.map((c, i) => {
                const cell = c.get(e);
                return (
                  <g key={c.label}>
                    <rect
                      x={LABEL_W + i * CELL_W + 3}
                      y={y + 3}
                      width={CELL_W - 6}
                      height={ROW_H - 6}
                      rx={4}
                      fill={colour(cell)}
                      opacity={c.needed ? 0.9 : 0.3}
                      stroke={c.needed ? 'var(--text-strong)' : 'none'}
                      strokeWidth={1}
                    />
                    <text
                      x={LABEL_W + i * CELL_W + CELL_W / 2}
                      y={y + ROW_H / 2 + 5}
                      textAnchor="middle"
                      fontSize={14}
                      fontWeight={700}
                      fill={cell === 'backend' ? '#1a1a1a' : '#fff'}>
                      {SYMBOL[cell]}
                    </text>
                  </g>
                );
              })}
              <text
                className={s.dataLabel}
                x={LABEL_W + columns.length * CELL_W + 14}
                y={y + ROW_H / 2 + 4}
                fill={verdictColour(verdict.verdict)}>
                {verdict.verdict === 'out' ? `out: ${verdict.reason === 'maintenance mode' ? 'maintenance' : 'gap'}` : verdict.verdict}
              </text>
            </g>
          );
        })}
      </svg>
    </VizPanel>
  );
}
