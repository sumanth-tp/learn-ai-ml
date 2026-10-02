import {useState} from 'react';

import {KV_MODELS, kvElementsPerToken, kvSequencesThatFit} from './inferenceMath';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 330;
const CONTEXTS = [640, 2048, 4096, 8192, 32768, 131072];
const RESERVES = [2048, 4096, 8192, 32768];
const BLOCKS = [1, 8, 16, 32, 128];
const DTYPES: {label: string; bytes: number}[] = [
  {label: 'bf16 (2 bytes)', bytes: 2},
  {label: '8-bit (1 byte)', bytes: 1},
  {label: '4-bit (0.5 bytes)', bytes: 0.5},
];

const fmt = (v: number) => Math.round(v).toLocaleString('en-US');

export default function KvCacheLab() {
  const dark = useDarkViz();
  const [model, setModel] = useState(3);
  const [dtype, setDtype] = useState(0);
  const [context, setContext] = useState(640);
  const [budget, setBudget] = useState(40);
  const [reserve, setReserve] = useState(4096);
  const [block, setBlock] = useState(16);

  const dtypeBytes = DTYPES[dtype].bytes;
  const perToken = (i: number) => kvElementsPerToken(KV_MODELS[i]) * dtypeBytes;
  const bytes = perToken(model);
  const contiguous = kvSequencesThatFit(budget, bytes, reserve, 1);
  const paged = kvSequencesThatFit(budget, bytes, context, block);
  const roundedSlots = Math.ceil(context / block) * block;
  const maxKib = Math.max(...KV_MODELS.map((_, i) => perToken(i) / 1024));
  const maxSeq = Math.max(contiguous, paged, 1);

  const used = seriesColor(0, dark);
  const waste = seriesColor(1, dark);
  const muted = dark ? '#848c99' : '#9aa0a6';

  const barArea = {x: 150, w: 150};
  const rowH = 28;
  const right = {x: 360, w: 250};
  const stripW = right.w;
  const reserveScale = stripW / Math.max(reserve, context);

  const rows = KV_MODELS.map((m, i) => [
    m.name,
    m.kind,
    fmt(kvElementsPerToken(m)),
    (perToken(i) / 1024).toFixed(1),
    ((perToken(i) * context) / 1e9).toFixed(2),
  ]);

  return (
    <VizPanel
      title="KV cache size and how many sequences fit"
      hint="Pick a model and watch the cost per token move with the attention design: grouped-query attention, multi-query attention and latent compression all shrink it. Then compare reserving the maximum length for every sequence with allocating small blocks on demand. Defaults match the chapter: Llama 3.1 8B at 128 KiB per token, 476 sequences paged against 74 with a 4,096-token reservation."
      legend={[
        {label: 'KiB per token (selected model)', color: used},
        {label: 'KiB per token (other models)', color: muted},
        {label: 'reserved but unused', color: waste},
      ]}
      table={{columns: ['model', 'kind', 'elements per token', 'KiB per token', `GB at ${context} tokens`], rows}}
      controls={
        <>
          <label className={s.control}>
            model
            <select className={s.select} value={model} onChange={(e) => setModel(Number(e.target.value))}>
              {KV_MODELS.map((m, i) => (
                <option key={m.name} value={i}>{m.name}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            cache dtype
            <select className={s.select} value={dtype} onChange={(e) => setDtype(Number(e.target.value))}>
              {DTYPES.map((d, i) => (
                <option key={d.label} value={i}>{d.label}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            context per sequence
            <select className={s.select} value={context} onChange={(e) => setContext(Number(e.target.value))}>
              {CONTEXTS.map((c) => (
                <option key={c} value={c}>{c.toLocaleString('en-US')}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            cache budget GB
            <input type="range" min={1} max={80} step={1} value={budget}
                   onChange={(e) => setBudget(Number(e.target.value))} />
            <span className={s.value}>{budget}</span>
          </label>
          <label className={s.control}>
            reserved length
            <select className={s.select} value={reserve} onChange={(e) => setReserve(Number(e.target.value))}>
              {RESERVES.map((r) => (
                <option key={r} value={r}>{r.toLocaleString('en-US')}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            block size
            <select className={s.select} value={block} onChange={(e) => setBlock(Number(e.target.value))}>
              {BLOCKS.map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`${KV_MODELS[model].name} needs ${(bytes / 1024).toFixed(1)} KiB of cache per token. With a ${budget} gigabyte budget, ${fmt(paged)} sequences of ${context} tokens fit with ${block}-token blocks and ${fmt(contiguous)} fit with a ${reserve}-token reservation.`}>
        <text className={s.axisLabel} x={8} y={16}>KiB of cache per token</text>
        {KV_MODELS.map((m, i) => {
          const kib = perToken(i) / 1024;
          const y = 28 + i * rowH;
          return (
            <g key={m.name}>
              <text className={s.tick} x={barArea.x - 6} y={y + 15} textAnchor="end"
                    fontWeight={i === model ? 700 : 400}>{m.name}</text>
              <rect x={barArea.x} y={y + 3} width={Math.max(2, (kib / maxKib) * barArea.w)} height={18} rx={3}
                    fill={i === model ? used : muted} opacity={i === model ? 1 : 0.55} />
              <text className={s.dataLabel} x={barArea.x + Math.max(2, (kib / maxKib) * barArea.w) + 5} y={y + 16}>
                {kib.toFixed(1)}
              </text>
            </g>
          );
        })}
        <text className={s.axisLabel} x={right.x} y={16}>sequences that fit in {budget} GB</text>
        {[
          {label: `reserve ${reserve.toLocaleString('en-US')}`, value: contiguous, color: waste},
          {label: `paged, block ${block}`, value: paged, color: used},
        ].map((bar, k) => {
          const y = 28 + k * 56;
          const width = Math.max(2, (bar.value / maxSeq) * (right.w - 40));
          return (
            <g key={bar.label}>
              <text className={s.tick} x={right.x} y={y + 10}>{bar.label}</text>
              <rect x={right.x} y={y + 16} width={width} height={22} rx={3} fill={bar.color} />
              <text className={s.dataLabel} x={right.x + width + 6} y={y + 32}>{fmt(bar.value)}</text>
            </g>
          );
        })}
        <text className={s.axisLabel} x={right.x} y={162}>one sequence of {context.toLocaleString('en-US')} tokens</text>
        <text className={s.tick} x={right.x} y={182}>contiguous reservation</text>
        <rect x={right.x} y={188} width={stripW} height={16} fill="none" stroke="var(--border-strong)" />
        <rect x={right.x} y={188} width={reserveScale * Math.min(context, reserve)} height={16} fill={used} />
        <rect x={right.x + reserveScale * Math.min(context, reserve)} y={188}
              width={Math.max(0, reserveScale * (reserve - context))} height={16} fill={waste} opacity={0.75} />
        <text className={s.tick} x={right.x} y={230}>paged blocks</text>
        <rect x={right.x} y={236} width={stripW} height={16} fill="none" stroke="var(--border-strong)" />
        <rect x={right.x} y={236} width={reserveScale * context} height={16} fill={used} />
        <rect x={right.x + reserveScale * context} y={236} width={reserveScale * (roundedSlots - context)} height={16}
              fill={waste} opacity={0.75} />
        <text className={s.dataLabel} x={right.x} y={278}>
          contiguous waste {(100 * (1 - Math.min(context, reserve) / reserve)).toFixed(1)}%
        </text>
        <text className={s.dataLabel} x={right.x} y={296}>
          paged waste {(100 * (1 - context / roundedSlots)).toFixed(1)}% ({roundedSlots - context} slots)
        </text>
      </svg>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.4rem'}}>
        <span aria-live="polite">
          {KV_MODELS[model].name}: {fmt(bytes)} bytes per token, {((bytes * context) / 1e9).toFixed(3)} GB for one
          sequence of {context.toLocaleString('en-US')} tokens.
        </span>
      </div>
    </VizPanel>
  );
}
