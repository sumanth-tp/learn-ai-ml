import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const SERVERS = [1, 2, 4, 8, 16];
const MODELS = [25, 100, 400, 1600];
const LINKS = [1, 10, 100];

export function psLoad(n: number, p: number, sizeMb: number) {
  return (n * sizeMb) / p;
}

export function psTime(n: number, p: number, sizeMb: number, gbps: number) {
  return (2 * Math.max(sizeMb, psLoad(n, p, sizeMb))) / (gbps * 1000);
}

export function ringSent(n: number, sizeMb: number) {
  return ((2 * (n - 1)) / n) * sizeMb;
}

export function ringTime(n: number, sizeMb: number, gbps: number) {
  return ringSent(n, sizeMb) / (gbps * 1000);
}

export default function ParameterServerLab() {
  const dark = useDarkViz();
  const [n, setN] = useState(4);
  const [p, setP] = useState(1);
  const [size, setSize] = useState(100);
  const [gbps, setGbps] = useState(10);

  const server = psLoad(n, p, size);
  const worker = 2 * size;
  const ring = ringSent(n, size);
  const bars = [
    {label: 'one server shard receives', value: server, color: seriesColor(1, dark)},
    {label: 'one worker pushes and pulls', value: worker, color: seriesColor(0, dark)},
    {label: 'one ring worker sends', value: ring, color: seriesColor(2, dark)},
  ];
  const max = Math.max(...bars.map((b) => b.value), 1);

  const left = 190;
  const plotW = 400;
  const rowH = 52;

  const rows = [4, 16, 64].flatMap((nn) =>
    [1, 4, 16].map((pp) => [
      nn,
      pp,
      psLoad(nn, pp, size).toFixed(0),
      psTime(nn, pp, size, gbps).toFixed(3),
      ringTime(nn, size, gbps).toFixed(5),
    ]),
  );

  return (
    <VizPanel
      title="Parameter server against ring all-reduce"
      hint="Each step every worker sends its gradient to the servers and pulls fresh parameters. One server must take in all N copies; more server shards split the load. The ring needs no server at all."
      legend={bars.map((b) => ({label: b.label, color: b.color}))}
      table={{columns: ['workers', 'servers', 'shard receives MB', 'PS time s', 'ring time s'], rows}}
      controls={
        <>
          <label className={s.control}>
            workers
            <input type="range" min={1} max={64} step={1} value={n} onChange={(e) => setN(Number(e.target.value))} />
            <span className={s.value}>{n}</span>
          </label>
          <label className={s.control}>
            server shards
            <select className={s.select} value={p} onChange={(e) => setP(Number(e.target.value))}>
              {SERVERS.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            model MB
            <select className={s.select} value={size} onChange={(e) => setSize(Number(e.target.value))}>
              {MODELS.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            link GB/s
            <select className={s.select} value={gbps} onChange={(e) => setGbps(Number(e.target.value))}>
              {LINKS.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <span className={s.value} aria-live="polite">
            parameter server {psTime(n, p, size, gbps).toFixed(3)} s, ring {ringTime(n, size, gbps).toFixed(3)} s
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`With ${n} workers, ${p} server shards and a ${size} megabyte model, one server shard receives ${server.toFixed(0)} megabytes per step and a ring worker sends ${ring.toFixed(0)}`}>
        <text className={s.axisLabel} x={W / 2} y={20} textAnchor="middle">
          megabytes handled per step, by one node
        </text>
        {bars.map((b, i) => {
          const y = 44 + i * rowH;
          const w = (b.value / max) * plotW;
          return (
            <g key={b.label}>
              <text className={s.tick} x={left - 8} y={y + 22} textAnchor="end">
                {b.label}
              </text>
              <rect x={left} y={y} width={Math.max(w, 1)} height={30} fill={b.color} opacity={0.9} />
              <text className={s.dataLabel} x={left + Math.min(w, plotW - 70) + 8} y={y + 20}>
                {b.value.toFixed(0)} MB
              </text>
            </g>
          );
        })}
        <text className={s.axisLabel} x={W / 2} y={H - 40} textAnchor="middle">
          parameter server: 2 x max(model, workers x model / shards) / link
        </text>
        <text className={s.axisLabel} x={W / 2} y={H - 18} textAnchor="middle">
          ring: 2(N-1)/N x model / link
        </text>
      </svg>
    </VizPanel>
  );
}
