import type {KeyboardEvent} from 'react';
import {useId, useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Point = {name: string; x: number; y: number; label: 1 | -1};
type Neighbour = Point & {d: number; weight: number; chosen: boolean};

const W = 640;
const H = 300;
const PAD = {top: 14, right: 16, bottom: 34, left: 40};

const LECTURE: Point[] = [
  {name: 'A', x: 1, y: 1, label: 1},
  {name: 'B', x: 2, y: 2, label: 1},
  {name: 'C', x: 3, y: 3, label: -1},
  {name: 'D', x: 5, y: 1, label: -1},
  {name: 'E', x: 1, y: 4, label: 1},
];

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildScales() {
  const rand = mulberry32(11);
  const raw: {income: number; age: number; label: 1 | -1}[] = [];
  for (let i = 0; i < 40; i += 1) {
    const income = 20000 + rand() * 100000;
    const age = 20 + rand() * 50;
    let label: 1 | -1 = age >= 45 ? 1 : -1;
    if (i % 10 === 3) label = label === 1 ? -1 : 1;
    raw.push({income, age, label});
  }
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
  const sd = (v: number[]) => {
    const m = mean(v);
    return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length);
  };
  const mx = mean(raw.map((r) => r.income));
  const my = mean(raw.map((r) => r.age));
  const sx = sd(raw.map((r) => r.income));
  const sy = sd(raw.map((r) => r.age));
  const points: Point[] = raw.map((r, i) => ({
    name: `p${i + 1}`,
    x: (r.income - mx) / sx,
    y: (r.age - my) / sy,
    label: r.label,
  }));
  return {points, sx, sy};
}

const SCALES = buildScales();

export function rank(
  points: Point[],
  qx: number,
  qy: number,
  k: number,
  fx: number,
  fy: number,
  weighting: 'uniform' | 'inverse-square',
  skip = -1,
): Neighbour[] {
  const all = points.map((p, i) => ({
    ...p,
    d: Math.hypot((p.x - qx) * fx, (p.y - qy) * fy),
    weight: 0,
    chosen: false,
    i,
  }));
  const order = all.filter((p) => p.i !== skip).sort((a, b) => a.d - b.d || a.i - b.i);
  order.forEach((p, idx) => {
    p.chosen = idx < k;
    p.weight = p.chosen ? (weighting === 'uniform' ? 1 : 1 / Math.max(p.d * p.d, 1e-12)) : 0;
  });
  return all;
}

export function tally(neighbours: Neighbour[]) {
  let plus = 0;
  let minus = 0;
  neighbours.forEach((n) => {
    if (!n.chosen) return;
    if (n.label === 1) plus += n.weight;
    else minus += n.weight;
  });
  return {plus, minus};
}

export function distanceRatio(dimensions: number, points = 200, queries = 30) {
  const rand = mulberry32(7);
  const data: number[][] = [];
  for (let i = 0; i < points; i += 1) data.push(Array.from({length: dimensions}, rand));
  const probes: number[][] = [];
  for (let i = 0; i < queries; i += 1) probes.push(Array.from({length: dimensions}, rand));
  let total = 0;
  let first: number[] = [];
  probes.forEach((q, qi) => {
    const dists = data.map((p) => Math.sqrt(p.reduce((a, v, j) => a + (v - q[j]) ** 2, 0)));
    total += Math.min(...dists) / Math.max(...dists);
    if (qi === 0) first = dists;
  });
  return {ratio: total / queries, first};
}

export default function KnnLab() {
  const dark = useDarkViz();
  const clipId = useId();
  const [view, setView] = useState<'neighbours' | 'dimensions'>('neighbours');
  const [data, setData] = useState<'lecture' | 'scales'>('lecture');
  const [qx, setQx] = useState(2);
  const [qy, setQy] = useState(3);
  const [k, setK] = useState(3);
  const [weighting, setWeighting] = useState<'uniform' | 'inverse-square'>('uniform');
  const [standardise, setStandardise] = useState(false);
  const [dimensions, setDimensions] = useState(50);

  const plusColor = seriesColor(0, dark);
  const minusColor = seriesColor(1, dark);

  const lecture = data === 'lecture';
  const points = lecture ? LECTURE : SCALES.points;
  const lo = lecture ? 0 : -2.4;
  const hi = lecture ? 6 : 2.4;
  const fx = lecture || standardise ? 1 : SCALES.sx;
  const fy = lecture || standardise ? 1 : SCALES.sy;
  const maxK = lecture ? 5 : 15;
  const kk = Math.min(k, maxK);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const unit = innerH / (hi - lo);
  const left = PAD.left + (innerW - innerH) / 2;
  const px = (v: number) => left + (v - lo) * unit;
  const py = (v: number) => PAD.top + innerH - (v - lo) * unit;

  const neighbours = useMemo(
    () => rank(points, qx, qy, kk, fx, fy, weighting),
    [points, qx, qy, kk, fx, fy, weighting],
  );
  const {plus, minus} = tally(neighbours);
  const total = plus + minus;
  const predicted = plus === minus ? 'tie' : plus > minus ? '+' : '-';
  const chosen = neighbours.filter((n) => n.chosen);
  const radius = chosen.length ? Math.max(...chosen.map((n) => n.d)) : 0;

  const leaveOneOut = useMemo(() => {
    if (lecture) return null;
    let right = 0;
    points.forEach((p, i) => {
      const t = tally(rank(points, p.x, p.y, kk, fx, fy, weighting, i));
      const guess = t.plus === t.minus ? 0 : t.plus > t.minus ? 1 : -1;
      if (guess === p.label) right += 1;
    });
    return right / points.length;
  }, [lecture, points, kk, fx, fy, weighting]);

  const ratio = useMemo(() => distanceRatio(dimensions), [dimensions]);
  const histogram = useMemo(() => {
    const bins = 24;
    const counts = new Array(bins).fill(0);
    const top = Math.max(...ratio.first);
    ratio.first.forEach((d) => {
      counts[Math.min(bins - 1, Math.floor((d / top) * bins))] += 1;
    });
    return {counts, top};
  }, [ratio]);

  const ratioTable = useMemo(
    () => [2, 10, 50, 100, 200, 300].map((d) => [d, distanceRatio(d).ratio.toFixed(3)]),
    [],
  );

  const move = (dx: number, dy: number) => {
    const step = lecture ? 0.1 : 0.05;
    setQx((v) => Math.min(hi, Math.max(lo, Math.round((v + dx * step) * 100) / 100)));
    setQy((v) => Math.min(hi, Math.max(lo, Math.round((v + dy * step) * 100) / 100)));
  };

  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    const keys: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, 1],
      ArrowDown: [0, -1],
    };
    const delta = keys[e.key];
    if (!delta) return;
    e.preventDefault();
    move(delta[0], delta[1]);
  };

  const fmtDist = (d: number) => (d >= 1000 ? d.toFixed(0) : d.toFixed(2));
  const table =
    view === 'neighbours'
      ? {
          columns: ['point', 'class', 'distance', 'in the k nearest', 'weight'],
          rows: [...neighbours]
            .sort((a, b) => a.d - b.d)
            .map((n) => [
              n.name,
              n.label === 1 ? '+' : '-',
              fmtDist(n.d),
              n.chosen ? 'yes' : 'no',
              n.chosen ? n.weight.toFixed(2) : '0',
            ]),
        }
      : {columns: ['dimensions', 'mean nearest / farthest'], rows: ratioTable};

  const controls =
    view === 'neighbours' ? (
      <>
        <label className={s.control}>
          view
          <select className={s.select} value={view} onChange={(e) => setView(e.target.value as typeof view)}>
            <option value="neighbours">neighbours vote</option>
            <option value="dimensions">high dimensions</option>
          </select>
        </label>
        <label className={s.control}>
          data
          <select
            className={s.select}
            value={data}
            onChange={(e) => {
              const next = e.target.value as typeof data;
              setData(next);
              setQx(next === 'lecture' ? 2 : 0);
              setQy(next === 'lecture' ? 3 : 0);
              setK(next === 'lecture' ? 3 : 5);
            }}>
            <option value="lecture">the lecture's five points</option>
            <option value="scales">two features on different scales</option>
          </select>
        </label>
        <label className={s.control}>
          query x
          <input type="range" min={lo} max={hi} step={lecture ? 0.1 : 0.05} value={qx}
                 onChange={(e) => setQx(Number(e.target.value))} />
          <span className={s.value}>{qx.toFixed(2)}</span>
        </label>
        <label className={s.control}>
          query y
          <input type="range" min={lo} max={hi} step={lecture ? 0.1 : 0.05} value={qy}
                 onChange={(e) => setQy(Number(e.target.value))} />
          <span className={s.value}>{qy.toFixed(2)}</span>
        </label>
        <label className={s.control}>
          k
          <input type="range" min={1} max={maxK} step={1} value={kk}
                 onChange={(e) => setK(Number(e.target.value))} />
          <span className={s.value}>{kk}</span>
        </label>
        <label className={s.control}>
          votes
          <select className={s.select} value={weighting}
                  onChange={(e) => setWeighting(e.target.value as typeof weighting)}>
            <option value="uniform">one each</option>
            <option value="inverse-square">weighted 1/d²</option>
          </select>
        </label>
        {!lecture && (
          <label className={s.control}>
            <input type="checkbox" checked={standardise} onChange={(e) => setStandardise(e.target.checked)} />
            standardise features
          </label>
        )}
      </>
    ) : (
      <>
        <label className={s.control}>
          view
          <select className={s.select} value={view} onChange={(e) => setView(e.target.value as typeof view)}>
            <option value="neighbours">neighbours vote</option>
            <option value="dimensions">high dimensions</option>
          </select>
        </label>
        <label className={s.control}>
          dimensions
          <input type="range" min={2} max={300} step={1} value={dimensions}
                 onChange={(e) => setDimensions(Number(e.target.value))} />
          <span className={s.value}>{dimensions}</span>
        </label>
      </>
    );

  const bins = histogram.counts.length;
  const barMax = Math.max(...histogram.counts, 1);
  const hx = (frac: number) => PAD.left + frac * innerW;

  return (
    <VizPanel
      title={view === 'neighbours' ? 'k-NN playground' : 'Why "nearest" fades in high dimensions'}
      hint={
        view === 'neighbours'
          ? lecture
            ? 'Default: query (2, 3), k = 3. B and C sit at distance 1.00 and E at 1.41, so the vote is + 2 to - 1. Switch the votes to weighted 1/d² and the totals become + 1.50 against - 1.00. Click the plot and use the arrow keys to move the query.'
            : 'The class depends only on the second feature (age-like, small range); the first feature (income-like, huge range) is noise. Untick standardise and the huge range decides who is "nearest", so the leave-one-out accuracy drops.'
          : 'Each draw is 200 uniform points and 30 random queries from a fixed seed. As the dimension grows the nearest and farthest points end up almost the same distance away, so the histogram collapses to a spike. The chapter code uses numpy draws, so its figures differ somewhat from these; the climb towards 1 is the same.'
      }
      legend={
        view === 'neighbours'
          ? [
              {label: '+ class (circle)', color: plusColor},
              {label: '- class (square)', color: minusColor},
              {label: 'query (diamond)', color: dark ? '#dfe3e9' : '#23262b'},
            ]
          : [{label: 'distance from one query to 200 points', color: plusColor}]
      }
      table={table}
      controls={controls}>
      {view === 'neighbours' ? (
        <>
          <svg
            className={s.svg}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            tabIndex={0}
            onKeyDown={onKey}
            aria-label="k nearest neighbours of the query point; arrow keys move the query">
            {[0, 1, 2, 3, 4, 5, 6].map((i) => {
              const v = lo + ((hi - lo) * i) / 6;
              return (
                <g key={i}>
                  <line className={s.grid} x1={px(v)} y1={PAD.top} x2={px(v)} y2={PAD.top + innerH} />
                  <line className={s.grid} x1={px(lo)} y1={py(v)} x2={px(hi)} y2={py(v)} />
                  <text className={s.tick} x={px(v)} y={H - 18} textAnchor="middle">{v.toFixed(lecture ? 0 : 1)}</text>
                  <text className={s.tick} x={PAD.left - 6} y={py(v) + 3} textAnchor="end">{v.toFixed(lecture ? 0 : 1)}</text>
                </g>
              );
            })}
            <text className={s.axisLabel} x={W / 2} y={H - 3} textAnchor="middle">
              {lecture ? 'x' : 'income (standardised units)'}
            </text>
            <defs>
              <clipPath id={clipId}>
                <rect x={px(lo)} y={PAD.top} width={px(hi) - px(lo)} height={innerH} />
              </clipPath>
            </defs>
            {radius > 0 && (
              <ellipse
                clipPath={`url(#${clipId})`}
                cx={px(qx)}
                cy={py(qy)}
                rx={(radius / fx) * unit}
                ry={(radius / fy) * unit}
                fill="none"
                stroke="var(--text-faint)"
                strokeDasharray="4 4"
              />
            )}
            {chosen.map((n) => (
              <line key={n.name} x1={px(qx)} y1={py(qy)} x2={px(n.x)} y2={py(n.y)}
                    stroke="var(--text-faint)" strokeWidth={1.2} />
            ))}
            {neighbours.map((n) => {
              const color = n.label === 1 ? plusColor : minusColor;
              const cx = px(n.x);
              const cy = py(n.y);
              return (
                <g key={n.name} opacity={n.chosen ? 1 : 0.55}>
                  {n.label === 1 ? (
                    <circle cx={cx} cy={cy} r={lecture ? 7 : 5} fill={color}
                            stroke={n.chosen ? 'var(--text-strong)' : 'none'} strokeWidth={2} />
                  ) : (
                    <rect x={cx - (lecture ? 7 : 5)} y={cy - (lecture ? 7 : 5)} width={lecture ? 14 : 10}
                          height={lecture ? 14 : 10} fill={color}
                          stroke={n.chosen ? 'var(--text-strong)' : 'none'} strokeWidth={2} />
                  )}
                  {lecture && (
                    <text className={s.dataLabel} x={cx + 11} y={cy - 9}>{n.name}</text>
                  )}
                </g>
              );
            })}
            <path
              d={`M${px(qx)},${py(qy) - 9} L${px(qx) + 9},${py(qy)} L${px(qx)},${py(qy) + 9} L${px(qx) - 9},${py(qy)} Z`}
              fill="var(--text-strong)"
            />
          </svg>
          <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem'}}>
            <span>
              {weighting === 'uniform' ? 'votes' : 'weights'} among the {kk} nearest: + {plus.toFixed(2)}, - {minus.toFixed(2)}
              {total > 0 && ` (share of + ${((plus / total) * 100).toFixed(0)}%)`} → predicts{' '}
              <strong>{predicted}</strong>
              {leaveOneOut !== null && ` · leave-one-out accuracy ${(leaveOneOut * 100).toFixed(0)}%`}
              {!lecture && ` · metric: ${standardise ? 'standardised' : 'raw units'}`}
            </span>
          </div>
        </>
      ) : (
        <>
          <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
               aria-label="Histogram of distances from one query to 200 random points">
            <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={PAD.left + innerW} y2={PAD.top + innerH} />
            {histogram.counts.map((c, i) => {
              const bw = innerW / bins;
              const bh = (c / barMax) * innerH;
              return (
                <rect key={i} x={PAD.left + i * bw + 1} y={PAD.top + innerH - bh} width={bw - 2} height={bh}
                      fill={plusColor} opacity={0.85} />
              );
            })}
            {[0, 0.25, 0.5, 0.75, 1].map((f) => (
              <text key={f} className={s.tick} x={hx(f)} y={H - 18} textAnchor="middle">
                {(f * histogram.top).toFixed(2)}
              </text>
            ))}
            <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 3} textAnchor="middle">
              distance from the query
            </text>
            <line x1={hx(Math.min(...ratio.first) / histogram.top)} y1={PAD.top}
                  x2={hx(Math.min(...ratio.first) / histogram.top)} y2={PAD.top + innerH}
                  stroke="var(--text-strong)" strokeDasharray="4 3" />
            <text className={s.dataLabel} x={hx(Math.min(...ratio.first) / histogram.top) + 4} y={PAD.top + 10}>
              nearest
            </text>
          </svg>
          <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem'}}>
            <span>
              {dimensions} dimensions: mean nearest / farthest distance = <strong>{ratio.ratio.toFixed(3)}</strong>
              {' '}(1.000 would mean every point is equally far)
            </span>
          </div>
        </>
      )}
    </VizPanel>
  );
}
