import {useMemo, useState} from 'react';

import {breakEven, buildBuyMonthly, fmt} from './craftMath';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const PAD = {top: 20, right: 24, bottom: 40, left: 64};
const VOLUMES = [100_000, 300_000, 1_000_000, 3_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000, 200_000_000];
const LOG_MIN = Math.log10(100_000);
const LOG_MAX = Math.log10(200_000_000);
const TABLE_VOLUMES = [100_000, 1_000_000, 5_000_000, 20_000_000, 100_000_000];

export default function BuildBuyLab() {
  const dark = useDarkViz();
  const [volume, setVolume] = useState(5_000_000);
  const [apiOutPerM, setApiOutPerM] = useState(8);
  const [gpuPerHour, setGpuPerHour] = useState(2.5);
  const [platformFte, setPlatformFte] = useState(0.5);
  const [extraErrorPoints, setExtraErrorPoints] = useState(3);

  const params = useMemo(
    () => ({requests: volume, apiOutPerM, gpuPerHour, platformFte, extraErrorPoints}),
    [volume, apiOutPerM, gpuPerHour, platformFte, extraErrorPoints],
  );
  const here = buildBuyMonthly(params, volume);
  const even = useMemo(() => breakEven(params), [params]);

  const curve = useMemo(() => {
    const pts: {v: number; buy: number; host: number}[] = [];
    for (let i = 0; i <= 120; i += 1) {
      const v = 10 ** (LOG_MIN + ((LOG_MAX - LOG_MIN) * i) / 120);
      const m = buildBuyMonthly(params, v);
      pts.push({v, buy: m.buy, host: m.host});
    }
    return pts;
  }, [params]);

  const per1000 = (cost: number, v: number) => (cost / v) * 1000;
  const logs = curve.flatMap((p) => [Math.log10(per1000(p.buy, p.v)), Math.log10(per1000(p.host, p.v))]);
  const yLo = Math.floor(Math.min(...logs));
  const yHi = Math.ceil(Math.max(...logs));
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const x = (v: number) => PAD.left + ((Math.log10(v) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * innerW;
  const y = (c: number) => PAD.top + innerH - ((Math.log10(c) - yLo) / (yHi - yLo || 1)) * innerH;
  const path = (key: 'buy' | 'host') =>
    curve.map((p, i) => `${i ? 'L' : 'M'}${x(p.v).toFixed(1)},${y(per1000(p[key], p.v)).toFixed(1)}`).join(' ');

  const buyColor = seriesColor(0, dark);
  const hostColor = seriesColor(1, dark);
  const cheaper = here.buy <= here.host ? 'buying' : 'hosting';

  const rows = TABLE_VOLUMES.map((v) => {
    const m = buildBuyMonthly(params, v);
    return [fmt(v), m.replicas, fmt(m.buy), fmt(m.host), m.buy <= m.host ? 'buy' : 'host'];
  });

  const yTicks = Array.from({length: yHi - yLo + 1}, (_, i) => 10 ** (yLo + i));
  const xTicks = [100_000, 1_000_000, 10_000_000, 100_000_000];

  return (
    <VizPanel
      title="Build or buy: cost per 1,000 requests against volume"
      hint="Defaults reproduce the chapter: at 5,000,000 requests a month buying costs 57,000 and hosting 56,150, and the break-even is 4,645,833 requests. Set the extra error points to 0 and it drops to 2,064,815. Halve the API output price to 4 and hosting never catches up below 500 million."
      legend={[
        {label: 'buy (API)', color: buyColor},
        {label: 'host (open weights)', color: hostColor},
      ]}
      table={{columns: ['requests / month', 'replicas', 'buy', 'host', 'cheaper'], rows}}
      controls={
        <>
          <label className={s.control}>
            requests per month
            <select className={s.select} value={volume} onChange={(e) => setVolume(Number(e.target.value))}>
              {VOLUMES.map((v) => (
                <option key={v} value={v}>
                  {fmt(v)}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            API output price per million
            <input type="range" min={2} max={16} step={1} value={apiOutPerM} onChange={(e) => setApiOutPerM(Number(e.target.value))} />
            <span className={s.value}>{apiOutPerM}</span>
          </label>
          <label className={s.control}>
            GPU per hour
            <input type="range" min={1} max={6} step={0.25} value={gpuPerHour} onChange={(e) => setGpuPerHour(Number(e.target.value))} />
            <span className={s.value}>{gpuPerHour.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            platform engineers (FTE)
            <input type="range" min={0} max={2} step={0.25} value={platformFte} onChange={(e) => setPlatformFte(Number(e.target.value))} />
            <span className={s.value}>{platformFte.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            extra error points, open model
            <input type="range" min={0} max={10} step={1} value={extraErrorPoints} onChange={(e) => setExtraErrorPoints(Number(e.target.value))} />
            <span className={s.value}>{extraErrorPoints}</span>
          </label>
          <span className={s.value} aria-live="polite">
            at {fmt(volume)} requests: buy {fmt(here.buy)}, host {fmt(here.host)} on {here.replicas} replicas, {cheaper} is cheaper;{' '}
            {even === null ? 'no break-even below 500,000,000' : `break-even ${fmt(even)} requests`}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Monthly cost of buying and hosting against requests per month. ${cheaper} is cheaper at ${fmt(volume)} requests.`}>
        <line className={s.axis} x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH} />
        <line className={s.axis} x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + innerH} />
        {xTicks.map((t) => (
          <text key={t} className={s.tick} x={x(t)} y={H - 20} textAnchor="middle">
            {t >= 1_000_000 ? `${t / 1_000_000}M` : `${t / 1000}k`}
          </text>
        ))}
        {yTicks.map((t) => (
          <text key={t} className={s.tick} x={PAD.left - 6} y={y(t) + 3} textAnchor="end">
            {fmt(t)}
          </text>
        ))}
        <text className={s.axisLabel} x={PAD.left + innerW / 2} y={H - 4} textAnchor="middle">
          requests per month (log scale); vertical axis: cost per 1,000 requests (log scale)
        </text>
        <path d={path('buy')} fill="none" stroke={buyColor} strokeWidth={2.5} />
        <path d={path('host')} fill="none" stroke={hostColor} strokeWidth={2.5} strokeDasharray="6 3" />
        {even !== null && (
          <g>
            <line x1={x(even)} y1={PAD.top} x2={x(even)} y2={PAD.top + innerH} stroke="var(--text-strong)" strokeWidth={1.5} strokeDasharray="3 3" />
            <text className={s.dataLabel} x={x(even)} y={PAD.top + 10} textAnchor="middle">
              break-even {fmt(even)}
            </text>
          </g>
        )}
        <circle cx={x(volume)} cy={y(per1000(here.buy, volume))} r={5} fill={buyColor} stroke="var(--surface-raised)" strokeWidth={1.5} />
        <circle cx={x(volume)} cy={y(per1000(here.host, volume))} r={5} fill={hostColor} stroke="var(--surface-raised)" strokeWidth={1.5} />
      </svg>
    </VizPanel>
  );
}
