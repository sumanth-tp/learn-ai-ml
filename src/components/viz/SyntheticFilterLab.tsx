import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 330;
const TOTAL = 600;

export const JUDGE = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0];
export const ROUGE = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0];

export const GRID: number[][][] = [[[130,457,0,13,0.676,0.183,0.0],[130,441,0,29,0.412,0.232,0.0],[130,432,0,38,0.35,0.229,0.053],[130,419,0,51,0.348,0.212,0.0],[130,400,0,70,0.269,0.224,0.029],[130,375,0,95,0.228,0.215,0.063],[130,340,0,130,0.182,0.217,0.092],[130,303,0,167,0.161,0.218,0.174],[130,260,0,210,0.133,0.213,0.29],[130,173,0,297,0.109,0.209,0.428],[130,117,0,353,0.098,0.206,0.513],[130,84,0,386,0.09,0.206,0.57],[130,80,0,390,0.089,0.207,0.574]],[[130,457,1,12,0.685,0.191,0.0],[130,440,2,28,0.41,0.243,0.0],[130,431,3,36,0.355,0.239,0.056],[130,413,5,52,0.348,0.215,0.0],[130,393,7,70,0.272,0.224,0.029],[130,368,8,94,0.229,0.217,0.064],[130,333,10,127,0.183,0.22,0.079],[130,298,14,158,0.163,0.223,0.165],[130,253,15,202,0.132,0.217,0.287],[130,169,17,284,0.109,0.214,0.423],[130,115,18,337,0.097,0.21,0.504],[130,82,18,370,0.09,0.21,0.57],[130,78,18,374,0.088,0.211,0.575]],[[130,457,1,12,0.685,0.191,0.0],[130,440,2,28,0.41,0.243,0.0],[130,430,4,36,0.357,0.239,0.056],[130,411,8,51,0.349,0.213,0.0],[130,391,10,69,0.274,0.226,0.029],[130,365,12,93,0.23,0.218,0.065],[130,330,16,124,0.184,0.221,0.081],[130,295,21,154,0.163,0.224,0.169],[130,244,36,190,0.134,0.222,0.279],[130,159,41,270,0.11,0.216,0.404],[130,107,42,321,0.098,0.213,0.486],[130,74,42,354,0.09,0.212,0.554],[130,70,42,358,0.089,0.214,0.559]],[[130,457,2,11,0.69,0.196,0.0],[130,440,5,25,0.417,0.25,0.0],[130,429,7,34,0.35,0.246,0.059],[130,405,17,48,0.346,0.225,0.0],[130,381,24,65,0.273,0.238,0.031],[130,351,31,88,0.232,0.224,0.068],[130,318,38,114,0.186,0.229,0.105],[130,279,51,140,0.164,0.232,0.186],[130,225,76,169,0.136,0.231,0.284],[130,145,93,232,0.112,0.227,0.392],[130,95,105,270,0.101,0.224,0.456],[130,65,110,295,0.093,0.224,0.515],[130,61,110,299,0.091,0.226,0.522]],[[130,457,3,10,0.695,0.202,0.0],[130,437,11,22,0.417,0.27,0.0],[130,422,16,32,0.346,0.25,0.062],[130,399,30,41,0.354,0.24,0.0],[130,363,49,58,0.274,0.256,0.0],[130,312,79,79,0.224,0.25,0.076],[130,281,88,101,0.187,0.249,0.099],[130,252,99,119,0.165,0.254,0.202],[130,191,136,143,0.138,0.253,0.287],[130,120,167,183,0.116,0.248,0.421],[130,80,184,206,0.106,0.246,0.481],[130,53,190,227,0.098,0.247,0.537],[130,49,190,231,0.096,0.249,0.545]],[[130,457,3,10,0.695,0.202,0.0],[130,434,15,21,0.41,0.275,0.0],[130,421,20,29,0.346,0.259,0.069],[130,397,36,37,0.359,0.246,0.0],[130,353,65,52,0.277,0.263,0.0],[130,293,103,74,0.229,0.255,0.081],[130,254,120,96,0.192,0.253,0.104],[130,232,126,112,0.169,0.259,0.214],[130,171,166,133,0.14,0.26,0.293],[130,112,196,162,0.118,0.262,0.451],[130,75,214,181,0.107,0.26,0.525],[130,49,223,198,0.099,0.262,0.581],[130,45,223,202,0.097,0.264,0.589]],[[130,457,4,9,0.709,0.222,0.0],[130,434,17,19,0.418,0.299,0.0],[130,419,27,24,0.378,0.279,0.0],[130,378,60,32,0.354,0.269,0.0],[130,328,98,44,0.288,0.282,0.045],[130,261,150,59,0.245,0.271,0.136],[130,218,176,76,0.201,0.278,0.118],[130,186,195,89,0.179,0.283,0.258],[130,130,239,101,0.153,0.287,0.356],[130,86,268,116,0.131,0.294,0.448],[130,63,277,130,0.117,0.296,0.546],[130,42,285,143,0.108,0.299,0.629],[130,38,285,147,0.105,0.302,0.639]],[[130,457,4,9,0.709,0.222,0.0],[130,434,18,18,0.429,0.292,0.0],[130,419,29,22,0.389,0.273,0.0],[130,372,68,30,0.372,0.26,0.0],[130,320,109,41,0.298,0.281,0.049],[130,258,157,55,0.252,0.27,0.127],[130,214,182,74,0.201,0.279,0.135],[130,183,203,84,0.186,0.282,0.214],[130,126,247,97,0.156,0.286,0.361],[130,83,277,110,0.135,0.292,0.418],[130,62,285,123,0.121,0.294,0.528],[130,41,297,132,0.114,0.296,0.598],[130,37,298,135,0.112,0.298,0.607]],[[130,453,9,8,0.705,0.215,0.0],[130,425,27,18,0.444,0.32,0.0],[130,408,43,19,0.404,0.28,0.0],[130,353,92,25,0.366,0.281,0.0],[130,309,130,31,0.333,0.294,0.0],[130,248,182,40,0.299,0.274,0.125],[130,203,212,55,0.234,0.294,0.2],[130,172,235,63,0.215,0.291,0.254],[130,116,285,69,0.195,0.294,0.304],[130,74,316,80,0.166,0.301,0.338],[130,53,330,87,0.153,0.306,0.448],[130,38,339,93,0.143,0.309,0.516],[130,34,341,95,0.141,0.311,0.526]]];

export default function SyntheticFilterLab() {
  const dark = useDarkViz();
  const [judge, setJudge] = useState(7.0);
  const [rouge, setRouge] = useState(0.7);

  const ji = JUDGE.indexOf(judge);
  const ri = ROUGE.findIndex((r) => Math.abs(r - rouge) < 1e-9);
  const [ruleDrop, dup, judgeDrop, kept, distinct2, meanCos, near] = GRID[ji][ri];
  const row = GRID[ji];

  const colors = {
    rule: dark ? DIVERGING.dark.mid : DIVERGING.light.mid,
    dup: seriesColor(1, dark),
    judge: seriesColor(3, dark),
    kept: seriesColor(2, dark),
  };

  const BAR = {x: 24, y: 28, w: 592, h: 30};
  const parts: [string, number, string][] = [
    ['rules', ruleDrop, colors.rule],
    ['duplicates', dup, colors.dup],
    ['judge', judgeDrop, colors.judge],
    ['kept', kept, colors.kept],
  ];
  let cursor = BAR.x;

  const PLOT = {top: 130, h: 130, left: 56, w: 230, gap: 330};
  const px = (panel: number, i: number) => PLOT.left + panel * PLOT.gap + (i / (ROUGE.length - 1)) * PLOT.w;
  const keptMax = 450;
  const pyKept = (v: number) => PLOT.top + PLOT.h - (v / keptMax) * PLOT.h;
  const pyNear = (v: number) => PLOT.top + PLOT.h - v * PLOT.h;
  const path = (panel: number, f: (c: number[]) => number) =>
    row.map((c, i) => `${i ? 'L' : 'M'}${px(panel, i).toFixed(1)},${f(c).toFixed(1)}`).join(' ');

  const table = row.map((c, i) => [
    ROUGE[i].toFixed(2),
    c[1],
    c[2],
    c[3],
    c[4].toFixed(3),
    c[5].toFixed(3),
    c[6].toFixed(3),
  ]);

  return (
    <VizPanel
      title="Filter thresholds, kept volume and diversity"
      hint="Lower the ROUGE-L threshold and more rows count as duplicates, so fewer survive and near-duplicates vanish, but a very tight filter keeps few rows and throws away useful variants. Raise the judge threshold and the volume falls without removing near-duplicates. Defaults match block 1: 130 dropped by rules, 281 as duplicates, 88 by the judge, 101 kept, distinct-2 0.187, mean pairwise cosine 0.249. Distinct-2 looks high when only a handful of rows remain, because it is measured on very little text."
      legend={[
        {label: 'dropped by rules', color: colors.rule},
        {label: 'dropped as duplicates', color: colors.dup},
        {label: 'dropped by the judge', color: colors.judge},
        {label: 'kept', color: colors.kept},
      ]}
      table={{
        columns: ['ROUGE-L threshold', 'duplicates', 'judge drops', 'kept', 'distinct-2', 'mean cosine', 'near share'],
        rows: table,
      }}
      controls={
        <>
          <label className={s.control}>
            judge threshold
            <input type="range" min={5} max={9} step={0.5} value={judge} onChange={(e) => setJudge(Number(e.target.value))} />
            <span className={s.value}>{judge.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            ROUGE-L threshold
            <input type="range" min={0.4} max={1} step={0.05} value={rouge} onChange={(e) => setRouge(Math.round(Number(e.target.value) * 100) / 100)} />
            <span className={s.value}>{rouge.toFixed(2)}</span>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Of 600 generated rows, ${ruleDrop} dropped by rules, ${dup} as duplicates, ${judgeDrop} by the judge, ${kept} kept`}>
        <text className={s.axisLabel} x={BAR.x} y={BAR.y - 8}>600 generated rows</text>
        {parts.map(([label, count, fill]) => {
          const w = (count / TOTAL) * BAR.w;
          const x = cursor;
          cursor += w;
          return (
            <g key={label}>
              <rect x={x} y={BAR.y} width={w} height={BAR.h} fill={fill} stroke="var(--surface-raised)" strokeWidth={1.5} />
              {w > 34 && (
                <text className={s.dataLabel} x={x + w / 2} y={BAR.y + BAR.h / 2 + 4} textAnchor="middle" fill="#fff">
                  {count}
                </text>
              )}
            </g>
          );
        })}
        <text className={s.dataLabel} x={BAR.x} y={BAR.y + BAR.h + 22}>
          kept {kept} | distinct-2 {distinct2.toFixed(3)} | mean pairwise cosine {meanCos.toFixed(3)} | share with a neighbour above 0.9: {near.toFixed(3)}
        </text>
        {[0, 1].map((panel) => (
          <g key={panel}>
            <text className={s.axisLabel} x={PLOT.left} y={PLOT.top - 14}>
              {panel === 0 ? 'rows kept' : 'share of kept rows with a neighbour above cosine 0.9'}
            </text>
            <line className={s.axis} x1={px(panel, 0)} y1={PLOT.top + PLOT.h} x2={px(panel, ROUGE.length - 1)} y2={PLOT.top + PLOT.h} />
            <line className={s.axis} x1={px(panel, 0)} y1={PLOT.top} x2={px(panel, 0)} y2={PLOT.top + PLOT.h} />
            {[0.4, 0.7, 1.0].map((t) => (
              <text key={t} className={s.tick} x={px(panel, ROUGE.indexOf(t))} y={PLOT.top + PLOT.h + 14} textAnchor="middle">
                {t.toFixed(1)}
              </text>
            ))}
            <text className={s.tick} x={px(panel, 0) - 6} y={PLOT.top + 4} textAnchor="end">{panel === 0 ? keptMax : 1}</text>
            <text className={s.tick} x={px(panel, 0) - 6} y={PLOT.top + PLOT.h + 3} textAnchor="end">0</text>
            <text className={s.axisLabel} x={px(panel, 0) + PLOT.w / 2} y={PLOT.top + PLOT.h + 30} textAnchor="middle">ROUGE-L threshold</text>
          </g>
        ))}
        <path d={path(0, (c) => pyKept(c[3]))} fill="none" stroke={colors.kept} strokeWidth={2.5} />
        <path d={path(1, (c) => pyNear(c[6]))} fill="none" stroke={colors.dup} strokeWidth={2.5} />
        <circle cx={px(0, ri)} cy={pyKept(kept)} r={5} fill={colors.kept} stroke="var(--surface-raised)" strokeWidth={2} />
        <circle cx={px(1, ri)} cy={pyNear(near)} r={5} fill={colors.dup} stroke="var(--surface-raised)" strokeWidth={2} />
      </svg>
    </VizPanel>
  );
}
