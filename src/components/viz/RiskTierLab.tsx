import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Row = [string, string, string];

const SYSTEMS: Record<string, Row[]> = {
  "CV screening tool": [
    [
      "high-risk (Annex III point 4)",
      "Art 6(2), Chapter III Sections 1 to 3",
      "2027-12-02"
    ]
  ],
  "credit scoring model": [
    [
      "high-risk (Annex III point 5)",
      "Art 6(2), Chapter III Sections 1 to 3",
      "2027-12-02"
    ]
  ],
  "exam proctoring": [
    [
      "high-risk (Annex III point 3)",
      "Art 6(2), Chapter III Sections 1 to 3",
      "2027-12-02"
    ]
  ],
  "customer support chatbot": [
    [
      "transparency duty",
      "Art 50(1)",
      "2026-08-02"
    ]
  ],
  "image generator": [
    [
      "transparency duty",
      "Art 50(2) marking, 50(4) deepfakes",
      "2026-08-02"
    ]
  ],
  "workplace emotion inference": [
    [
      "prohibited",
      "Art 5(1)(f)",
      "2025-02-02"
    ]
  ],
  "nudification app": [
    [
      "prohibited",
      "Art 5(1)(ba), as inserted",
      "2026-12-02"
    ]
  ],
  "payslip date extractor in HR": [
    [
      "not high-risk if the assessment is documented",
      "Art 6(3) and 6(4)",
      "2027-12-02"
    ]
  ],
  "spam filter": [
    [
      "minimal risk",
      "Art 4 AI literacy only",
      "2025-02-02"
    ]
  ],
  "3e25 FLOP foundation model": [
    [
      "GPAI model with systemic risk",
      "Arts 53 and 55",
      "2025-08-02"
    ]
  ],
  "AI safety component in a medical device": [
    [
      "high-risk (Annex I product)",
      "Art 6(1), Chapter III Sections 1 to 3",
      "2028-08-02"
    ]
  ]
};

const REFERENCES = ['2026-10-02', '2026-12-02', '2027-12-02', '2028-08-02'];

const MILESTONES: {date: string; label: string}[] = [
  {date: '2025-02-02', label: 'prohibitions, AI literacy'},
  {date: '2025-08-02', label: 'general-purpose models'},
  {date: '2026-08-02', label: 'general date, Article 50'},
  {date: '2026-12-02', label: 'new prohibitions'},
  {date: '2027-12-02', label: 'high-risk, Annex III'},
  {date: '2028-08-02', label: 'high-risk, Annex I'},
];

const W = 640;
const H = 150;
const LEFT = 30;
const RIGHT = 30;

const DAY = 86400000;
const stamp = (iso: string) => Date.parse(`${iso}T00:00:00Z`);

function status(start: string, ref: string): string {
  const days = Math.round((stamp(start) - stamp(ref)) / DAY);
  return days <= 0 ? 'in application' : `in ${days} days`;
}

export default function RiskTierLab() {
  const dark = useDarkViz();
  const names = Object.keys(SYSTEMS);
  const [system, setSystem] = useState(names[0]);
  const [ref, setRef] = useState(REFERENCES[0]);

  const rows = SYSTEMS[system];
  const mark = seriesColor(1, dark);
  const line = seriesColor(0, dark);
  const faint = dark ? DIVERGING.dark.mid : DIVERGING.light.mid;

  const t0 = stamp('2025-01-01');
  const t1 = stamp('2029-01-01');
  const x = (iso: string) => LEFT + ((stamp(iso) - t0) / (t1 - t0)) * (W - LEFT - RIGHT);

  const table = names.flatMap((n) =>
    SYSTEMS[n].map((r) => [n, r[0], r[2], status(r[2], ref)]),
  );

  return (
    <VizPanel
      title="Which AI Act obligations reach a system, and when"
      hint="A teaching aid, not legal advice. The default row is the chapter's: the CV screening tool is high-risk under Annex III point 4, with the obligations applying from 2 December 2027, which is 426 days after 2 October 2026. Dates are those of Regulation (EU) 2024/1689 as amended by Regulation (EU) 2026/1744."
      legend={[
        {label: 'application date', color: line},
        {label: 'reference date', color: mark},
      ]}
      table={{columns: ['system', 'tier', 'applies from', 'status'], rows: table}}
      controls={
        <>
          <label className={s.control}>
            system
            <select className={s.select} value={system} onChange={(e) => setSystem(e.target.value)}>
              {names.map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            reference date
            <select className={s.select} value={ref} onChange={(e) => setRef(e.target.value)}>
              {REFERENCES.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Timeline of AI Act application dates with the reference date ${ref}`}>
        <line x1={LEFT} y1={70} x2={W - RIGHT} y2={70} stroke={faint} strokeWidth={2} />
        {MILESTONES.map((m, i) => (
          <g key={m.date}>
            <circle cx={x(m.date)} cy={70} r={5} fill={line} />
            <text className={s.tick} x={x(m.date)} y={i % 2 ? 100 : 46} textAnchor="middle">{m.date}</text>
            <text className={s.tick} x={x(m.date)} y={i % 2 ? 112 : 34} textAnchor="middle">{m.label}</text>
          </g>
        ))}
        <line x1={x(ref)} y1={56} x2={x(ref)} y2={84} stroke={mark} strokeWidth={3} />
        <text className={s.dataLabel} x={x(ref)} y={H - 22} textAnchor="middle" fill={mark}>reference {ref}</text>
      </svg>
      <table className={s.table}>
        <thead>
          <tr>
            <th>tier</th>
            <th>article</th>
            <th>applies from</th>
            <th>status at {ref}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[0] + r[1]}>
              <td>{r[0]}</td>
              <td>{r[1]}</td>
              <td>{r[2]}</td>
              <td>{status(r[2], ref)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </VizPanel>
  );
}
