import {useState} from 'react';

import {normaliseText, wordErrors} from './speechMath';
import {SpeechSelect} from './speechLabParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const PRESETS: {label: string; reference: string; hypothesis: string}[] = [
  {
    label: 'Whisper output as written',
    reference: 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL',
    hypothesis: 'Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.',
  },
  {
    label: 'Cat and mat',
    reference: 'the cat sat on the mat',
    hypothesis: 'the big cat sit on mat',
  },
  {
    label: 'One word, long guess',
    reference: 'yes',
    hypothesis: 'yes please send the report now',
  },
];

const W = 640;
const LEFT = 150;

export default function WerLab() {
  const dark = useDarkViz();
  const [preset, setPreset] = useState(0);
  const [reference, setReference] = useState(PRESETS[0].reference);
  const [hypothesis, setHypothesis] = useState(PRESETS[0].hypothesis);
  const [normalise, setNormalise] = useState(false);

  const ref = normalise ? normaliseText(reference) : reference;
  const hyp = normalise ? normaliseText(hypothesis) : hypothesis;
  const r = wordErrors(ref, hyp);
  const bars = [
    {label: 'substitutions', value: r.substitutions, color: seriesColor(1, dark)},
    {label: 'deletions', value: r.deletions, color: seriesColor(0, dark)},
    {label: 'insertions', value: r.insertions, color: seriesColor(3, dark)},
  ];
  const maxBar = Math.max(1, r.reference, r.substitutions + r.deletions + r.insertions);

  const choose = (index: number) => {
    setPreset(index);
    setReference(PRESETS[index].reference);
    setHypothesis(PRESETS[index].hypothesis);
  };

  const rows: (string | number)[][] = [
    ['Reference words (N)', r.reference],
    ['Substitutions (S)', r.substitutions],
    ['Deletions (D)', r.deletions],
    ['Insertions (I)', r.insertions],
    ['WER = (S + D + I) / N', r.wer.toFixed(4)],
  ];

  return (
    <div data-testid="wer-lab">
      <VizPanel
        title="Word error rate, counted the way jiwer counts it"
        hint="Defaults: Whisper's output exactly as written against an upper-case reference. Every word differs in case or punctuation, so WER is 1.0000 (17 substitutions out of 17). Tick the box to lower-case and strip punctuation: it drops to 0.0588, one word in 17, because Mr is not mister."
        table={{columns: ['Quantity', 'Value'], rows}}
        legend={bars.map((b) => ({label: b.label, color: b.color}))}
        controls={
          <>
            <SpeechSelect label="Example" value={preset} options={[0, 1, 2]} onChange={choose} format={(v) => PRESETS[v].label} />
            <label className={s.control}>
              <input type="checkbox" checked={normalise} onChange={(event) => setNormalise(event.target.checked)} aria-label="Normalise case and punctuation" />
              lower-case and drop punctuation
            </label>
            <label className={s.control} style={{flexBasis: '100%'}}>
              Reference
              <input
                type="text"
                value={reference}
                aria-label="Reference text"
                style={{flex: 1, minWidth: '12rem'}}
                onChange={(event) => setReference(event.target.value)}
              />
            </label>
            <label className={s.control} style={{flexBasis: '100%'}}>
              Hypothesis
              <input
                type="text"
                value={hypothesis}
                aria-label="Hypothesis text"
                style={{flex: 1, minWidth: '12rem'}}
                onChange={(event) => setHypothesis(event.target.value)}
              />
            </label>
            <span className={s.value} aria-live="polite" data-testid="wer-summary">
              WER {r.wer.toFixed(4)} = ({r.substitutions} + {r.deletions} + {r.insertions}) / {r.reference}
            </span>
          </>
        }>
        <svg className={s.svg} viewBox={`0 0 ${W} ${bars.length * 36 + 40}`} role="img" aria-label="Counts of substitutions, deletions and insertions">
          {bars.map((bar, i) => {
            const y = 8 + i * 36;
            const width = (bar.value / maxBar) * (W - LEFT - 80);
            return (
              <g key={bar.label}>
                <text className={s.dataLabel} x={LEFT - 10} y={y + 17} textAnchor="end">
                  {bar.label}
                </text>
                <rect x={LEFT} y={y} width={Math.max(1, width)} height={24} fill={bar.color} opacity={0.9} />
                <text className={s.dataLabel} x={LEFT + Math.max(1, width) + 6} y={y + 17}>
                  {bar.value}
                </text>
              </g>
            );
          })}
          <text className={s.axisLabel} x={W / 2} y={bars.length * 36 + 30} textAnchor="middle">
            word errors out of {r.reference} reference words
          </text>
        </svg>
      </VizPanel>
    </div>
  );
}
