import {useMemo, useState} from 'react';

import {aliasFrequency, twoToneSpectrum} from './speechMath';
import {SpeechSelect, SpeechSlider} from './speechLabParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const FIRST_TONE = 1000;
const W = 640;
const H = 250;
const LEFT = 44;
const RIGHT = 16;
const TOP = 16;
const BOTTOM = 40;

export default function SpectrumResolutionLab() {
  const dark = useDarkViz();
  const [windowMs, setWindowMs] = useState(25);
  const [gap, setGap] = useState(60);
  const [tone, setTone] = useState(700);
  const [rate, setRate] = useState(1000);

  const spectrum = useMemo(() => twoToneSpectrum(FIRST_TONE, FIRST_TONE + gap, windowMs), [windowMs, gap]);
  const seen = aliasFrequency(tone, rate);
  const nyquist = rate / 2;

  const fMin = spectrum.freqs[0];
  const fMax = spectrum.freqs[spectrum.freqs.length - 1];
  const top = Math.max(...spectrum.mags) || 1;
  const x = (f: number) => LEFT + ((f - fMin) / (fMax - fMin || 1)) * (W - LEFT - RIGHT);
  const y = (m: number) => TOP + (1 - m / top) * (H - TOP - BOTTOM);
  const path = spectrum.freqs.map((f, i) => `${i ? 'L' : 'M'}${x(f).toFixed(1)},${y(spectrum.mags[i]).toFixed(1)}`).join(' ');
  const line = seriesColor(0, dark);
  const mark = seriesColor(1, dark);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => fMin + t * (fMax - fMin));

  const rows: (string | number)[][] = [
    ['Window', `${windowMs} ms = ${spectrum.samples} samples`],
    ['Bin width (rate / samples)', `${spectrum.binWidth.toFixed(1)} Hz`],
    ['Second tone', `${FIRST_TONE + gap} Hz`],
    ['Peaks above half height', spectrum.peaks.length ? spectrum.peaks.map((p) => p.toFixed(1)).join(', ') + ' Hz' : 'none'],
    ['Two tones resolved', spectrum.resolved ? 'yes' : 'no'],
    ['Tone played', `${tone} Hz`],
    ['Sampling rate', `${rate} Hz (Nyquist ${nyquist} Hz)`],
    ['Frequency seen', `${seen.toFixed(0)} Hz`],
  ];

  return (
    <div data-testid="spectrum-lab">
      <VizPanel
        title="Can a short window tell two tones apart, and what does a low sampling rate do to a tone?"
        hint="Defaults: a 25 ms window, a second tone 60 Hz above 1000 Hz, and a 700 Hz tone recorded at 1000 Hz. The chapter prints that 25 ms resolves a 30 Hz gap, and that the 700 Hz tone appears at 300 Hz."
        table={{columns: ['Quantity', 'Value'], rows}}
        legend={[
          {label: 'Hann-windowed spectrum', color: line},
          {label: 'peak above half height', color: mark},
        ]}
        controls={
          <>
            <SpeechSelect label="Window" value={windowMs} options={[5, 10, 25, 50, 100]} onChange={setWindowMs} format={(v) => `${v} ms`} />
            <SpeechSlider label="Gap to second tone (Hz)" value={gap} min={5} max={400} step={5} onChange={setGap} unit=" Hz" />
            <SpeechSlider label="Tone played (Hz)" value={tone} min={100} max={7900} step={50} onChange={setTone} unit=" Hz" />
            <SpeechSelect label="Sampling rate" value={rate} options={[16000, 8000, 4000, 1000, 800]} onChange={setRate} format={(v) => `${v} Hz`} />
            <span className={s.value} aria-live="polite" data-testid="spectrum-summary">
              {spectrum.resolved ? 'two peaks' : 'one blurred peak'} at {spectrum.binWidth.toFixed(1)} Hz per bin; a {tone} Hz tone at {rate} Hz appears at {seen.toFixed(0)} Hz
            </span>
          </>
        }>
        <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Magnitude spectrum of two tones">
          <line className={s.axis} x1={LEFT} y1={H - BOTTOM} x2={W - RIGHT} y2={H - BOTTOM} />
          <line className={s.axis} x1={LEFT} y1={TOP} x2={LEFT} y2={H - BOTTOM} />
          {ticks.map((f) => (
            <g key={f}>
              <line className={s.grid} x1={x(f)} y1={TOP} x2={x(f)} y2={H - BOTTOM} />
              <text className={s.tick} x={x(f)} y={H - BOTTOM + 14} textAnchor="middle">
                {Math.round(f)}
              </text>
            </g>
          ))}
          <path d={path} fill="none" stroke={line} strokeWidth={2} />
          {spectrum.peaks.map((p) => (
            <g key={p}>
              <line x1={x(p)} y1={TOP} x2={x(p)} y2={H - BOTTOM} stroke={mark} strokeWidth={1.5} strokeDasharray="4 3" />
              <text className={s.dataLabel} x={x(p)} y={TOP + 10} textAnchor="middle">
                {p.toFixed(0)}
              </text>
            </g>
          ))}
          <text className={s.axisLabel} x={W / 2} y={H - 6} textAnchor="middle">
            frequency (Hz)
          </text>
          <text className={s.axisLabel} x={12} y={H / 2} textAnchor="middle" transform={`rotate(-90 12 ${H / 2})`}>
            magnitude
          </text>
        </svg>
      </VizPanel>
    </div>
  );
}
