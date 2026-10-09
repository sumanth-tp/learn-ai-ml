import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const DATA = {"total":124.75,"utterances":[[0.0,5.86],[7.36,12.17],[13.67,26.16],[27.66,37.55],[39.05,68.45],[69.95,78.97],[80.47,86.11],[87.61,96.84],[98.34,103.47],[104.97,123.25]],"spans":{"100":[[0.5,5.5],[7.9,11.8],[14.2,20.7],[21.3,25.7],[28.2,33.6],[33.8,37.2],[39.6,41.7],[42.0,43.3],[43.5,44.1],[44.5,49.1],[50.6,57.7],[58.8,63.5],[63.6,68.1],[70.5,78.5],[81.0,85.7],[88.2,89.8],[90.2,92.4],[92.9,94.7],[95.1,96.4],[98.9,101.3],[101.6,102.1],[102.4,103.0],[105.5,110.9],[111.3,113.2],[113.9,121.2],[121.4,122.9]],"200":[[0.5,5.5],[7.9,11.8],[14.2,20.7],[21.3,25.7],[28.2,37.2],[39.6,41.7],[42.0,44.1],[44.5,49.1],[50.6,57.7],[58.8,68.1],[70.5,78.5],[81.0,86.1],[88.2,89.8],[90.2,92.4],[92.9,94.7],[95.1,96.4],[98.9,101.3],[101.6,103.0],[105.5,110.9],[111.3,113.2],[113.9,122.9]],"300":[[0.5,5.5],[7.9,11.8],[14.2,20.7],[21.3,25.7],[28.2,37.2],[39.6,44.1],[44.5,49.1],[50.6,57.7],[58.8,68.1],[70.5,78.5],[81.0,86.1],[88.2,89.8],[90.2,92.4],[92.9,94.7],[95.1,96.4],[98.9,103.0],[105.5,110.9],[111.3,113.2],[113.9,122.9]],"500":[[0.5,5.5],[7.9,11.8],[14.2,20.7],[21.3,25.7],[28.2,37.2],[39.6,49.1],[50.6,57.7],[58.8,68.1],[70.5,78.5],[81.0,86.1],[88.2,96.4],[98.9,103.0],[105.5,113.2],[113.9,122.9]],"800":[[0.5,5.5],[7.9,11.8],[14.2,25.7],[28.2,37.2],[39.6,49.1],[50.6,57.7],[58.8,68.1],[70.5,78.5],[81.0,86.1],[88.2,96.4],[98.9,103.0],[105.5,122.9]],"1200":[[0.5,5.5],[7.9,11.8],[14.2,25.7],[28.2,37.2],[39.6,49.1],[50.6,68.1],[70.5,78.5],[81.0,86.1],[88.2,96.4],[98.9,103.0],[105.5,122.9]]}} as {total: number; utterances: [number, number][]; spans: Record<string, [number, number][]>};

const SILENCES = [100, 200, 300, 500, 800, 1200];
const CLIPS = [
  {name: 'real interruption (1.5 s of speech)', run: 46, first: 64},
  {name: 'backchannel (0.25 s of speech)', run: 7, first: 64},
  {name: 'noise burst (0.3 s)', run: 0, first: -1},
];
const CHUNK_MS = 32;
const SHOWN = 28;
const W = 640;
const H = 150;
const LEFT = 16;
const RIGHT = 16;

export default function TurnTakingLab() {
  const dark = useDarkViz();
  const [silence, setSilence] = useState(300);
  const [asr, setAsr] = useState(420);
  const [llm, setLlm] = useState(307);
  const [tts, setTts] = useState(150);
  const [net, setNet] = useState(80);
  const [clip, setClip] = useState(1);
  const [needed, setNeeded] = useState(100);

  const spans = DATA.spans[String(silence)];
  const cuts: number[] = [];
  spans.forEach((span, i) => {
    if (i === 0) return;
    const prev = spans[i - 1];
    const inside = DATA.utterances.some(([a, b]) => prev[0] >= a - 0.05 && span[1] <= b + 0.05);
    if (inside) cuts.push((prev[1] + span[0]) / 2);
  });
  const premature = spans.length - DATA.utterances.length;
  const total = silence + asr + llm + tts + net;

  const x = (t: number) => LEFT + (t / SHOWN) * (W - LEFT - RIGHT);
  const speech = seriesColor(0, dark);
  const truth = dark ? '#848c99' : '#9aa0a6';
  const bad = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;
  const ink = dark ? '#e6e8ec' : '#1f2328';

  const current = CLIPS[clip];
  const runMs = current.run * CHUNK_MS;
  const fires = runMs >= Math.max(needed, 1) && current.first >= 0;
  const stopAfter = current.first + needed;

  const rows = SILENCES.map((ms) => [ms, DATA.spans[String(ms)].length, DATA.spans[String(ms)].length - DATA.utterances.length, `${ms} ms`]);

  return (
    <VizPanel
      title="When has the user finished? Real speech, real voice activity detector"
      hint="Ten LibriSpeech utterances, 124.8 s, each followed by 1.5 s of silence, run through Silero VAD 6.2.3. A shorter silence rule answers sooner but cuts people off mid-sentence. The defaults (300 ms, 19 segments, 9 premature cut-offs) reproduce the printed row, and 1,257 ms to first audio is the printed budget line within the run-to-run noise of the timings."
      legend={[
        {label: 'speech found by the detector', color: speech},
        {label: 'what was really said', color: truth},
        {label: 'cut in the middle of an utterance', color: bad},
      ]}
      table={{columns: ['silence rule (ms)', 'segments', 'premature cut-offs', 'delay at every turn end'], rows}}
      controls={
        <>
          <label className={s.control}>
            end of turn after silence
            <select className={s.select} value={silence} onChange={(e) => setSilence(Number(e.target.value))}>
              {SILENCES.map((v) => (
                <option key={v} value={v}>
                  {v} ms
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            speech to text
            <input type="range" min={100} max={1000} step={10} value={asr} onChange={(e) => setAsr(Number(e.target.value))} />
            <span className={s.value}>{asr} ms</span>
          </label>
          <label className={s.control}>
            model first token
            <input type="range" min={100} max={1000} step={10} value={llm} onChange={(e) => setLlm(Number(e.target.value))} />
            <span className={s.value}>{llm} ms</span>
          </label>
          <label className={s.control}>
            text to speech
            <input type="range" min={50} max={600} step={10} value={tts} onChange={(e) => setTts(Number(e.target.value))} />
            <span className={s.value}>{tts} ms</span>
          </label>
          <label className={s.control}>
            network
            <input type="range" min={0} max={400} step={10} value={net} onChange={(e) => setNet(Number(e.target.value))} />
            <span className={s.value}>{net} ms</span>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`First 28 seconds of the stream with the detector's speech segments for a ${silence} ms silence rule. ${spans.length} segments over 10 utterances, ${premature} premature cut-offs.`}>
        <text className={s.axisLabel} x={LEFT} y={14}>
          what was said (grey) and what the detector heard (blue)
        </text>
        {DATA.utterances
          .filter(([a]) => a < SHOWN)
          .map(([a, b], i) => (
            <rect key={i} x={x(a)} y={28} width={x(Math.min(b, SHOWN)) - x(a)} height={26} fill="none" stroke={truth} strokeWidth={1.5} rx={3} />
          ))}
        {spans
          .filter(([a]) => a < SHOWN)
          .map(([a, b], i) => (
            <rect key={i} x={x(a)} y={62} width={Math.max(2, x(Math.min(b, SHOWN)) - x(a))} height={26} fill={speech} rx={3} />
          ))}
        {cuts
          .filter((c) => c < SHOWN)
          .map((c, i) => (
            <g key={i}>
              <line x1={x(c)} x2={x(c)} y1={24} y2={96} stroke={bad} strokeWidth={2.2} />
              <text className={s.tick} x={x(c)} y={110} textAnchor="middle" fill={bad}>
                cut
              </text>
            </g>
          ))}
        {[0, 5, 10, 15, 20, 25].map((t) => (
          <text key={t} className={s.tick} x={x(t)} y={H - 12} textAnchor="middle">
            {t} s
          </text>
        ))}
      </svg>
      <p className={s.hint} style={{padding: '0.4rem 0 0.6rem'}} aria-live="polite">
        Silence rule {silence} ms: {spans.length} segments for 10 utterances, so {premature} premature cut-offs. Time from the end of speech to the first audio of the reply:{' '}
        {silence} + {asr} + {llm} + {tts} + {net} = {total.toLocaleString('en-GB')} ms.
      </p>
      <div className={s.controls} style={{padding: '0.2rem 0'}}>
        <label className={s.control}>
          interruption sound
          <select className={s.select} value={clip} onChange={(e) => setClip(Number(e.target.value))}>
            {CLIPS.map((c, i) => (
              <option key={c.name} value={i}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        <label className={s.control}>
          speech needed before the agent stops
          <input type="range" min={0} max={400} step={10} value={needed} onChange={(e) => setNeeded(Number(e.target.value))} />
          <span className={s.value}>{needed} ms</span>
        </label>
      </div>
      <p className={s.hint} style={{padding: '0.4rem 0 0'}} aria-live="polite">
        <span style={{color: ink}}>
          {current.name}: the detector hears {runMs} ms of continuous speech.{' '}
          {fires ? `The agent stops talking ${stopAfter} ms after the sound begins.` : 'The agent keeps talking: the sound is ignored.'}
        </span>
      </p>
    </VizPanel>
  );
}
