import {useState} from 'react';

import {replyTimeline} from './speechMath';
import {SpeechSlider} from './speechLabParts';
import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const LEFT = 20;
const RIGHT = 20;

export const REPLY_DEFAULTS = {
  endpointMs: 700,
  asrMs: 141,
  llmFirstTokenMs: 142,
  llmTokensPerSecond: 88,
  tokensPerSentence: 12,
  sentences: 2,
  audioSecondsPerSentence: 2.1,
  ttsRealTimeFactor: 0.22,
  networkMs: 0,
};

export default function ReplyLatencyLab() {
  const dark = useDarkViz();
  const [endpointMs, setEndpointMs] = useState(REPLY_DEFAULTS.endpointMs);
  const [asrMs, setAsrMs] = useState(REPLY_DEFAULTS.asrMs);
  const [llmFirstTokenMs, setLlmFirst] = useState(REPLY_DEFAULTS.llmFirstTokenMs);
  const [llmTokensPerSecond, setTps] = useState(REPLY_DEFAULTS.llmTokensPerSecond);
  const [tokensPerSentence, setTokens] = useState(REPLY_DEFAULTS.tokensPerSentence);
  const [sentences, setSentences] = useState(REPLY_DEFAULTS.sentences);
  const [audioSecondsPerSentence, setAudio] = useState(REPLY_DEFAULTS.audioSecondsPerSentence);
  const [ttsRealTimeFactor, setRtf] = useState(REPLY_DEFAULTS.ttsRealTimeFactor);
  const [networkMs, setNetwork] = useState(REPLY_DEFAULTS.networkMs);

  const r = replyTimeline({endpointMs, asrMs, llmFirstTokenMs, llmTokensPerSecond, tokensPerSentence, sentences, audioSecondsPerSentence, ttsRealTimeFactor, networkMs});
  const firstSentenceLlm = r.rows[0].ready - endpointMs - asrMs;
  const synthFirst = ttsRealTimeFactor * audioSecondsPerSentence * 1000;
  const parts = [
    {label: 'wait for end of turn', value: endpointMs, color: seriesColor(0, dark)},
    {label: 'speech to text', value: asrMs, color: seriesColor(1, dark)},
    {label: 'language model, first sentence', value: firstSentenceLlm, color: seriesColor(2, dark)},
    {label: 'speech synthesis, first sentence', value: synthFirst, color: seriesColor(3, dark)},
    {label: 'network', value: networkMs, color: seriesColor(4, dark)},
  ];
  const total = parts.reduce((a, p) => a + p.value, 0) || 1;
  const scale = (W - LEFT - RIGHT) / Math.max(total, r.waitForAll);
  let cursor = LEFT;

  const rows: (string | number)[][] = [
    ['First audio, streamed by sentence', `${r.firstAudio.toFixed(0)} ms`],
    ['First audio, waiting for the whole reply', `${r.waitForAll.toFixed(0)} ms`],
    ['Saved by streaming', `${r.saved.toFixed(0)} ms`],
    ['Silent gaps while the reply plays', `${r.stallMs.toFixed(0)} ms`],
    ['Reply finishes playing at', `${r.finish.toFixed(0)} ms`],
  ];

  return (
    <div data-testid="reply-latency-lab">
      <VizPanel
        title="How long after you stop talking does a voice agent start to answer?"
        hint="Defaults are this chapter's measured laptop pipeline with a 700 ms end-of-turn rule: 141 ms to transcribe, 142 ms to the first token, 88 tokens a second, two 12-token sentences of about 2.1 s of audio each, and speech synthesis at 0.22 times real time. The first audio arrives 1,570 ms after the last word, against 2,168 ms if the agent waited for the whole reply."
        table={{columns: ['Quantity', 'Value'], rows}}
        legend={parts.map((p) => ({label: p.label, color: p.color}))}
        controls={
          <>
            <SpeechSlider label="End-of-turn wait" value={endpointMs} min={100} max={1200} step={50} onChange={setEndpointMs} unit=" ms" />
            <SpeechSlider label="Speech to text" value={asrMs} min={50} max={1000} step={25} onChange={setAsrMs} unit=" ms" />
            <SpeechSlider label="Model first token" value={llmFirstTokenMs} min={50} max={1000} step={25} onChange={setLlmFirst} unit=" ms" />
            <SpeechSlider label="Model speed (tokens/s)" value={llmTokensPerSecond} min={5} max={150} step={1} onChange={setTps} />
            <SpeechSlider label="Tokens per sentence" value={tokensPerSentence} min={4} max={40} step={2} onChange={setTokens} />
            <SpeechSlider label="Sentences in reply" value={sentences} min={1} max={6} step={1} onChange={setSentences} />
            <SpeechSlider label="Audio per sentence (s)" value={audioSecondsPerSentence} min={0.5} max={6} step={0.1} digits={1} onChange={setAudio} unit=" s" />
            <SpeechSlider label="Synthesis real-time factor" value={ttsRealTimeFactor} min={0.1} max={2} step={0.01} digits={2} onChange={setRtf} />
            <SpeechSlider label="Network" value={networkMs} min={0} max={400} step={20} onChange={setNetwork} unit=" ms" />
            <span className={s.value} aria-live="polite" data-testid="reply-summary">
              first audio {r.firstAudio.toFixed(0)} ms after the last word, {r.waitForAll.toFixed(0)} ms if it waited for the whole reply, {r.stallMs.toFixed(0)} ms of silent gaps
            </span>
          </>
        }>
        <svg className={s.svg} viewBox={`0 0 ${W} 130`} role="img" aria-label="Where the time goes before the first audio plays">
          {parts.map((p) => {
            const width = p.value * scale;
            const x = cursor;
            cursor += width;
            return <rect key={p.label} x={x} y={18} width={Math.max(0, width)} height={34} fill={p.color} opacity={0.92} />;
          })}
          <text className={s.dataLabel} x={LEFT} y={12}>
            first audio at {r.firstAudio.toFixed(0)} ms
          </text>
          <rect x={LEFT} y={74} width={r.waitForAll * scale} height={22} fill={seriesColor(1, dark)} opacity={0.45} />
          <text className={s.dataLabel} x={LEFT} y={70}>
            waiting for the whole reply: {r.waitForAll.toFixed(0)} ms
          </text>
          <text className={s.tick} x={LEFT} y={122}>
            0 ms
          </text>
          <text className={s.tick} x={W - RIGHT} y={122} textAnchor="end">
            {Math.max(total, r.waitForAll).toFixed(0)} ms
          </text>
        </svg>
      </VizPanel>
    </div>
  );
}
