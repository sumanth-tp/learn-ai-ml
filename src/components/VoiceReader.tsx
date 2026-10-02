import {useEffect, useRef, useState} from 'react';

import styles from './VoiceReader.module.css';

type Playback = 'idle' | 'playing' | 'paused';

const MAX_CHUNK_LENGTH = 220;

function readableChunks(root: HTMLElement): string[] {
  const chunks: string[] = [];
  const blocks = root.querySelectorAll<HTMLElement>('h1, h2, h3, h4, p, li, blockquote, th, td');

  for (const block of blocks) {
    if (block.closest('pre, script, style, nav, [aria-hidden="true"], details:not([open])')) {
      continue;
    }
    if (block.matches('p') && block.parentElement?.closest('li, blockquote, th, td')) {
      continue;
    }

    const copy = block.cloneNode(true) as HTMLElement;
    copy.querySelectorAll('pre, svg, button, .katex, [aria-hidden="true"]').forEach((element) => element.remove());
    if (block.matches('li')) {
      copy.querySelectorAll('ul, ol').forEach((element) => element.remove());
    }
    const text = (copy.textContent ?? '').replace(/\s+/g, ' ').trim();
    if (!text) {
      continue;
    }

    let part = '';
    for (const word of text.split(' ')) {
      if (part && part.length + word.length + 1 > MAX_CHUNK_LENGTH) {
        chunks.push(part);
        part = word;
      } else {
        part = part ? `${part} ${word}` : word;
      }
    }
    if (part) {
      chunks.push(part);
    }
  }

  return chunks;
}

export default function VoiceReader({articleRef, permalink}: {
  articleRef: React.RefObject<HTMLDivElement | null>;
  permalink: string;
}) {
  const [supported, setSupported] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [voiceURI, setVoiceURI] = useState('');
  const [rate, setRate] = useState(1);
  const [playback, setPlayback] = useState<Playback>('idle');
  const [progress, setProgress] = useState({current: 0, total: 0});
  const [error, setError] = useState('');
  const generation = useRef(0);

  useEffect(() => {
    if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) {
      return;
    }

    setSupported(true);
    const synth = window.speechSynthesis;
    const updateVoices = () => {
      const local = synth.getVoices().filter((voice) => voice.localService);
      setVoices(local);
      setVoiceURI((previous) => {
        if (local.some((voice) => voice.voiceURI === previous)) {
          return previous;
        }
        const language = navigator.language.split('-')[0];
        return local.find((voice) => voice.lang.toLowerCase().startsWith(language.toLowerCase()))?.voiceURI
          ?? local[0]?.voiceURI
          ?? '';
      });
    };

    updateVoices();
    synth.addEventListener('voiceschanged', updateVoices);
    return () => synth.removeEventListener('voiceschanged', updateVoices);
  }, []);

  useEffect(() => () => {
    generation.current += 1;
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }, [permalink]);

  function stop() {
    generation.current += 1;
    window.speechSynthesis.cancel();
    setPlayback('idle');
    setProgress({current: 0, total: 0});
  }

  function start() {
    const root = articleRef.current;
    const voice = voices.find((item) => item.voiceURI === voiceURI);
    if (!root || !voice) {
      return;
    }
    const selectedVoice = voice;

    const chunks = readableChunks(root);
    if (chunks.length === 0) {
      setError('There is no readable text on this page.');
      return;
    }

    const synth = window.speechSynthesis;
    const currentGeneration = ++generation.current;
    synth.cancel();
    setError('');
    setPlayback('playing');

    function speak(index: number) {
      if (currentGeneration !== generation.current) {
        return;
      }
      if (index === chunks.length) {
        setPlayback('idle');
        setProgress({current: 0, total: 0});
        return;
      }

      const utterance = new SpeechSynthesisUtterance(chunks[index]);
      utterance.voice = selectedVoice;
      utterance.lang = selectedVoice.lang;
      utterance.rate = rate;
      utterance.onend = () => speak(index + 1);
      utterance.onerror = (event) => {
        if (currentGeneration !== generation.current) {
          return;
        }
        setPlayback('idle');
        setProgress({current: 0, total: 0});
        setError(`Speech stopped: ${event.error}.`);
      };
      setProgress({current: index + 1, total: chunks.length});
      synth.speak(utterance);
    }

    speak(0);
  }

  const available = supported && voices.length > 0;
  const busy = playback !== 'idle';

  return (
    <div className={styles.reader} role="group" aria-label="Voice reader">
      <button
        type="button"
        className={styles.button}
        onClick={playback === 'idle' ? start : playback === 'paused'
          ? () => { window.speechSynthesis.resume(); setPlayback('playing'); }
          : () => { window.speechSynthesis.pause(); setPlayback('paused'); }}
        disabled={!available}
        title={!supported ? 'Voice reading is unavailable in this browser' : voices.length === 0
          ? 'No local voice is available on this device' : undefined}>
        {playback === 'playing' ? 'Pause' : playback === 'paused' ? 'Resume' : 'Read aloud'}
      </button>
      {busy && (
        <button type="button" className={styles.button} onClick={stop}>Stop</button>
      )}
      <select
        className={styles.select}
        aria-label="Reader voice"
        value={voiceURI}
        onChange={(event) => setVoiceURI(event.target.value)}
        disabled={!available || busy}>
        {voices.length === 0 && <option value="">No local voice</option>}
        {voices.map((voice) => (
          <option key={voice.voiceURI} value={voice.voiceURI}>{voice.name} ({voice.lang})</option>
        ))}
      </select>
      <select
        className={styles.select}
        aria-label="Reader speed"
        value={rate}
        onChange={(event) => setRate(Number(event.target.value))}
        disabled={!available || busy}>
        <option value={0.8}>0.8×</option>
        <option value={1}>1×</option>
        <option value={1.2}>1.2×</option>
        <option value={1.5}>1.5×</option>
      </select>
      <span className={styles.status} role="status">
        {error || (busy ? `${progress.current} of ${progress.total}` : '')}
      </span>
    </div>
  );
}
