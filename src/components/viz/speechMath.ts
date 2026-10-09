export function aliasFrequency(freq: number, rate: number): number {
  const wrapped = ((freq % rate) + rate) % rate;
  return wrapped > rate / 2 ? rate - wrapped : wrapped;
}

export type SpectrumResult = {
  samples: number;
  binWidth: number;
  freqs: number[];
  mags: number[];
  peaks: number[];
  resolved: boolean;
};

export function twoToneSpectrum(f1: number, f2: number, windowMs: number, rate = 16000, zeroPad = 16): SpectrumResult {
  const n = Math.floor((rate * windowMs) / 1000);
  const binWidth = rate / n;
  const step = rate / (zeroPad * n);
  const lo = Math.max(0, Math.min(f1, f2) - 6 * binWidth);
  const hi = Math.min(rate / 2, Math.max(f1, f2) + 6 * binWidth);
  const first = Math.ceil(lo / step);
  const last = Math.floor(hi / step);
  const windowed = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
    windowed[i] = w * (Math.sin((2 * Math.PI * f1 * i) / rate) + Math.sin((2 * Math.PI * f2 * i) / rate));
  }
  const freqs: number[] = [];
  const mags: number[] = [];
  for (let k = first; k <= last; k += 1) {
    const freq = k * step;
    const omega = (2 * Math.PI * freq) / rate;
    let re = 0;
    let im = 0;
    for (let i = 0; i < n; i += 1) {
      re += windowed[i] * Math.cos(omega * i);
      im -= windowed[i] * Math.sin(omega * i);
    }
    freqs.push(freq);
    mags.push(Math.hypot(re, im));
  }
  const top = Math.max(...mags);
  const peaks: number[] = [];
  for (let i = 1; i < mags.length - 1; i += 1) {
    if (mags[i] > mags[i - 1] && mags[i] >= mags[i + 1] && mags[i] >= top * 0.5) peaks.push(freqs[i]);
  }
  return {samples: n, binWidth, freqs, mags, peaks, resolved: peaks.length === 2};
}

export function smallestResolvedGap(f1: number, windowMs: number, rate = 16000, maxGap = 400, stepHz = 5): number | null {
  for (let gap = stepHz; gap < maxGap; gap += stepHz) {
    if (twoToneSpectrum(f1, f1 + gap, windowMs, rate).peaks.length === 2) return gap;
  }
  return null;
}

export type WordCounts = {substitutions: number; deletions: number; insertions: number; reference: number; wer: number};

export function normaliseText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9' ]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function wordErrors(reference: string, hypothesis: string): WordCounts {
  const r = reference.split(/\s+/).filter(Boolean);
  const h = hypothesis.split(/\s+/).filter(Boolean);
  const d: number[][] = Array.from({length: r.length + 1}, () => new Array<number>(h.length + 1).fill(0));
  for (let i = 0; i <= r.length; i += 1) d[i][0] = i;
  for (let j = 0; j <= h.length; j += 1) d[0][j] = j;
  for (let i = 1; i <= r.length; i += 1) {
    for (let j = 1; j <= h.length; j += 1) {
      const cost = r[i - 1] === h[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1);
    }
  }
  let i = r.length;
  let j = h.length;
  let substitutions = 0;
  let deletions = 0;
  let insertions = 0;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + (r[i - 1] !== h[j - 1] ? 1 : 0)) {
      if (r[i - 1] !== h[j - 1]) substitutions += 1;
      i -= 1;
      j -= 1;
    } else if (i > 0 && d[i][j] === d[i - 1][j] + 1) {
      deletions += 1;
      i -= 1;
    } else {
      insertions += 1;
      j -= 1;
    }
  }
  const total = substitutions + deletions + insertions;
  return {substitutions, deletions, insertions, reference: r.length, wer: r.length ? total / r.length : 0};
}

export const CTC_LABELS = ['_', 'a', 'b'] as const;

export function ctcCollapse(path: string): string {
  let out = '';
  let previous = '';
  for (const symbol of path) {
    if (symbol !== previous && symbol !== '_') out += symbol;
    previous = symbol;
  }
  return out;
}

export type CtcFrame = {a: number; b: number};

export function ctcFrameProbs(frame: CtcFrame): number[] {
  const a = Math.max(0, Math.min(1, frame.a));
  const b = Math.max(0, Math.min(1 - a, frame.b));
  return [1 - a - b, a, b];
}

export function ctcPaths(frames: CtcFrame[], target: string) {
  const probs = frames.map(ctcFrameProbs);
  const paths: {path: string; probability: number}[] = [];
  const total = Math.pow(3, frames.length);
  for (let code = 0; code < total; code += 1) {
    let rest = code;
    let path = '';
    let probability = 1;
    for (let t = 0; t < frames.length; t += 1) {
      const k = rest % 3;
      rest = Math.floor(rest / 3);
      path += CTC_LABELS[k];
      probability *= probs[t][k];
    }
    if (ctcCollapse(path) === target) paths.push({path, probability});
  }
  paths.sort((x, y) => (x.path < y.path ? -1 : 1));
  const sum = paths.reduce((acc, p) => acc + p.probability, 0);
  const greedyPath = probs.map((row) => CTC_LABELS[row.indexOf(Math.max(...row))]).join('');
  return {paths, total: sum, loss: sum > 0 ? -Math.log(sum) : Infinity, greedyPath, greedyText: ctcCollapse(greedyPath)};
}

export type LatencyInput = {
  endpointMs: number;
  asrMs: number;
  llmFirstTokenMs: number;
  llmTokensPerSecond: number;
  tokensPerSentence: number;
  sentences: number;
  audioSecondsPerSentence: number;
  ttsRealTimeFactor: number;
  networkMs: number;
};

export function replyTimeline(input: LatencyInput) {
  const {endpointMs, asrMs, llmFirstTokenMs, llmTokensPerSecond, tokensPerSentence, sentences, audioSecondsPerSentence, ttsRealTimeFactor, networkMs} = input;
  const textReady = endpointMs + asrMs;
  const synthMs = ttsRealTimeFactor * audioSecondsPerSentence * 1000;
  const playMs = audioSecondsPerSentence * 1000;
  let synthEnd = 0;
  let playEnd = 0;
  let stallMs = 0;
  let firstAudio = 0;
  const rows: {sentence: number; ready: number; synthEnd: number; playStart: number; stall: number}[] = [];
  for (let i = 0; i < sentences; i += 1) {
    const ready = textReady + llmFirstTokenMs + (((i + 1) * tokensPerSentence - 1) / llmTokensPerSecond) * 1000;
    const synthStart = Math.max(ready, synthEnd);
    synthEnd = synthStart + synthMs;
    const arrives = synthEnd + networkMs;
    const playStart = i === 0 ? arrives : Math.max(arrives, playEnd);
    const stall = i === 0 ? 0 : playStart - playEnd;
    if (i === 0) firstAudio = playStart;
    stallMs += stall;
    playEnd = playStart + playMs;
    rows.push({sentence: i + 1, ready, synthEnd, playStart, stall});
  }
  const wholeReplyReady = textReady + llmFirstTokenMs + ((sentences * tokensPerSentence - 1) / llmTokensPerSecond) * 1000;
  const waitForAll = wholeReplyReady + synthMs * sentences + networkMs;
  return {firstAudio, stallMs, finish: playEnd, waitForAll, saved: waitForAll - firstAudio, rows};
}
