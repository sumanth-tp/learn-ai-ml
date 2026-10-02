export type RooflinePoint = {
  intensity: number;
  stepMs: number;
  tokensPerSecond: number;
  computeUtil: number;
  bound: 'compute' | 'memory';
};

export function rooflinePoint(
  peakTflops: number,
  bandwidthTb: number,
  paramsBillions: number,
  bytesPerParam: number,
  rows: number,
): RooflinePoint {
  const params = paramsBillions * 1e9;
  const flops = 2 * params * rows;
  const traffic = params * bytesPerParam;
  const tCompute = flops / (peakTflops * 1e12);
  const tMemory = traffic / (bandwidthTb * 1e12);
  const t = Math.max(tCompute, tMemory);
  return {
    intensity: flops / traffic,
    stepMs: t * 1e3,
    tokensPerSecond: rows / t,
    computeUtil: tCompute / t,
    bound: tCompute > tMemory ? 'compute' : 'memory',
  };
}

export type KvModel = {
  name: string;
  kind: 'MHA' | 'GQA' | 'MQA' | 'MLA';
  layers: number;
  kvHeads: number;
  headDim: number;
  latent?: number;
  decoupledKey?: number;
};

export const KV_MODELS: KvModel[] = [
  {name: 'gpt2', kind: 'MHA', layers: 12, kvHeads: 12, headDim: 64},
  {name: 'falcon-7b', kind: 'MQA', layers: 32, kvHeads: 1, headDim: 64},
  {name: 'Mistral-7B-v0.1', kind: 'GQA', layers: 32, kvHeads: 8, headDim: 128},
  {name: 'Llama 3.1 8B', kind: 'GQA', layers: 32, kvHeads: 8, headDim: 128},
  {name: 'Qwen2.5-7B-Instruct', kind: 'GQA', layers: 28, kvHeads: 4, headDim: 128},
  {name: 'Llama 3.1 70B', kind: 'GQA', layers: 80, kvHeads: 8, headDim: 128},
  {name: 'SmolLM2-135M-Instruct', kind: 'GQA', layers: 30, kvHeads: 3, headDim: 64},
  {name: 'DeepSeek-V2-Lite', kind: 'MLA', layers: 27, kvHeads: 0, headDim: 0, latent: 512, decoupledKey: 64},
];

export function kvElementsPerToken(m: KvModel): number {
  if (m.kind === 'MLA') return ((m.latent ?? 0) + (m.decoupledKey ?? 0)) * m.layers;
  return 2 * m.layers * m.kvHeads * m.headDim;
}

export function kvSequencesThatFit(
  budgetGb: number,
  bytesPerToken: number,
  tokensPerSequence: number,
  blockSize: number,
): number {
  const slots = Math.ceil(tokensPerSequence / blockSize) * blockSize;
  return Math.floor((budgetGb * 1e9) / (slots * bytesPerToken));
}

export const WEIGHT_MS = 4.79;
export const COMPUTE_MS_PER_TOKEN = 0.01623;
export const KV_MS_PER_TOKEN = 3.913e-5;

export type Req = {id: number; arrival: number; prompt: number; output: number};

export type BatchingResult = {
  tokensPerS: number;
  latencyMeanS: number;
  latencyP99S: number;
  ttftMeanMs: number;
  ttftP99Ms: number;
  gapP99Ms: number;
  makespanS: number;
  inSystem: number;
  firstToken: number[];
  finish: number[];
};

export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeTrace(n: number, ratePerS: number, seed = 7): Req[] {
  const draw = mulberry32(seed);
  const normal = () => Math.sqrt(-2 * Math.log(1 - draw())) * Math.cos(2 * Math.PI * draw());
  let clock = 0;
  const trace: Req[] = [];
  for (let i = 0; i < n; i += 1) {
    clock += (-Math.log(1 - draw()) / ratePerS) * 1000;
    const prompt = Math.min(2048, Math.max(8, Math.round(Math.exp(5.2 + 0.8 * normal()))));
    const output = Math.min(1024, Math.max(8, Math.round(Math.exp(4.8 + 0.9 * normal()))));
    trace.push({id: i, arrival: clock, prompt, output});
  }
  return trace;
}

export function stepMs(batchTokens: number, contextTokens: number): number {
  return Math.max(WEIGHT_MS, COMPUTE_MS_PER_TOKEN * batchTokens) + KV_MS_PER_TOKEN * contextTokens;
}

function percentile(values: number[], q: number): number {
  const ordered = values.slice().sort((a, b) => a - b);
  return ordered[Math.min(ordered.length - 1, Math.ceil(q * ordered.length) - 1)];
}

function summarise(
  trace: Req[],
  firstToken: number[],
  finish: number[],
  gaps: number[],
  makespan: number,
  inSystem: number,
): BatchingResult {
  const latency = trace.map((r) => finish[r.id] - r.arrival);
  const ttft = trace.map((r) => firstToken[r.id] - r.arrival);
  const tokens = trace.reduce((a, r) => a + r.output, 0);
  return {
    tokensPerS: tokens / (makespan / 1000),
    latencyMeanS: latency.reduce((a, b) => a + b, 0) / latency.length / 1000,
    latencyP99S: percentile(latency, 0.99) / 1000,
    ttftMeanMs: ttft.reduce((a, b) => a + b, 0) / ttft.length,
    ttftP99Ms: percentile(ttft, 0.99),
    gapP99Ms: percentile(gaps, 0.99),
    makespanS: makespan / 1000,
    inSystem,
    firstToken,
    finish,
  };
}

export function runStatic(trace: Req[], maxBatch: number): BatchingResult {
  let t = 0;
  let i = 0;
  const firstToken: number[] = new Array(trace.length).fill(0);
  const finish: number[] = new Array(trace.length).fill(0);
  const gaps: number[] = [];
  while (i < trace.length) {
    t = Math.max(t, trace[i].arrival);
    const batch: Req[] = [];
    while (i < trace.length && batch.length < maxBatch && trace[i].arrival <= t) {
      batch.push(trace[i]);
      i += 1;
    }
    t += stepMs(batch.reduce((a, r) => a + r.prompt, 0), 0);
    for (const r of batch) firstToken[r.id] = t;
    const longest = Math.max(...batch.map((r) => r.output));
    for (let k = 1; k < longest; k += 1) {
      const context = batch.reduce((a, r) => a + r.prompt + Math.min(k, r.output), 0);
      const d = stepMs(batch.length, context);
      t += d;
      const live = batch.filter((r) => r.output > k).length;
      for (let g = 0; g < live; g += 1) gaps.push(d);
    }
    for (const r of batch) finish[r.id] = t;
  }
  return summarise(trace, firstToken, finish, gaps, t, 0);
}

type Slot = {req: Req; prefilled: number; generated: number};

export function runContinuous(
  trace: Req[],
  maxBatch: number,
  tokenBudget: number | null = null,
  shortestFirst = false,
): BatchingResult {
  let t = 0;
  let i = 0;
  let waiting: Req[] = [];
  let running: Slot[] = [];
  const firstToken: number[] = new Array(trace.length).fill(0);
  const finish: number[] = new Array(trace.length).fill(0);
  const gaps: number[] = [];
  let done = 0;
  let area = 0;
  while (done < trace.length) {
    while (i < trace.length && trace[i].arrival <= t) {
      waiting.push(trace[i]);
      i += 1;
    }
    if (running.length === 0 && waiting.length === 0) {
      t = trace[i].arrival;
      continue;
    }
    if (shortestFirst) waiting.sort((a, b) => a.prompt - b.prompt || a.id - b.id);
    while (waiting.length > 0 && running.length < maxBatch) {
      running.push({req: waiting.shift() as Req, prefilled: 0, generated: 0});
    }
    const decoding = running.filter((s) => s.prefilled === s.req.prompt);
    const budget = tokenBudget === null ? null : Math.max(0, tokenBudget - decoding.length);
    let chunkTokens = 0;
    const chunks: [Slot, number][] = [];
    for (const s of running) {
      const need = s.req.prompt - s.prefilled;
      if (need === 0) continue;
      const take = budget === null ? need : Math.min(need, budget - chunkTokens);
      if (take <= 0) break;
      chunks.push([s, take]);
      chunkTokens += take;
    }
    const context = decoding.reduce((a, s) => a + s.req.prompt + s.generated, 0);
    const d = stepMs(decoding.length + chunkTokens, context);
    area += (running.length + waiting.length) * d;
    t += d;
    for (let g = 0; g < decoding.length; g += 1) gaps.push(d);
    for (const s of decoding) s.generated += 1;
    for (const [s, take] of chunks) {
      s.prefilled += take;
      if (s.prefilled === s.req.prompt) {
        s.generated = 1;
        firstToken[s.req.id] = t;
      }
    }
    const still: Slot[] = [];
    for (const s of running) {
      if (s.generated >= s.req.output) {
        finish[s.req.id] = t;
        done += 1;
      } else {
        still.push(s);
      }
    }
    running = still;
  }
  return summarise(trace, firstToken, finish, gaps, t, area / t);
}

export const WEIGHT_SLICE: number[] = [0.0427, -0.0601, 0.0237, 0.0703, -0.0806, -0.0311, 0.1299, 0.0294, 0.0101, 0.0422, 0.0454, -0.0366, 0.0119, -0.1221, -0.1533, 0.1367, -0.0127, 0.0175, -0.0261, -0.1318, -0.042, -0.0957, -0.0006, -0.0762, -0.0132, 0.0718, 0.0535, -0.0179, 0.1182, 0.0522, 0.025, -0.0486, 0.0405, 0.0562, -0.0098, 0.0098, 0.0713, 0.0386, -0.0099, -0.0522, -0.1089, 0.0025, 0.0942, 0.0879, 0.0767, -0.0488, -0.05, -0.0444, 0.0396, 0.0155, 0.0645, 0.0815, 0.0198, -0.0247, -0.0383, 0.0508, 0.1089, -0.1055, -0.0493, -0.0515, 0.106, -0.0203, -0.0474, 0.1543, 0.0344, 0.084, -0.0571, -0.0007, 0.0344, 0.0187, 0.0767, 0.0054, -0.0737, 0.0342, 0.103, -0.1377, 0.0483, -0.0245, 0.0767, -0.0201, 0.0552, -0.0156, 0.0588, -0.0303, -0.0383, 0.0869, -0.0306, -0.0007, -0.04, 0.0322, -0.0107, -0.0713, -0.1206, 0.0304, -0.1162, 0.1094, -0.0143, 0.0192, 0.0422, 0.0938, -3.2188, 0.0713, 0.1836, 0.0171, -0.1045, -0.0601, -0.0854, 0.0055, 0.0442, -0.0378, 0.1504, -0.127, 0.0747, -0.0442, 0.0913, 0.1504, -0.0386, -0.0008, 0.0459, 0.0791, -0.0664, 0.085, 0.1191, -0.0884, 0.1289, 0.0286, 0.0776, -0.0067, 0.1235, -0.0364, 0.0439, -0.0167, 0.0267, -0.0322, 0.0596, -0.0076, -0.014, -0.1006, -0.0247, -0.0471, -0.0113, 0.0559, -0.009, -0.0261, -0.0189, 0.0239, -0.0618, 0.0547, 0.1177, -0.0537, 0.0537, 0.0542, -0.0167, 0.0034, 0.1367, 0.0109, 0.0096, 0.1611, -0.1914, -0.1226, 0.0287, 0.0182, -0.0728, 0.0496, -0.0272, 0.1001, -0.105, 0.0286, 0.0544, -0.0376, 0.0104, -0.0034, 0.0022, -0.0352, -0.0159, -0.063, 0.022, 0.033, 0.0845, 0.0391, 0.1211, -0.0547, 0.0552, -0.0515, 0.0077, 0.0136, -0.084, -0.1079, 0.0415, 0.0693, 0.0459, -0.0349, 0.085, 0.0986, -0.0466, -0.0405, 0.0413, 0.0815, -0.0728, 0.0111, 0.0228, -0.0031, -0.0121, -0.0376, -0.0559, 0.0669, 0.0417, 0.031, -0.0284, -0.0011, -0.0532, -0.0006, -0.0322, 0.0654, 0.0938, 0.0403, -0.0369, 0.016, 0.0267, 0.012, 0.0364, -0.1465, 0.0679, 0.0288, 0.0796, 0.0088, 0.0986, 0.0439, 0.0108, -0.1357, 0.0364, -0.028, 0.1157, -0.0067, 0.0286, 0.0918, -0.032, 0.0518, 0.0942, -0.0825, 0.1787, 0.0153, 0.0137, -0.032, 0.0605, 0.051, -0.0437, 0.0094, 0.0811, -0.1641, -0.0214, -0.0537, -0.0371, -0.0432, 0.0203, -0.0146];

export type QuantScheme = 'absmax' | 'zeropoint';

export type SliceQuant = {
  recon: number[];
  relativeError: number;
  mse: number;
  maxError: number;
  bitsPerWeight: number;
};

export function quantiseSlice(
  bits: number,
  group: number,
  scheme: QuantScheme,
  keepLargest: boolean,
): SliceQuant {
  const x = WEIGHT_SLICE;
  let big = -1;
  if (keepLargest) {
    big = 0;
    for (let i = 1; i < x.length; i += 1) if (Math.abs(x[i]) > Math.abs(x[big])) big = i;
  }
  const recon = x.slice();
  for (let start = 0; start < x.length; start += group) {
    const idx: number[] = [];
    for (let i = start; i < Math.min(start + group, x.length); i += 1) if (i !== big) idx.push(i);
    if (idx.length === 0) continue;
    if (scheme === 'absmax') {
      const qmax = 2 ** (bits - 1) - 1;
      const scale = Math.max(1e-12, Math.max(...idx.map((i) => Math.abs(x[i])))) / qmax;
      for (const i of idx) {
        const code = Math.max(-qmax, Math.min(qmax, roundHalfEven(x[i] / scale)));
        recon[i] = code * scale;
      }
    } else {
      const levels = 2 ** bits - 1;
      const low = Math.min(...idx.map((i) => x[i]));
      const high = Math.max(...idx.map((i) => x[i]));
      const scale = Math.max(1e-12, (high - low) / levels);
      const zero = roundHalfEven(-low / scale);
      for (const i of idx) {
        const code = Math.max(0, Math.min(levels, roundHalfEven(x[i] / scale) + zero));
        recon[i] = (code - zero) * scale;
      }
    }
  }
  let num = 0;
  let den = 0;
  let maxError = 0;
  for (let i = 0; i < x.length; i += 1) {
    const e = recon[i] - x[i];
    num += e * e;
    den += x[i] * x[i];
    maxError = Math.max(maxError, Math.abs(e));
  }
  return {
    recon,
    relativeError: Math.sqrt(num / den),
    mse: num / x.length,
    maxError,
    bitsPerWeight: bits + 16 / group,
  };
}

export function roundHalfEven(v: number): number {
  const floor = Math.floor(v);
  const diff = v - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

export function expectedTokens(alpha: number, gamma: number): number {
  return (1 - alpha ** (gamma + 1)) / (1 - alpha);
}

export function speculativeSpeedup(alpha: number, gamma: number, c: number, v: number): number {
  return expectedTokens(alpha, gamma) / (gamma * c + v);
}

export type DraftStep = {accepted: number};

export function draftTimeline(alpha: number, gamma: number, steps: number, seed = 3): DraftStep[] {
  const draw = mulberry32(seed);
  const out: DraftStep[] = [];
  for (let s = 0; s < steps; s += 1) {
    let accepted = 0;
    let rejected = false;
    for (let i = 0; i < gamma; i += 1) {
      const u = draw();
      if (!rejected && u < alpha) accepted += 1;
      else rejected = true;
    }
    out.push({accepted});
  }
  return out;
}
