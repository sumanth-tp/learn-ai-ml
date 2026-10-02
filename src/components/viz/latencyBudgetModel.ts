export const SAMPLES = 20000;
export const LOOKUP_SLOTS = 20;
const SLOTS = 4 + 2 * LOOKUP_SLOTS;
const Z99 = 2.3263478740408408;
const Z95 = 1.6448536269514722;

export const FIXED_STAGES: Record<string, [number, number]> = {
  'network in': [3, 12],
  model: [6, 14],
  rules: [1, 3],
  'network out': [3, 12],
};

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let cache: Float64Array | null = null;

export function normals(): Float64Array {
  if (cache) return cache;
  const next = mulberry32(20261002);
  const out = new Float64Array(SAMPLES * SLOTS);
  for (let i = 0; i < out.length; i += 1) {
    const u1 = Math.max(next(), 1e-12);
    const u2 = next();
    out[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  cache = out;
  return out;
}

const lognormal = (p50: number, p99: number, z: number) => p50 * Math.exp((Math.log(p99 / p50) / Z99) * z);

export type Config = {
  lookups: number;
  lookupP50: number;
  lookupP99: number;
  parallel: boolean;
  hedge: boolean;
  timeout: number | null;
};

export type Result = {
  totals: Float64Array;
  p50: number;
  p95: number;
  p99: number;
  p999: number;
  over: (slo: number) => number;
  degraded: number;
  extra: number;
};

export function simulate(c: Config): Result {
  const z = normals();
  const hedgeDelay = c.lookupP50 * Math.exp((Math.log(c.lookupP99 / c.lookupP50) / Z99) * Z95);
  const totals = new Float64Array(SAMPLES);
  let degraded = 0;
  let hedged = 0;
  for (let n = 0; n < SAMPLES; n += 1) {
    const base = n * SLOTS;
    let longest = 0;
    let sum = 0;
    for (let j = 0; j < c.lookups; j += 1) {
      let t = lognormal(c.lookupP50, c.lookupP99, z[base + 4 + j]);
      if (c.hedge && t > hedgeDelay) {
        hedged += 1;
        t = Math.min(t, hedgeDelay + lognormal(c.lookupP50, c.lookupP99, z[base + 4 + LOOKUP_SLOTS + j]));
      }
      if (t > longest) longest = t;
      sum += t;
    }
    let fetch = c.parallel ? longest : sum;
    if (c.timeout !== null && fetch > c.timeout) {
      fetch = c.timeout;
      degraded += 1;
    }
    totals[n] =
      lognormal(...FIXED_STAGES['network in'], z[base]) +
      lognormal(...FIXED_STAGES.model, z[base + 1]) +
      lognormal(...FIXED_STAGES.rules, z[base + 2]) +
      lognormal(...FIXED_STAGES['network out'], z[base + 3]) +
      fetch;
  }
  const sorted = Float64Array.from(totals).sort();
  const pick = (p: number) => sorted[Math.ceil(p * SAMPLES) - 1];
  return {
    totals,
    p50: pick(0.5),
    p95: pick(0.95),
    p99: pick(0.99),
    p999: pick(0.999),
    over: (slo: number) => totals.reduce((a, t) => a + (t > slo ? 1 : 0), 0) / SAMPLES,
    degraded: degraded / SAMPLES,
    extra: hedged / (SAMPLES * c.lookups),
  };
}
