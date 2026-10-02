export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function triangular(u: number, lo: number, mode: number, hi: number): number {
  const cut = (mode - lo) / (hi - lo);
  if (u < cut) return lo + Math.sqrt(u * (hi - lo) * (mode - lo));
  return hi - Math.sqrt((1 - u) * (hi - lo) * (hi - mode));
}

export function percentile(sorted: number[], q: number): number {
  const pos = ((sorted.length - 1) * q) / 100;
  const lo = Math.floor(pos);
  const frac = pos - lo;
  if (lo + 1 < sorted.length) return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
  return sorted[lo];
}

export const fmt = (v: number, digits = 0) =>
  v.toLocaleString('en-GB', {minimumFractionDigits: digits, maximumFractionDigits: digits});

export type BuildBuyParams = {
  requests: number;
  apiOutPerM: number;
  gpuPerHour: number;
  platformFte: number;
  extraErrorPoints: number;
};

const BB = {
  inTokens: 1500,
  outTokens: 300,
  apiInPerM: 2.0,
  hoursPerMonth: 730,
  replicaTokensPerS: 1200,
  peakToMean: 4,
  minReplicas: 2,
  fteMonthly: 15000,
  apiErrorRate: 0.06,
  costPerError: 0.1,
};

export function replicasNeeded(requests: number): number {
  const meanRps = requests / (30 * 86400);
  const peakTokens = meanRps * BB.peakToMean * BB.outTokens;
  return Math.max(BB.minReplicas, Math.ceil(peakTokens / BB.replicaTokensPerS));
}

export function buildBuyMonthly(p: BuildBuyParams, requests: number, withErrors = true) {
  const perRequest = (BB.inTokens * BB.apiInPerM + BB.outTokens * p.apiOutPerM) / 1e6;
  const replicas = replicasNeeded(requests);
  const hostFixed = replicas * p.gpuPerHour * BB.hoursPerMonth + p.platformFte * BB.fteMonthly;
  const openErrorRate = BB.apiErrorRate + p.extraErrorPoints / 100;
  const buy = requests * perRequest + (withErrors ? requests * BB.apiErrorRate * BB.costPerError : 0);
  const host = hostFixed + (withErrors ? requests * openErrorRate * BB.costPerError : 0);
  return {buy, host, replicas};
}

export function breakEven(p: BuildBuyParams, withErrors = true): number | null {
  let lo = 1000;
  let hi = 500_000_000;
  for (let i = 0; i < 80; i += 1) {
    const mid = (lo + hi) / 2;
    const {buy, host} = buildBuyMonthly(p, mid, withErrors);
    if (buy > host) hi = mid;
    else lo = mid;
  }
  return hi > 499_999_000 ? null : hi;
}

export type RoiParams = {
  adoption: number;
  minutesSaved: number;
  rework: number;
  realisation: number;
  buildCost: number;
  steps: number;
};

const ROI = {users: 120, tasksPerUser: 400, loadedRate: 30, fixedMonthly: 6250, months: 24, annualDiscount: 0.1};

export function costPerTask(steps: number): number {
  const base = 1154;
  const grow = 172;
  const outTokens = 55;
  const pIn = 2.0;
  const pOut = 8.0;
  let paidIn = 0;
  for (let i = 0; i < steps; i += 1) {
    if (i === 0) paidIn += base * 1.25;
    else paidIn += (base + (i - 1) * grow) * 0.1 + grow * 1.25;
  }
  const cost = (paidIn * pIn + steps * outTokens * pOut) / 1e6;
  return Math.round(cost * 1e5) / 1e5;
}

export function roiSummary(p: RoiParams) {
  const tasks = ROI.users * ROI.tasksPerUser * p.adoption;
  const hours = (tasks * (1 - p.rework) * p.minutesSaved) / 60;
  const benefit = hours * ROI.loadedRate * p.realisation;
  const cost = tasks * costPerTask(p.steps) + ROI.fixedMonthly;
  const net = benefit - cost;
  const monthlyRate = (1 + ROI.annualDiscount) ** (1 / 12) - 1;
  let npv = -p.buildCost;
  for (let m = 1; m <= ROI.months; m += 1) npv += net / (1 + monthlyRate) ** m;
  const payback = net > 0 ? p.buildCost / net : Infinity;
  const roi = (net * ROI.months - p.buildCost) / p.buildCost;
  const cumulative = Array.from({length: ROI.months + 1}, (_, m) => -p.buildCost + net * m);
  return {benefit, cost, net, payback, npv, roi, cumulative};
}

export type EstimateTask = {name: string; lo: number; mode: number; hi: number; dataDependent: boolean};

export const ESTIMATE_TASKS: EstimateTask[] = [
  {name: 'data audit', lo: 2, mode: 3, hi: 8, dataDependent: false},
  {name: 'label the eval set', lo: 4, mode: 6, hi: 15, dataDependent: true},
  {name: 'baseline: prompt and retrieval', lo: 3, mode: 5, hi: 12, dataDependent: true},
  {name: 'eval harness', lo: 3, mode: 4, hi: 7, dataDependent: false},
  {name: 'iterate to the quality target', lo: 5, mode: 10, hi: 30, dataDependent: true},
  {name: 'integration and API', lo: 4, mode: 6, hi: 12, dataDependent: false},
  {name: 'safety and red-team review', lo: 2, mode: 3, hi: 8, dataDependent: false},
  {name: 'rollout and monitoring', lo: 2, mode: 3, hi: 6, dataDependent: false},
];

export function simulateProject(badProbability: number, slowdown: number, shared: boolean, trials = 4000, seed = 2026): number[] {
  const rnd = mulberry32(seed);
  const totals: number[] = [];
  for (let t = 0; t < trials; t += 1) {
    const sharedDraw = rnd();
    let total = 0;
    for (const task of ESTIMATE_TASKS) {
      let days = triangular(rnd(), task.lo, task.mode, task.hi);
      const own = rnd();
      const bad = (shared ? sharedDraw : own) < badProbability;
      if (task.dataDependent && bad) days *= slowdown;
      total += days;
    }
    totals.push(total);
  }
  return totals.sort((a, b) => a - b);
}

export const MATRIX_CRITERIA = ['answer quality', 'time to ship', 'run cost at volume', 'privacy and residency', 'on-call load', 'reversibility'];
export const MATRIX_DEFAULT_WEIGHTS = [5, 4, 3, 4, 3, 2];
export const MATRIX_OPTIONS: {name: string; scores: number[]}[] = [
  {name: 'hosted API, prompt only', scores: [3, 5, 3, 2, 5, 4]},
  {name: 'hosted API plus retrieval', scores: [5, 4, 2, 2, 3, 4]},
  {name: 'small fine-tuned model, self-hosted', scores: [3, 1, 5, 5, 2, 3]},
];

export function matrixTotals(weights: number[]): number[] {
  const sum = weights.reduce((a, w) => a + w, 0);
  return MATRIX_OPTIONS.map((o) => o.scores.reduce((a, s, i) => a + weights[i] * s, 0) / sum);
}

export function matrixWinShare(weights: number[], draws = 2000, seed = 99): number[] {
  const rnd = mulberry32(seed);
  const wins = MATRIX_OPTIONS.map(() => 0);
  for (let d = 0; d < draws; d += 1) {
    const jittered = weights.map((w) => w * (0.5 + rnd()));
    const totals = matrixTotals(jittered);
    let best = 0;
    for (let i = 1; i < totals.length; i += 1) if (totals[i] > totals[best]) best = i;
    wins[best] += 1;
  }
  return wins.map((w) => w / draws);
}

export const BURN_RULES = [
  {rate: 14.4, windowHours: 1, name: 'page 14.4x over 1 h'},
  {rate: 6, windowHours: 6, name: 'page 6x over 6 h'},
  {rate: 1, windowHours: 72, name: 'ticket 1x over 72 h'},
];

export function budgetEvent(slo: number, windowDays: number, badFraction: number, hours: number) {
  const budget = 1 - slo;
  const burn = badFraction / budget;
  const used = (badFraction * hours) / (windowDays * 24) / budget;
  const fired = BURN_RULES.filter((r) => (burn * Math.min(hours, r.windowHours)) / r.windowHours >= r.rate - 1e-9).map((r) => r.name);
  return {burn, used, fired};
}
