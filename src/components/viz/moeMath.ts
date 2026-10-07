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

export type RoutingResult = {
  counts: number[];
  capacity: number;
  balance: number;
  busiest: number;
  dropped: number;
  used: number;
};

export function simulateRouting(nExperts: number, k: number, tokens: number, cf: number, skew: number, seed = 7): RoutingResult {
  const draw = mulberry32(seed);
  const gauss = () => {
    const u1 = Math.max(draw(), 1e-12);
    const u2 = draw();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  const bias: number[] = [];
  for (let e = 0; e < nExperts; e++) bias.push(gauss());
  const counts = new Array<number>(nExperts).fill(0);
  const probSum = new Array<number>(nExperts).fill(0);
  for (let t = 0; t < tokens; t++) {
    const logits: number[] = [];
    for (let e = 0; e < nExperts; e++) logits.push(gauss() + skew * bias[e]);
    const max = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - max));
    const total = exps.reduce((a, b) => a + b, 0);
    for (let e = 0; e < nExperts; e++) probSum[e] += exps[e] / total;
    const order = logits.map((l, i) => i).sort((a, b) => logits[b] - logits[a]);
    for (let j = 0; j < k; j++) counts[order[j]] += 1;
  }
  const slots = tokens * k;
  let balance = 0;
  for (let e = 0; e < nExperts; e++) balance += (counts[e] / slots) * (probSum[e] / tokens);
  balance *= nExperts;
  const capacity = Math.ceil((cf * slots) / nExperts - 1e-9);
  const kept = counts.reduce((a, c) => a + Math.min(c, capacity), 0);
  return {
    counts,
    capacity,
    balance,
    busiest: Math.max(...counts) / (slots / nExperts),
    dropped: 1 - kept / slots,
    used: kept / (capacity * nExperts),
  };
}
