export type Aggregator = 'sum' | 'mean' | 'gcn' | 'max';

export const TINY_EDGES: [number, number][] = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]];
export const TINY_START = [1, 2, 3, 4, 5];

export function adjacency(edges: [number, number][], n: number) {
  const a = Array.from({length: n}, () => new Array<number>(n).fill(0));
  for (const [i, j] of edges) {
    a[i][j] = 1;
    a[j][i] = 1;
  }
  return a;
}

export function step(a: number[][], x: number[], mode: Aggregator): number[] {
  const n = x.length;
  const degree = a.map((row) => row.reduce((s, v) => s + v, 0));
  return x.map((own, i) => {
    const neighbours = a[i].map((v, j) => (v ? j : -1)).filter((j) => j >= 0);
    if (mode === 'sum') return neighbours.reduce((s, j) => s + x[j], 0);
    if (mode === 'mean') return neighbours.length ? neighbours.reduce((s, j) => s + x[j], 0) / neighbours.length : 0;
    if (mode === 'max') return neighbours.length ? Math.max(...neighbours.map((j) => x[j])) : 0;
    const dI = degree[i] + 1;
    let total = own / dI;
    for (const j of neighbours) total += x[j] / Math.sqrt(dI * (degree[j] + 1));
    return total;
  });
}

export function propagate(a: number[][], x: number[], mode: Aggregator, layers: number) {
  const history = [x];
  let current = x;
  for (let k = 0; k < layers; k += 1) {
    current = step(a, current, mode);
    history.push(current);
  }
  return history;
}

export function gatWeights(scores: number[], slope: number): number[] {
  const leaky = scores.map((v) => (v > 0 ? v : slope * v));
  const top = Math.max(...leaky);
  const exps = leaky.map((v) => Math.exp(v - top));
  const total = exps.reduce((s, v) => s + v, 0);
  return exps.map((v) => v / total);
}

export const NODE2_GCN_WEIGHTS = [1 / 4, 1 / Math.sqrt(12), 1 / Math.sqrt(12), 1 / Math.sqrt(12)];

export function samplingNodes(batch: number, fanouts: number[], meanDegree: number, totalNodes: number, sampled: boolean) {
  let width = batch;
  let total = batch;
  for (const f of fanouts) {
    width *= sampled ? Math.min(f, meanDegree) : meanDegree;
    total += width;
  }
  return Math.min(total, totalNodes);
}
