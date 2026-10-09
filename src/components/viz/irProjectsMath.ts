export type FrontierPoint = {name: string; ndcg: number; p50: number; p95: number};

export const FRONTIER: FrontierPoint[] = [
  {name: 'BM25', ndcg: 0.561, p50: 0.2, p95: 0.3},
  {name: 'BM25 + rerank 20', ndcg: 0.603, p50: 40.5, p95: 70.1},
  {name: 'dense', ndcg: 0.657, p50: 3.6, p95: 4.0},
  {name: 'hybrid', ndcg: 0.705, p50: 3.8, p95: 4.4},
  {name: 'hybrid + rerank 10', ndcg: 0.707, p50: 19.3, p95: 29.1},
  {name: 'hybrid + rerank 20', ndcg: 0.707, p50: 36.1, p95: 56.9},
  {name: 'hybrid + rerank 30', ndcg: 0.709, p50: 52.1, p95: 103.6},
  {name: 'hybrid + rerank 20 (replace)', ndcg: 0.634, p50: 29.6, p95: 48.0},
];

export type Pick = {best: FrontierPoint | null; chosen: FrontierPoint | null; feasible: FrontierPoint[]};

export function pickConfiguration(points: FrontierPoint[], budgetMs: number, noise: number): Pick {
  const feasible = points.filter((p) => p.p95 <= budgetMs);
  if (feasible.length === 0) return {best: null, chosen: null, feasible};
  const best = feasible.reduce((a, b) => (b.ndcg > a.ndcg ? b : a));
  const close = feasible.filter((p) => best.ndcg - p.ndcg <= noise + 1e-9);
  const chosen = close.reduce((a, b) => (b.p95 < a.p95 ? b : a));
  return {best, chosen, feasible};
}

export function coresNeeded(p50Ms: number, requestsPerSecond: number, utilisation = 0.6): number {
  return Math.max(1, Math.ceil((requestsPerSecond * p50Ms) / 1000 / utilisation));
}

export function postFilterFetch(wanted: number, visibleShare: number, safety = 1): number {
  return Math.ceil((wanted * safety) / Math.max(visibleShare, 1e-6));
}

export function expectedSurvivors(fetched: number, visibleShare: number): number {
  return fetched * visibleShare;
}

export function damerauLevenshtein(a: string, b: string): number {
  const d: number[][] = Array.from({length: a.length + 1}, (_, i) => [i, ...new Array(b.length).fill(0)]);
  for (let j = 0; j <= b.length; j += 1) d[0][j] = j;
  for (let i = 1; i <= a.length; i += 1) {
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
    }
  }
  return d[a.length][b.length];
}
