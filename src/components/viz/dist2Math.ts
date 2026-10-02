export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function normals(rnd: () => number, count: number): number[] {
  const out: number[] = [];
  while (out.length < count) {
    const u1 = Math.max(rnd(), 1e-12);
    const u2 = rnd();
    const r = Math.sqrt(-2 * Math.log(u1));
    out.push(r * Math.cos(2 * Math.PI * u2));
    out.push(r * Math.sin(2 * Math.PI * u2));
  }
  return out.slice(0, count);
}

export type LogisticData = {x: number[][]; y: number[]; n: number; d: number};

export function makeLogistic(n: number, d: number, seed: number): LogisticData {
  const rnd = mulberry32(seed);
  const flat = normals(rnd, n * d);
  const wTrue = normals(rnd, d);
  const eps = normals(rnd, n);
  const x: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const row = flat.slice(i * d, (i + 1) * d);
    let z = 0;
    for (let j = 0; j < d; j += 1) z += row[j] * wTrue[j];
    x.push(row);
    y.push(z + 0.5 * eps[i] > 0 ? 1 : 0);
  }
  return {x, y, n, d};
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function logisticLoss(data: LogisticData, w: number[]): number {
  let total = 0;
  for (let i = 0; i < data.n; i += 1) {
    let z = 0;
    for (let j = 0; j < data.d; j += 1) z += data.x[i][j] * w[j];
    total += Math.max(z, 0) - z * data.y[i] + Math.log1p(Math.exp(-Math.abs(z)));
  }
  return total / data.n;
}

export function logisticGrad(data: LogisticData, w: number[], lo: number, hi: number): number[] {
  const g = new Array<number>(data.d).fill(0);
  for (let i = lo; i < hi; i += 1) {
    let z = 0;
    for (let j = 0; j < data.d; j += 1) z += data.x[i][j] * w[j];
    const r = sigmoid(z) - data.y[i];
    for (let j = 0; j < data.d; j += 1) g[j] += r * data.x[i][j];
  }
  const m = hi - lo;
  return g.map((v) => v / m);
}
