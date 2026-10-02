export type Confusion = {tp: number; fp: number; fn: number; tn: number};

export type ConfusionMetrics = {
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  specificity: number | null;
};

const ratio = (num: number, den: number): number | null => (den === 0 ? null : num / den);

export function confusionMetrics({tp, fp, fn, tn}: Confusion): ConfusionMetrics {
  const precision = ratio(tp, tp + fp);
  const recall = ratio(tp, tp + fn);
  const f1 =
    precision === null || recall === null || precision + recall === 0
      ? null
      : (2 * precision * recall) / (precision + recall);
  return {
    accuracy: ratio(tp + tn, tp + fp + fn + tn),
    precision,
    recall,
    f1,
    specificity: ratio(tn, tn + fp),
  };
}

export function normalSurvival(x: number): number {
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * z);
  const poly =
    t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erfc = poly * Math.exp(-z * z);
  const upper = 0.5 * erfc;
  return x >= 0 ? upper : 1 - upper;
}

export const normalCdf = (x: number): number => 1 - normalSurvival(x);

export type Binormal = {
  tpr: number;
  fpr: number;
  tp: number;
  fn: number;
  fp: number;
  tn: number;
  precision: number;
  auc: number;
};

export function binormalAt(separation: number, prevalence: number, threshold: number, n: number): Binormal {
  const tpr = normalSurvival(threshold - separation);
  const fpr = normalSurvival(threshold);
  const tp = tpr * prevalence * n;
  const fn = (1 - tpr) * prevalence * n;
  const fp = fpr * (1 - prevalence) * n;
  const tn = (1 - fpr) * (1 - prevalence) * n;
  return {
    tpr,
    fpr,
    tp,
    fn,
    fp,
    tn,
    precision: tp + fp === 0 ? 1 : tp / (tp + fp),
    auc: normalCdf(separation / Math.SQRT2),
  };
}

export function binormalCurves(separation: number, prevalence: number, points = 400) {
  const lo = -6;
  const hi = separation + 6;
  const roc: {fpr: number; tpr: number}[] = [];
  const pr: {recall: number; precision: number}[] = [];
  for (let i = 0; i <= points; i += 1) {
    const t = hi - ((hi - lo) * i) / points;
    const tpr = normalSurvival(t - separation);
    const fpr = normalSurvival(t);
    roc.push({fpr, tpr});
    const denom = prevalence * tpr + (1 - prevalence) * fpr;
    pr.push({recall: tpr, precision: denom === 0 ? 1 : (prevalence * tpr) / denom});
  }
  return {roc, pr};
}

export function areaUnderPr(separation: number, prevalence: number): number {
  const lo = -6;
  const hi = separation + 6;
  const steps = 4000;
  let area = 0;
  let prev: {recall: number; precision: number} | null = null;
  for (let i = 0; i <= steps; i += 1) {
    const t = hi - ((hi - lo) * i) / steps;
    const tpr = normalSurvival(t - separation);
    const fpr = normalSurvival(t);
    const denom = prevalence * tpr + (1 - prevalence) * fpr;
    const cur = {recall: tpr, precision: denom === 0 ? 1 : (prevalence * tpr) / denom};
    if (prev) area += ((cur.precision + prev.precision) / 2) * (cur.recall - prev.recall);
    prev = cur;
  }
  return area;
}

const GOLDEN = 0.6180339887498949;

const logit = (p: number) => Math.log(p / (1 - p));
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const clip = (p: number) => Math.min(1 - 1e-6, Math.max(1e-6, p));

export type Scores = {q: number[]; y: number[]};

export function makeScores(n: number, slope: number, shift: number): Scores {
  const q: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const score = (i + 0.5) / n;
    const truth = sigmoid(slope * logit(score) + shift);
    const u = (((i + 1) * GOLDEN) % 1 + 1) % 1;
    q.push(score);
    y.push(u < truth ? 1 : 0);
  }
  return {q, y};
}

export type Calibrator = (q: number) => number;

export function fitPlatt(q: number[], y: number[], steps = 50): Calibrator {
  const x = q.map((v) => logit(clip(v)));
  let w = 1;
  let b = 0;
  for (let it = 0; it < steps; it += 1) {
    let gw = 0;
    let gb = 0;
    let hww = 1e-9;
    let hwb = 0;
    let hbb = 1e-9;
    for (let i = 0; i < x.length; i += 1) {
      const p = sigmoid(w * x[i] + b);
      const s = p * (1 - p);
      gw += (p - y[i]) * x[i];
      gb += p - y[i];
      hww += s * x[i] * x[i];
      hwb += s * x[i];
      hbb += s;
    }
    const det = hww * hbb - hwb * hwb;
    if (Math.abs(det) < 1e-18) break;
    w -= (hbb * gw - hwb * gb) / det;
    b -= (hww * gb - hwb * gw) / det;
  }
  return (v: number) => sigmoid(w * logit(clip(v)) + b);
}

export function fitIsotonic(q: number[], y: number[]): Calibrator {
  const order = q.map((_, i) => i).sort((a, b) => q[a] - q[b]);
  const blocks: {sum: number; count: number; lo: number; hi: number}[] = [];
  for (const i of order) {
    blocks.push({sum: y[i], count: 1, lo: q[i], hi: q[i]});
    while (blocks.length > 1) {
      const a = blocks[blocks.length - 2];
      const b = blocks[blocks.length - 1];
      if (a.sum / a.count <= b.sum / b.count) break;
      blocks.splice(blocks.length - 2, 2, {
        sum: a.sum + b.sum,
        count: a.count + b.count,
        lo: a.lo,
        hi: b.hi,
      });
    }
  }
  const knots: {x: number; y: number}[] = [];
  for (const blk of blocks) {
    const value = blk.sum / blk.count;
    knots.push({x: blk.lo, y: value});
    if (blk.hi !== blk.lo) knots.push({x: blk.hi, y: value});
  }
  return (v: number) => {
    if (v <= knots[0].x) return knots[0].y;
    const last = knots[knots.length - 1];
    if (v >= last.x) return last.y;
    let lo = 0;
    let hi = knots.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (knots[mid].x <= v) lo = mid;
      else hi = mid;
    }
    const a = knots[lo];
    const b = knots[hi];
    return b.x === a.x ? b.y : a.y + ((b.y - a.y) * (v - a.x)) / (b.x - a.x);
  };
}

export const brierScore = (p: number[], y: number[]): number =>
  p.reduce((acc, v, i) => acc + (v - y[i]) ** 2, 0) / p.length;

export type ReliabilityBin = {count: number; predicted: number; observed: number};

export function reliabilityBins(p: number[], y: number[], bins = 10): ReliabilityBin[] {
  const out = Array.from({length: bins}, () => ({count: 0, sumP: 0, sumY: 0}));
  p.forEach((v, i) => {
    const b = Math.min(bins - 1, Math.floor(v * bins));
    out[b].count += 1;
    out[b].sumP += v;
    out[b].sumY += y[i];
  });
  return out.map((b) => ({
    count: b.count,
    predicted: b.count ? b.sumP / b.count : 0,
    observed: b.count ? b.sumY / b.count : 0,
  }));
}

export function expectedCalibrationError(p: number[], y: number[], bins = 10): number {
  return reliabilityBins(p, y, bins).reduce(
    (acc, b) => acc + (b.count / p.length) * Math.abs(b.predicted - b.observed),
    0,
  );
}

export type CalibrationMethod = 'none' | 'platt' | 'isotonic';

export function calibrationRun(n: number, slope: number, shift: number, method: CalibrationMethod) {
  const {q, y} = makeScores(n, slope, shift);
  const fitIdx: number[] = [];
  const testIdx: number[] = [];
  for (let i = 0; i < n; i += 1) (i % 2 === 0 ? fitIdx : testIdx).push(i);
  const qFit = fitIdx.map((i) => q[i]);
  const yFit = fitIdx.map((i) => y[i]);
  const qTest = testIdx.map((i) => q[i]);
  const yTest = testIdx.map((i) => y[i]);
  const map: Calibrator =
    method === 'platt' ? fitPlatt(qFit, yFit) : method === 'isotonic' ? fitIsotonic(qFit, yFit) : (v) => v;
  const pTest = qTest.map(map);
  return {
    map,
    pTest,
    yTest,
    brier: brierScore(pTest, yTest),
    ece: expectedCalibrationError(pTest, yTest),
    bins: reliabilityBins(pTest, yTest),
  };
}
