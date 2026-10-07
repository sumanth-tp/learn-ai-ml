export type FloatFormat = {
  name: string;
  expBits: number;
  manBits: number;
  bias: number;
  maxFinite: number;
  overflow: 'inf' | 'saturate';
};

export const FORMATS: FloatFormat[] = [
  {name: 'fp32', expBits: 8, manBits: 23, bias: 127, maxFinite: 3.4028234663852886e38, overflow: 'inf'},
  {name: 'bf16', expBits: 8, manBits: 7, bias: 127, maxFinite: 3.3895313892515355e38, overflow: 'inf'},
  {name: 'fp16', expBits: 5, manBits: 10, bias: 15, maxFinite: 65504, overflow: 'inf'},
  {name: 'fp8 e4m3', expBits: 4, manBits: 3, bias: 7, maxFinite: 448, overflow: 'saturate'},
  {name: 'fp8 e5m2', expBits: 5, manBits: 2, bias: 15, maxFinite: 57344, overflow: 'inf'},
];

export function smallestNormal(f: FloatFormat): number {
  return Math.pow(2, 1 - f.bias);
}

export function smallestSubnormal(f: FloatFormat): number {
  return Math.pow(2, 1 - f.bias - f.manBits);
}

function roundHalfEven(n: number): number {
  const fl = Math.floor(n);
  const diff = n - fl;
  if (diff < 0.5) return fl;
  if (diff > 0.5) return fl + 1;
  return fl % 2 === 0 ? fl : fl + 1;
}

export function roundTo(x: number, f: FloatFormat): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  let e = Math.floor(Math.log2(a));
  if (Math.pow(2, e) > a) e -= 1;
  if (Math.pow(2, e + 1) <= a) e += 1;
  const emin = 1 - f.bias;
  const exponent = Math.max(e, emin);
  const quantum = Math.pow(2, exponent - f.manBits);
  const v = roundHalfEven(a / quantum) * quantum;
  if (v > f.maxFinite) return f.overflow === 'inf' ? sign * Infinity : sign * f.maxFinite;
  return sign * v;
}

export type Status = 'exact-ish' | 'subnormal' | 'underflow to zero' | 'overflow';

export function classify(x: number, stored: number, f: FloatFormat): Status {
  if (x !== 0 && stored === 0) return 'underflow to zero';
  if (!Number.isFinite(stored) && Number.isFinite(x)) return 'overflow';
  if (Math.abs(x) >= f.maxFinite && f.overflow === 'saturate' && Math.abs(stored) === f.maxFinite && Math.abs(x) > f.maxFinite) {
    return 'overflow';
  }
  if (Math.abs(stored) < smallestNormal(f)) return 'subnormal';
  return 'exact-ish';
}

export function formatNumber(v: number): string {
  if (v === 0) return '0';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(4);
  return Number(v.toPrecision(7)).toString();
}
