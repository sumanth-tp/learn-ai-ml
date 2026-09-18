/**
 * Chart tokens.
 *
 * Series hues are the validated categorical slots from the data-viz palette
 * (CVD ΔE ≥ 8 on adjacent pairs in both modes). Dark steps are *selected* for
 * the dark surface, not an automatic flip of the light ones. Charts never use
 * more than three series at once, which is the all-pairs-safe cap.
 */
export const SERIES = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#eda100'],
  dark: ['#3987e5', '#d95926', '#199e70', '#9085e9', '#c98500'],
} as const;

export const SERIES_NAMES = ['blue', 'orange', 'aqua', 'violet', 'yellow'] as const;

/** Sequential ramp for magnitude — one hue, light → dark. */
export const SEQUENTIAL = {
  light: ['#eef4fc', '#cfe0f6', '#a9c8ee', '#7aa9e3', '#4a86d8', '#2a78d6', '#1c5aa3'],
  dark: ['#16233a', '#1d3355', '#254474', '#2d5694', '#3568b4', '#3987e5', '#6aa6ee'],
} as const;

/** Diverging pair with a neutral midpoint — for polarity (advantage, error). */
export const DIVERGING = {
  light: {negative: '#e34948', mid: '#9aa0a6', positive: '#1baf7a'},
  dark: {negative: '#e66767', mid: '#848c99', positive: '#199e70'},
} as const;

export function seriesColor(index: number, dark: boolean): string {
  const ramp = dark ? SERIES.dark : SERIES.light;
  return ramp[index % ramp.length];
}

export function sequentialColor(t: number, dark: boolean): string {
  const ramp = dark ? SEQUENTIAL.dark : SEQUENTIAL.light;
  const clamped = Math.max(0, Math.min(1, t));
  return ramp[Math.round(clamped * (ramp.length - 1))];
}
