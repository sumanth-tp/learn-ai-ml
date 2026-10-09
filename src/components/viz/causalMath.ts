export type ConfoundingInput = {
  shareEngaged: number;
  notifyEngaged: number;
  notifyOther: number;
  engagementEffect: number;
  trueEffect: number;
};

export function confoundingGap(input: ConfoundingInput) {
  const {shareEngaged: p, notifyEngaged: e1, notifyOther: e0, engagementEffect: gamma, trueEffect: tau} = input;
  const treatedShare = p * e1 + (1 - p) * e0;
  const engagedAmongTreated = treatedShare > 0 ? (p * e1) / treatedShare : 0;
  const engagedAmongUntreated = treatedShare < 1 ? (p * (1 - e1)) / (1 - treatedShare) : 0;
  const naive = tau + gamma * (engagedAmongTreated - engagedAmongUntreated);
  return {
    treatedShare,
    engagedAmongTreated,
    engagedAmongUntreated,
    naive,
    adjusted: tau,
    bias: naive - tau,
  };
}

export type IpwInput = ConfoundingInput & {trim: number};

export const IPW_USERS = 1000;
export const IPW_BASE_SPEND = 10;

export function ipwEstimate(input: IpwInput) {
  const {shareEngaged: p, notifyEngaged, notifyOther, engagementEffect, trueEffect, trim} = input;
  const strata = [
    {share: p, e: notifyEngaged, base: IPW_BASE_SPEND + engagementEffect},
    {share: 1 - p, e: notifyOther, base: IPW_BASE_SPEND},
  ];
  const clipped = (e: number) => Math.min(1 - trim, Math.max(trim, e));
  let treatedMass = 0;
  let treatedSum = 0;
  let treatedSq = 0;
  let controlMass = 0;
  let controlSum = 0;
  let controlSq = 0;
  let maxWeight = 0;
  let naiveTreated = 0;
  let naiveControl = 0;
  let countTreated = 0;
  let countControl = 0;
  for (const s of strata) {
    const wt = 1 / clipped(s.e);
    const wc = 1 / (1 - clipped(s.e));
    const nt = IPW_USERS * s.share * s.e;
    const nc = IPW_USERS * s.share * (1 - s.e);
    treatedMass += nt * wt;
    treatedSum += nt * wt * (s.base + trueEffect);
    treatedSq += nt * wt * wt;
    controlMass += nc * wc;
    controlSum += nc * wc * s.base;
    controlSq += nc * wc * wc;
    naiveTreated += nt * (s.base + trueEffect);
    naiveControl += nc * s.base;
    countTreated += nt;
    countControl += nc;
    if (nt > 0) maxWeight = Math.max(maxWeight, wt);
    if (nc > 0) maxWeight = Math.max(maxWeight, wc);
  }
  const ipwTreated = treatedSum / treatedMass;
  const ipwControl = controlSum / controlMass;
  return {
    naive: naiveTreated / countTreated - naiveControl / countControl,
    ipwTreated,
    ipwControl,
    ipwEffect: ipwTreated - ipwControl,
    essTreated: (treatedMass * treatedMass) / treatedSq,
    essControl: (controlMass * controlMass) / controlSq,
    countTreated,
    countControl,
    maxWeight,
  };
}

export type DidInput = {
  controlBefore: number;
  commonChange: number;
  groupGap: number;
  trueEffect: number;
  extraTrend: number;
};

export function didTable(input: DidInput) {
  const {controlBefore, commonChange, groupGap, trueEffect, extraTrend} = input;
  const controlAfter = controlBefore + commonChange;
  const treatedBefore = controlBefore + groupGap;
  const treatedAfter = treatedBefore + commonChange + extraTrend + trueEffect;
  return {
    controlBefore,
    controlAfter,
    treatedBefore,
    treatedAfter,
    beforeAfterTreated: treatedAfter - treatedBefore,
    afterOnly: treatedAfter - controlAfter,
    did: treatedAfter - treatedBefore - (controlAfter - controlBefore),
    bias: extraTrend,
  };
}

export type WaldInput = {
  takeUpWithNudge: number;
  takeUpWithout: number;
  trueEffect: number;
  directEffect: number;
};

export function waldEstimate(input: WaldInput) {
  const {takeUpWithNudge, takeUpWithout, trueEffect, directEffect} = input;
  const firstStage = takeUpWithNudge - takeUpWithout;
  const reducedForm = trueEffect * firstStage + directEffect;
  const wald = firstStage !== 0 ? reducedForm / firstStage : NaN;
  return {firstStage, reducedForm, wald, bias: wald - trueEffect, noiseMultiplier: firstStage !== 0 ? 1 / Math.abs(firstStage) : NaN};
}

export const UPLIFT_SEGMENTS = [
  {name: 'persuadables', share: 0.176, effect: 2.5},
  {name: 'older customers', share: 0.253, effect: 0.5},
  {name: 'indifferent', share: 0.446, effect: 0},
  {name: 'sleeping dogs', share: 0.125, effect: -1.361},
];

export function upliftProfit(cost: number, targeted: number) {
  const ordered = [...UPLIFT_SEGMENTS].sort((a, b) => b.effect - a.effect);
  let remaining = targeted;
  let profit = 0;
  for (const segment of ordered) {
    const take = Math.min(segment.share, remaining);
    profit += take * (segment.effect - cost);
    remaining -= take;
  }
  return profit;
}

export function upliftSummary(cost: number, targeted: number) {
  const averageEffect = UPLIFT_SEGMENTS.reduce((sum, s) => sum + s.share * s.effect, 0);
  const bestShare = UPLIFT_SEGMENTS.filter((s) => s.effect > cost).reduce((sum, s) => sum + s.share, 0);
  return {
    averageEffect,
    byModel: upliftProfit(cost, targeted),
    byLottery: targeted * (averageEffect - cost),
    everyone: averageEffect - cost,
    bestShare,
    bestProfit: upliftProfit(cost, bestShare),
  };
}
