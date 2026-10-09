export type LatenessRow = {days: number; payments: number; netMinor: number};

export const LATENESS: LatenessRow[] = [
  {days: 0, payments: 57736, netMinor: 302413333},
  {days: 1, payments: 721, netMinor: 3770942},
  {days: 2, payments: 726, netMinor: 3952999},
  {days: 3, payments: 700, netMinor: 3617520},
  {days: 4, payments: 149, netMinor: 769295},
  {days: 5, payments: 158, netMinor: 898591},
  {days: 6, payments: 168, netMinor: 943006},
];

export type WindowSummary = {
  heldPayments: number;
  heldNet: number;
  heldShare: number;
  restatedPayments: number;
  restatedNet: number;
  restatedShare: number;
  totalNet: number;
};

export function windowSummary(windowDays: number): WindowSummary {
  const totalNet = LATENESS.reduce((a, r) => a + r.netMinor, 0);
  const held = LATENESS.filter((r) => r.days > windowDays);
  const restated = LATENESS.filter((r) => r.days >= 1 && r.days <= windowDays);
  const heldNet = held.reduce((a, r) => a + r.netMinor, 0);
  const restatedNet = restated.reduce((a, r) => a + r.netMinor, 0);
  return {
    heldPayments: held.reduce((a, r) => a + r.payments, 0),
    heldNet,
    heldShare: heldNet / totalNet,
    restatedPayments: restated.reduce((a, r) => a + r.payments, 0),
    restatedNet,
    restatedShare: restatedNet / totalNet,
    totalNet,
  };
}
