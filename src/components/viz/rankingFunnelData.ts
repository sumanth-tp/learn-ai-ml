export const FUNNEL_CATALOGUE = 8000;
export const FUNNEL_RELEVANT = 80;
export const FUNNEL_USERS = 300;

export type FunnelRow = [k1: number, k2: number, recallK1: number, recallK2: number, precisionAt10: number, ndcgAt10: number];

export const FUNNEL_GRID: FunnelRow[] = [[100, 50, 0.072, 0.045, 0.227, 0.24], [100, 100, 0.072, 0.072, 0.238, 0.248], [300, 50, 0.156, 0.072, 0.307, 0.309], [300, 100, 0.156, 0.095, 0.317, 0.315], [300, 200, 0.156, 0.118, 0.321, 0.319], [1000, 50, 0.335, 0.065, 0.289, 0.295], [1000, 100, 0.335, 0.114, 0.33, 0.327], [1000, 200, 0.335, 0.181, 0.345, 0.337], [2000, 50, 0.49, 0.065, 0.288, 0.294], [2000, 100, 0.49, 0.11, 0.326, 0.324], [2000, 200, 0.49, 0.182, 0.351, 0.344]];
