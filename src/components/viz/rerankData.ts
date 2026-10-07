export type RerankData = {nrel: number[]; rel: string[]; general: number[][]; tuned: number[][]; examples: {q: number; text: string; titles: string[]}[]};

export const RERANK: RerankData = {nrel: [1], rel: ['1'.padEnd(50, '0')], general: [[0.5]], tuned: [[0.5]], examples: [{q: 0, text: 'placeholder', titles: ['a']}]};
