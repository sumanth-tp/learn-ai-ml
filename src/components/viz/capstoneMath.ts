import type {FusionItem} from './fusionData';

export function summarise(spans: [number, number][], docLength: number) {
  const lengths = spans.map(([a, b]) => b - a);
  const total = lengths.reduce((a, b) => a + b, 0);
  let shared = 0;
  for (let i = 1; i < spans.length; i += 1) {
    shared += Math.max(0, spans[i - 1][1] - spans[i][0]);
  }
  return {
    count: spans.length,
    average: spans.length ? Math.round((total / spans.length) * 10) / 10 : 0,
    shared,
    stored: total,
    inflation: docLength ? total / docLength : 0,
  };
}

export type Fused = {id: string; text: string; score: number; semanticRank: number | null; keywordRank: number | null};

export function fuse(semantic: FusionItem[], keyword: FusionItem[], semanticWeight: number, k: number): Fused[] {
  const keywordWeight = 1 - semanticWeight;
  const table = new Map<string, Fused>();
  const add = (list: FusionItem[], weight: number, field: 'semanticRank' | 'keywordRank') => {
    list.forEach((item, index) => {
      const entry = table.get(item.id) ?? {id: item.id, text: item.text, score: 0, semanticRank: null, keywordRank: null};
      entry.score += weight / (k + index + 1);
      entry[field] = index + 1;
      table.set(item.id, entry);
    });
  };
  add(semantic, semanticWeight, 'semanticRank');
  add(keyword, keywordWeight, 'keywordRank');
  return [...table.values()].sort((a, b) => b.score - a.score);
}
