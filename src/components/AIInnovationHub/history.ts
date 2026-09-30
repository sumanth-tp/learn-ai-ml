import type {Discovery, RawDiscovery, SourceReport} from './types';

export const HISTORY_KEY = 'learn-ai-ml:discoveries:history:v1';
const PREVIOUS_CACHE_KEY = 'learn-ai-ml:discoveries:v7';
export const HISTORY_LIMIT = 3;

export type UpdateBatch = {createdAt: number; items: Discovery[]; reports: SourceReport[]};
export type UpdateHistory = {batches: UpdateBatch[]; selectedAt: number | null; seen: string[]};
export const emptyHistory = (): UpdateHistory => ({batches: [], selectedAt: null, seen: []});

// Keep identities after a batch leaves history so new items stay prioritised.
export function discoveryKeys(item: Pick<RawDiscovery, 'id' | 'url'>): string[] {
  let url = item.url;
  try {
    const parsed = new URL(url);
    parsed.hash = '';
    [...parsed.searchParams.keys()].forEach((key) => {
      if (key.startsWith('utm_')) parsed.searchParams.delete(key);
    });
    parsed.searchParams.sort();
    url = parsed.toString().replace(/\/$/, '');
  } catch { /* The source id still identifies records without a valid URL. */ }
  return [`id:${item.id}`, `url:${url}`];
}

const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every((entry) => typeof entry === 'string');
function validBatch(value: unknown): value is UpdateBatch {
  if (!value || typeof value !== 'object') return false;
  const batch = value as UpdateBatch;
  return Number.isFinite(batch.createdAt) && batch.createdAt > 0
    && Array.isArray(batch.items) && batch.items.length > 0 && batch.items.every((item) => item
      && (['id', 'title', 'url', 'source', 'publishedAt', 'description', 'simpleExplanation', 'whyImportant'] as const).every((key) => typeof item[key] === 'string')
      && ['papers', 'models', 'tools', 'videos'].includes(item.category)
      && typeof item.aiEdited === 'boolean'
      && [item.useCases, item.prerequisites, item.projectIdeas].every(strings)
      && [item.authors, item.metrics].every((value) => value === undefined || strings(value))
      && (item.image === undefined || typeof item.image === 'string')
      && (item.sharedAt === undefined || typeof item.sharedAt === 'string'))
    && Array.isArray(batch.reports) && batch.reports.every((report) => report
      && typeof report.source === 'string' && ['ok', 'partial', 'skipped', 'error'].includes(report.status)
      && Number.isFinite(report.count) && (report.message === undefined || typeof report.message === 'string'));
}

export function readHistory(storage: Storage): UpdateHistory {
  let saved: Partial<UpdateHistory> | null = null;
  try { saved = JSON.parse(storage.getItem(HISTORY_KEY) || 'null'); } catch { /* Try the previous cache. */ }
  let batches = Array.isArray(saved?.batches) ? saved.batches.filter(validBatch) : [];
  if (batches.length === 0) {
    try {
      const previous: unknown = JSON.parse(storage.getItem(PREVIOUS_CACHE_KEY) || 'null');
      if (validBatch(previous)) batches = [previous];
    } catch { /* Ignore invalid older cache data. */ }
  }
  batches = [...new Map(batches.map((batch) => [batch.createdAt, batch])).values()]
    .sort((a, b) => b.createdAt - a.createdAt).slice(0, HISTORY_LIMIT);
  return {
    batches,
    selectedAt: batches.some((batch) => batch.createdAt === saved?.selectedAt) ? saved!.selectedAt! : batches[0]?.createdAt ?? null,
    seen: [...new Set([...(strings(saved?.seen) ? saved.seen : []), ...batches.flatMap((batch) => batch.items.flatMap(discoveryKeys))])],
  };
}

export function addBatch(history: UpdateHistory, batch: UpdateBatch): UpdateHistory {
  return {
    batches: [batch, ...history.batches].slice(0, HISTORY_LIMIT),
    selectedAt: batch.createdAt,
    seen: [...new Set([...history.seen, ...batch.items.flatMap(discoveryKeys)])],
  };
}
