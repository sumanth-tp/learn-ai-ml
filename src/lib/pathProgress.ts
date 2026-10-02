import {useSyncExternalStore} from 'react';

const OUTCOMES_KEY = 'learn-ai-ml:path-outcomes';

export type OutcomeMap = Record<string, number>;

const EMPTY: OutcomeMap = {};
const listeners = new Set<() => void>();

let cache: OutcomeMap = EMPTY;
let cacheRaw: string | null = null;

function subscribe(listener: () => void) {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === OUTCOMES_KEY || event.key === null) {
      listener();
    }
  };
  window.addEventListener('storage', onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener('storage', onStorage);
  };
}

function getSnapshot(): OutcomeMap {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(OUTCOMES_KEY);
  } catch {
    raw = null;
  }
  if (raw === cacheRaw) {
    return cache;
  }
  cacheRaw = raw;
  try {
    const parsed = raw ? (JSON.parse(raw) as OutcomeMap) : EMPTY;
    cache = parsed && typeof parsed === 'object' ? parsed : EMPTY;
  } catch {
    cache = EMPTY;
  }
  return cache;
}

function getServerSnapshot(): OutcomeMap {
  return EMPTY;
}

export function outcomeKey(stageId: string, index: number) {
  return `${stageId}:${index}`;
}

export function useOutcomes(): OutcomeMap {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

export function setOutcome(key: string, value: boolean) {
  const next = {...getSnapshot()};
  if (value) {
    next[key] = Date.now();
  } else {
    delete next[key];
  }
  try {
    window.localStorage.setItem(OUTCOMES_KEY, JSON.stringify(next));
  } catch {
  }
  listeners.forEach((listener) => listener());
}
