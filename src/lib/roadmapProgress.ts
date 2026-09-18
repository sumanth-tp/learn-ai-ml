/**
 * Tick boxes for the LLM engineering roadmap.
 *
 * Same shape as `src/lib/progress.ts`: two localStorage maps exposed through
 * `useSyncExternalStore` so the table, the phase headers and the summary panel
 * all re-render together. Nothing leaves the device, and every access is
 * try/caught because storage can be blocked or full.
 */
import {useCallback, useSyncExternalStore} from 'react';

const UNIT_KEY = 'learn-ai-ml:roadmap:units';
const LINK_KEY = 'learn-ai-ml:roadmap:links';

export type DoneMap = Record<string, number>;

const EMPTY: DoneMap = {};

const listeners = new Set<() => void>();

type Cache = {raw: string | null; value: DoneMap};

const caches: Record<string, Cache> = {
  [UNIT_KEY]: {raw: null, value: EMPTY},
  [LINK_KEY]: {raw: null, value: EMPTY},
};

function emit() {
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === UNIT_KEY || event.key === LINK_KEY || event.key === null) {
      listener();
    }
  };
  window.addEventListener('storage', onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener('storage', onStorage);
  };
}

function read(key: string): DoneMap {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(key);
  } catch {
    return EMPTY;
  }
  const cache = caches[key];
  if (raw === cache.raw) {
    return cache.value;
  }
  cache.raw = raw;
  try {
    const parsed = raw ? (JSON.parse(raw) as DoneMap) : EMPTY;
    cache.value = parsed && typeof parsed === 'object' ? parsed : EMPTY;
  } catch {
    cache.value = EMPTY;
  }
  return cache.value;
}

function write(key: string, value: DoneMap) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable — progress simply does not persist */
  }
  emit();
}

function serverSnapshot(): DoneMap {
  return EMPTY;
}

function useMap(key: string): DoneMap {
  return useSyncExternalStore(subscribe, () => read(key), serverSnapshot);
}

function toggleIn(key: string, id: string, next: boolean) {
  const current = {...read(key)};
  if (next) {
    current[id] = Date.now();
  } else {
    delete current[id];
  }
  write(key, current);
}

/* ---------------------------------- units --------------------------------- */

export function useDoneUnits(): DoneMap {
  return useMap(UNIT_KEY);
}

export function setUnitDone(id: string, done: boolean) {
  toggleIn(UNIT_KEY, id, done);
}

export function useToggleUnit(id: string, done: boolean) {
  return useCallback(() => setUnitDone(id, !done), [id, done]);
}

/* ---------------------------------- links --------------------------------- */

export function useDoneLinks(): DoneMap {
  return useMap(LINK_KEY);
}

export function setLinkDone(id: string, done: boolean) {
  toggleIn(LINK_KEY, id, done);
}

/** Tick or clear a whole unit's links in one write. */
export function setLinksDone(ids: string[], done: boolean) {
  const current = {...read(LINK_KEY)};
  const now = Date.now();
  ids.forEach((id) => {
    if (done) {
      current[id] = now;
    } else {
      delete current[id];
    }
  });
  write(LINK_KEY, current);
}

/* ---------------------------------- reset --------------------------------- */

export function clearRoadmap() {
  write(UNIT_KEY, {});
  write(LINK_KEY, {});
}
