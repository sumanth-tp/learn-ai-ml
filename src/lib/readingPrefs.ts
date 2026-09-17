/**
 * Reader-controlled typography. Stored per browser and applied as data
 * attributes on <html>, which src/css/custom.css turns into type size and
 * measure. A tiny inline script (see plugins/learn-index) applies the saved
 * value before first paint so there is no flash.
 */
import {useCallback, useSyncExternalStore} from 'react';

export type ReadingSize = 'cozy' | 'default' | 'large';
export type ReadingWidth = 'default' | 'wide';

export type ReadingPrefs = {size: ReadingSize; width: ReadingWidth};

const KEY = 'learn-ai-ml:reading';
const DEFAULTS: ReadingPrefs = {size: 'default', width: 'default'};

const listeners = new Set<() => void>();
let cache: ReadingPrefs = DEFAULTS;
let cacheRaw: string | null = null;

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot(): ReadingPrefs {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(KEY);
  } catch {
    return DEFAULTS;
  }
  if (raw === cacheRaw) {
    return cache;
  }
  cacheRaw = raw;
  try {
    const parsed = raw ? (JSON.parse(raw) as Partial<ReadingPrefs>) : {};
    cache = {...DEFAULTS, ...parsed};
  } catch {
    cache = DEFAULTS;
  }
  return cache;
}

function getServerSnapshot(): ReadingPrefs {
  return DEFAULTS;
}

function apply(prefs: ReadingPrefs) {
  const root = document.documentElement;
  root.dataset.readingSize = prefs.size;
  root.dataset.readingWidth = prefs.width;
}

export function useReadingPrefs() {
  const prefs = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const update = useCallback((patch: Partial<ReadingPrefs>) => {
    const next = {...getSnapshot(), ...patch};
    try {
      window.localStorage.setItem(KEY, JSON.stringify(next));
    } catch {
      /* not persisted, but still applied for this session */
    }
    apply(next);
    listeners.forEach((listener) => listener());
  }, []);

  return [prefs, update] as const;
}
