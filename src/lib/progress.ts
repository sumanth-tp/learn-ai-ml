/**
 * Per-browser reading progress.
 *
 * Nothing leaves the device: state lives in localStorage and is exposed through
 * `useSyncExternalStore`, so every component (doc toolbar, explore page,
 * homepage) re-renders together. All access is try/caught because storage can
 * be blocked or full, and the site must still work when it is.
 */
import {useCallback, useSyncExternalStore} from 'react';

const READ_KEY = 'learn-ai-ml:read';
const LAST_KEY = 'learn-ai-ml:last-visited';

export type ReadMap = Record<string, number>;
export type LastVisited = {permalink: string; title: string; at: number} | null;

const EMPTY_READ: ReadMap = {};

const listeners = new Set<() => void>();

let readCache: ReadMap = EMPTY_READ;
let readCacheRaw: string | null = null;
let lastCache: LastVisited = null;
let lastCacheRaw: string | null = null;

function emit() {
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  // Keep tabs in sync with each other.
  const onStorage = (event: StorageEvent) => {
    if (event.key === READ_KEY || event.key === LAST_KEY || event.key === null) {
      listener();
    }
  };
  window.addEventListener('storage', onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener('storage', onStorage);
  };
}

function readRaw(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeRaw(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable — progress simply does not persist */
  }
}

/* --------------------------------- read map -------------------------------- */

function getReadSnapshot(): ReadMap {
  const raw = readRaw(READ_KEY);
  if (raw === readCacheRaw) {
    return readCache;
  }
  readCacheRaw = raw;
  try {
    const parsed = raw ? (JSON.parse(raw) as ReadMap) : EMPTY_READ;
    readCache = parsed && typeof parsed === 'object' ? parsed : EMPTY_READ;
  } catch {
    readCache = EMPTY_READ;
  }
  return readCache;
}

function getReadServerSnapshot(): ReadMap {
  return EMPTY_READ;
}

export function useReadDocs(): ReadMap {
  return useSyncExternalStore(subscribe, getReadSnapshot, getReadServerSnapshot);
}

export function useIsRead(permalink: string): boolean {
  const read = useReadDocs();
  return Boolean(read[permalink]);
}

export function setRead(permalink: string, value: boolean) {
  const next = {...getReadSnapshot()};
  if (value) {
    next[permalink] = Date.now();
  } else {
    delete next[permalink];
  }
  writeRaw(READ_KEY, JSON.stringify(next));
  emit();
}

export function useToggleRead(permalink: string) {
  const isRead = useIsRead(permalink);
  return useCallback(() => setRead(permalink, !isRead), [permalink, isRead]);
}

export function clearRead() {
  writeRaw(READ_KEY, JSON.stringify({}));
  emit();
}

/* ------------------------------- last visited ------------------------------ */

function getLastSnapshot(): LastVisited {
  const raw = readRaw(LAST_KEY);
  if (raw === lastCacheRaw) {
    return lastCache;
  }
  lastCacheRaw = raw;
  try {
    lastCache = raw ? (JSON.parse(raw) as LastVisited) : null;
  } catch {
    lastCache = null;
  }
  return lastCache;
}

function getLastServerSnapshot(): LastVisited {
  return null;
}

export function useLastVisited(): LastVisited {
  return useSyncExternalStore(subscribe, getLastSnapshot, getLastServerSnapshot);
}

export function recordVisit(permalink: string, title: string) {
  const current = getLastSnapshot();
  if (current?.permalink === permalink) {
    return;
  }
  writeRaw(LAST_KEY, JSON.stringify({permalink, title, at: Date.now()}));
  emit();
}
