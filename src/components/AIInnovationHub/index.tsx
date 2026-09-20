import {useEffect, useMemo, useRef, useState} from 'react';

import AISettingsDialog from '@site/src/components/AISettingsDialog';
import {loadAISettings, type AISettings} from '@site/src/lib/aiSettings';

import {fetchAllDiscoveries} from './fetchers';
import {analyseDiscoveries} from './GeminiAnalyzer';
import NewsCard from './NewsCard';
import type {Discovery, DiscoveryCategory, SourceReport} from './types';
import styles from './styles.module.css';

const CACHE_KEY = 'learn-ai-ml:discoveries:v7';
const LEGACY_CACHE_KEYS = [
  'learn-ai-ml:discoveries:v1',
  'learn-ai-ml:discoveries:v2',
  'learn-ai-ml:discoveries:v3',
  'learn-ai-ml:discoveries:v4',
  'learn-ai-ml:discoveries:v5',
  'learn-ai-ml:discoveries:v6',
];
const CACHE_MS = 12 * 60 * 60 * 1000;
const CATEGORIES: Array<{key: DiscoveryCategory | 'all'; label: string}> = [
  {key: 'all', label: 'All updates'}, {key: 'papers', label: 'Papers'}, {key: 'models', label: 'Models'}, {key: 'tools', label: 'Tools'}, {key: 'videos', label: 'Videos'},
];

type Cache = {createdAt: number; items: Discovery[]; reports: SourceReport[]};

export default function AIInnovationHub() {
  const [settings, setSettings] = useState<AISettings>(() => loadAISettings());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [items, setItems] = useState<Discovery[]>([]);
  const [reports, setReports] = useState<SourceReport[]>([]);
  const [createdAt, setCreatedAt] = useState<number | null>(null);
  const [category, setCategory] = useState<DiscoveryCategory | 'all'>('all');
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState('');
  const [error, setError] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    try {
      LEGACY_CACHE_KEYS.forEach((key) => localStorage.removeItem(key));
      const cached = JSON.parse(localStorage.getItem(CACHE_KEY) || 'null') as Cache | null;
      if (cached && Date.now() - cached.createdAt < CACHE_MS) {
        setItems(cached.items);
        setReports(cached.reports);
        setCreatedAt(cached.createdAt);
      }
    } catch {
      // Ignore malformed or unavailable cache.
    }
    return () => abortRef.current?.abort();
  }, []);

  const visible = useMemo(() => category === 'all' ? items : items.filter((item) => item.category === category), [category, items]);
  const counts = useMemo(() => new Map(CATEGORIES.map(({key}) => [key, key === 'all' ? items.length : items.filter((item) => item.category === key).length])), [items]);

  async function discover() {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError('');
    setStage('Fetching fresh papers, models and tools…');
    try {
      const result = await fetchAllDiscoveries(settings, controller.signal);
      if (result.items.length === 0) throw new Error('No source returned an update. Check the source messages and try again.');
      setReports(result.reports);
      setStage(settings.geminiApiKey || settings.groqApiKey ? 'AI is turning source records into learning notes…' : 'Preparing source summaries…');
      let discoveries: Discovery[];
      try {
        discoveries = await analyseDiscoveries(result.items, settings, controller.signal);
      } catch (analysisError) {
        discoveries = await analyseDiscoveries(result.items, {...settings, geminiApiKey: '', groqApiKey: ''}, controller.signal);
        setError(`Sources loaded, but AI editing was skipped: ${(analysisError as Error).message}`);
      }
      const now = Date.now();
      setItems(discoveries);
      setCreatedAt(now);
      try { localStorage.setItem(CACHE_KEY, JSON.stringify({createdAt: now, items: discoveries, reports: result.reports} satisfies Cache)); } catch { /* Cache is optional. */ }
    } catch (caught) {
      if ((caught as Error).name !== 'AbortError') setError((caught as Error).message);
    } finally {
      setLoading(false);
      setStage('');
      abortRef.current = null;
    }
  }

  const editedCount = items.filter((item) => item.aiEdited).length;

  return (
    <>
      <section className={styles.controlPanel}>
        <div className={styles.controlCopy}>
          <span className={styles.liveMark}><i /> Live, on demand</span>
          <h2>What changed in AI?</h2>
          <p>Pull recent records from public sources when you want them. Nothing runs in the background.</p>
        </div>
        <div className={styles.controlActions}>
          <button type="button" className={styles.settingsButton} onClick={() => setSettingsOpen(true)} aria-label="Open API settings">⚙ API settings</button>
          <button type="button" className={styles.discoverButton} onClick={discover} disabled={loading}>
            <span aria-hidden="true" className={loading ? styles.spinning : ''}>↻</span>
            {loading ? 'Discovering…' : items.length ? 'Refresh updates' : 'Discover latest AI updates'}
          </button>
        </div>
      </section>

      {loading && <div className={styles.stage} role="status"><span /><div><strong>Scanning public sources</strong><p>{stage}</p></div></div>}
      {error && <div className={styles.error} role="alert">{error}</div>}

      {(reports.length > 0 || items.length > 0) && (
        <>
          <div className={styles.statusRow}>
            <div className={styles.sources} aria-label="Source status">
              {reports.map((report) => <span key={report.source} data-status={report.status} title={report.message}>{report.source}<b>{report.status === 'ok' ? report.count : report.status === 'skipped' ? 'off' : '!'}</b></span>)}
            </div>
            {createdAt && <span className={styles.updated}>Updated {new Date(createdAt).toLocaleString(undefined, {dateStyle: 'medium', timeStyle: 'short'})}{editedCount > 0 ? ` · ${editedCount} AI-edited` : ' · source summaries'}</span>}
          </div>
          {reports.some((report) => report.status !== 'ok') && (
            <div className={styles.sourceNotes}>
              {reports.filter((report) => report.status !== 'ok').map((report) => (
                <div key={report.source} data-status={report.status}>
                  <strong>{report.source}</strong>
                  <span>
                    {report.source === 'YouTube' && settings.youtubeApiKey
                      ? 'Key added. Refresh updates to load videos.'
                      : report.message}
                  </span>
                  {report.source.includes('Videos') && !settings.youtubeApiKey && (
                    <button type="button" onClick={() => setSettingsOpen(true)}>Use official API</button>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {items.length > 0 && (
        <>
          <nav className={styles.tabs} aria-label="Filter discoveries">
            {CATEGORIES.map(({key, label}) => <button type="button" key={key} data-active={category === key} onClick={() => setCategory(key)}>{label}<span>{counts.get(key) ?? 0}</span></button>)}
          </nav>
          {visible.length > 0 ? <div className={styles.grid}>{visible.map((item) => <NewsCard key={item.id} item={item} />)}</div> : <div className={styles.empty}>No {category} arrived in this refresh.</div>}
        </>
      )}

      {!loading && items.length === 0 && (
        <div className={styles.initial}>
          <div><span>01</span><strong>Click discover</strong><p>The browser calls each public API in parallel.</p></div>
          <div><span>02</span><strong>Optional AI edit</strong><p>Gemini simplifies only the records returned by those sources.</p></div>
          <div><span>03</span><strong>Read the original</strong><p>Every card keeps a direct link to its source.</p></div>
        </div>
      )}

      <AISettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} onSave={setSettings} />
    </>
  );
}
