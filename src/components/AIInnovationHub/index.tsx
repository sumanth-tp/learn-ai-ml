import {useEffect, useMemo, useRef, useState} from 'react';

import AISettingsDialog from '@site/src/components/AISettingsDialog';
import {loadAISettings, type AISettings} from '@site/src/lib/aiSettings';

import {fetchAllDiscoveries, ITEMS_PER_SOURCE} from './fetchers';
import {analyseDiscoveries} from './GeminiAnalyzer';
import {addBatch, emptyHistory, HISTORY_KEY, readHistory, type UpdateHistory} from './history';
import NewsCard from './NewsCard';
import type {Discovery, DiscoveryCategory} from './types';
import styles from './styles.module.css';

const CATEGORIES: Array<{key: DiscoveryCategory | 'all'; label: string}> = [
  {key: 'all', label: 'All updates'}, {key: 'papers', label: 'Papers'}, {key: 'models', label: 'Models'}, {key: 'tools', label: 'Tools'}, {key: 'videos', label: 'Videos'},
];

export default function AIInnovationHub() {
  const [settings, setSettings] = useState<AISettings>(() => loadAISettings());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [history, setHistory] = useState<UpdateHistory>(emptyHistory);
  const [ready, setReady] = useState(false);
  const [category, setCategory] = useState<DiscoveryCategory | 'all'>('all');
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [storageError, setStorageError] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const selected = history.batches.find((batch) => batch.createdAt === history.selectedAt) ?? history.batches[0];
  const items = selected?.items ?? [];
  const reports = selected?.reports ?? [];
  const createdAt = selected?.createdAt;

  useEffect(() => {
    try {
      const saved = readHistory(localStorage);
      setHistory(saved);
      if (saved.batches.length > 0) setNotice('Showing saved updates. Refresh updates checks the sources again.');
    } catch {
      setStorageError('Browser storage is unavailable. Updates will only be kept while this page is open.');
    }
    setReady(true);
    return () => abortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (!ready) return;
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
      setStorageError('');
    } catch {
      setStorageError('Updates could not be saved in this browser. Keep this page open to use the current history.');
    }
  }, [history, ready]);

  const visible = useMemo(() => category === 'all' ? items : items.filter((item) => item.category === category), [category, items]);
  const counts = useMemo(() => new Map(CATEGORIES.map(({key}) => [key, key === 'all' ? items.length : items.filter((item) => item.category === key).length])), [items]);

  async function discover() {
    if (abortRef.current || !ready) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError('');
    setNotice('');
    setStage('Fetching fresh papers, models and tools…');
    try {
      const result = await fetchAllDiscoveries(settings, controller.signal, history.seen, history.batches.flatMap((batch) => batch.items));
      const failures = result.reports.filter((report) => report.status === 'error');
      if (result.items.length === 0) {
        if (failures.length > 0) setError(failures.map((report) => `${report.source}: ${report.message}`).join(' '));
        setNotice(failures.length === result.reports.length
          ? 'Sources could not be checked. Your saved updates are still available.'
          : 'No updates are available from the sources yet. Try again later.');
        return;
      }
      setStage(settings.geminiApiKey || settings.groqApiKey ? 'AI is turning source records into learning notes…' : 'Preparing source summaries…');
      let discoveries: Discovery[];
      try {
        discoveries = await analyseDiscoveries(result.items, settings, controller.signal);
      } catch (analysisError) {
        controller.signal.throwIfAborted();
        discoveries = await analyseDiscoveries(result.items, {...settings, geminiApiKey: '', groqApiKey: ''}, controller.signal);
        setError(`Sources loaded, but AI editing was skipped: ${(analysisError as Error).message}`);
      }
      controller.signal.throwIfAborted();
      const now = Math.max(Date.now(), (history.batches[0]?.createdAt ?? 0) + 1);
      setHistory((previous) => addBatch(previous, {createdAt: now, items: discoveries, reports: result.reports}));
      setCategory('all');
      const repeatedCount = discoveries.length - result.newCount;
      setNotice(`${discoveries.length} updates loaded · ${result.newCount} new${repeatedCount > 0 ? ` · ${repeatedCount} previously shown` : ''}.`);
    } catch (caught) {
      if ((caught as Error).name !== 'AbortError') setError((caught as Error).message);
    } finally {
      if (abortRef.current === controller) {
        setLoading(false);
        setStage('');
        abortRef.current = null;
      }
    }
  }

  const editedCount = items.filter((item) => item.aiEdited).length;

  return (
    <>
      <section className={styles.controlPanel}>
        <div className={styles.controlCopy}>
          <span className={styles.liveMark}><i /> Live, on demand</span>
          <h2>What changed in AI?</h2>
          <p>{ITEMS_PER_SOURCE} updates per category, with new items first and previously shown items filling any gaps. Videos cover the last 90 days. Your last three batches are saved in this browser.</p>
        </div>
        <div className={styles.controlActions}>
          <button type="button" className={styles.settingsButton} onClick={() => setSettingsOpen(true)} aria-label="Open API settings">⚙ API settings</button>
          <button type="button" className={styles.discoverButton} onClick={discover} disabled={loading || !ready}>
            <span aria-hidden="true" className={loading ? styles.spinning : ''}>↻</span>
            {loading ? 'Discovering…' : items.length ? 'Refresh updates' : 'Discover latest AI updates'}
          </button>
        </div>
      </section>

      {loading && <div className={styles.stage} role="status"><span /><div><strong>Scanning public sources</strong><p>{stage}</p></div></div>}
      {error && <div className={styles.error} role="alert">{error}</div>}
      {storageError && <div className={styles.error} role="alert">{storageError}</div>}
      {notice && <div className={styles.notice} role="status">{notice}</div>}

      {history.batches.length > 0 && (
        <div className={styles.history}>
          <label htmlFor="update-batch">Saved updates</label>
          <select id="update-batch" value={history.selectedAt ?? ''} onChange={(event) => {
            setHistory((previous) => ({...previous, selectedAt: Number(event.target.value)}));
            setCategory('all');
            setNotice('Showing saved updates. Refresh updates checks the sources again.');
            setError('');
          }}>
            {history.batches.map((batch, index) => (
              <option key={batch.createdAt} value={batch.createdAt}>
                {index === 0 ? 'Latest' : index === 1 ? 'Previous' : 'Oldest'} · {new Date(batch.createdAt).toLocaleString(undefined, {dateStyle: 'medium', timeStyle: 'medium'})} · {batch.items.length} updates
              </option>
            ))}
          </select>
          <span>{history.batches.length} of 3 saved in this browser</span>
        </div>
      )}

      {selected && CATEGORIES.some(({key}) => key !== 'all' && (counts.get(key) ?? 0) < ITEMS_PER_SOURCE) && (
        <div className={styles.notice}>This saved batch has fewer than {ITEMS_PER_SOURCE} items in one or more categories. Refresh updates to check all sources for a fuller batch.</div>
      )}

      {(reports.length > 0 || items.length > 0) && (
        <>
          <div className={styles.statusRow}>
            <div className={styles.sources} aria-label="Source status">
              {reports.map((report) => <span key={report.source} data-status={report.status} title={report.message}>{report.source}<b>{report.status === 'skipped' ? 'off' : `${report.count}/${ITEMS_PER_SOURCE}${report.status === 'error' ? ' !' : ''}`}</b></span>)}
            </div>
            {createdAt && <span className={styles.updated}>Fetched {new Date(createdAt).toLocaleString(undefined, {dateStyle: 'medium', timeStyle: 'short'})}{editedCount > 0 ? ` · ${editedCount} AI-edited` : ' · source summaries'}</span>}
          </div>
          {reports.some((report) => report.message) && (
            <div className={styles.sourceNotes}>
              {reports.filter((report) => report.message).map((report) => (
                <div key={report.source} data-status={report.status}>
                  <strong>{report.source}</strong>
                  <span>{report.message}</span>
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
