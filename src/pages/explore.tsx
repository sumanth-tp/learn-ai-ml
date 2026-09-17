import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import {useMemo, useRef, useState, type ReactNode} from 'react';

import {SECTIONS, sectionOf, useDocsIndex} from '@site/src/lib/docsIndex';
import {clearRead, useReadDocs} from '@site/src/lib/progress';

import styles from './explore.module.css';

function normalise(value: string) {
  return value.toLowerCase().normalize('NFKD');
}

export default function Explore(): ReactNode {
  const docs = useDocsIndex();
  const read = useReadDocs();
  const [query, setQuery] = useState('');
  const [section, setSection] = useState<string>('all');
  const [unreadOnly, setUnreadOnly] = useState(false);
  const [sort, setSort] = useState<'order' | 'title' | 'updated'>('order');
  const inputRef = useRef<HTMLInputElement>(null);

  const decorated = useMemo(
    () =>
      docs.map((doc) => ({
        doc,
        section: sectionOf(doc),
        haystack: normalise(`${doc.title} ${doc.description} ${doc.tags.join(' ')} ${doc.dir}`),
      })),
    [docs],
  );

  const sectionCounts = useMemo(() => {
    const counts = new Map<string, {total: number; read: number}>();
    decorated.forEach(({doc, section: docSection}) => {
      const entry = counts.get(docSection.key) ?? {total: 0, read: 0};
      entry.total += 1;
      if (read[doc.permalink]) {
        entry.read += 1;
      }
      counts.set(docSection.key, entry);
    });
    return counts;
  }, [decorated, read]);

  const results = useMemo(() => {
    const needle = normalise(query.trim());
    const filtered = decorated.filter(({doc, section: docSection, haystack}) => {
      if (section !== 'all' && docSection.key !== section) {
        return false;
      }
      if (unreadOnly && read[doc.permalink]) {
        return false;
      }
      return needle === '' || haystack.includes(needle);
    });

    if (sort === 'title') {
      return [...filtered].sort((a, b) => a.doc.title.localeCompare(b.doc.title));
    }
    if (sort === 'updated') {
      return [...filtered].sort((a, b) => (b.doc.updatedAt ?? 0) - (a.doc.updatedAt ?? 0));
    }
    return filtered;
  }, [decorated, query, section, unreadOnly, read, sort]);

  const totalRead = Object.keys(read).length;
  const overallPct = docs.length ? Math.round((totalRead / docs.length) * 100) : 0;

  const visibleSections = SECTIONS.filter((item) => sectionCounts.has(item.key));

  return (
    <Layout
      title="Explore all notes"
      description="Search and filter every note on the site by topic, tag or reading status.">
      <main className={styles.page}>
        <div className="container">
          <header className={styles.head}>
            <Heading as="h1" className={styles.title}>
              Explore
            </Heading>
            <p className={styles.subtitle}>
              Every note on the site in one list — {docs.length} of them. Filter by
              topic, search titles, descriptions and tags, or hide what you have
              already read.
            </p>
          </header>

          <section className={styles.progressPanel} aria-label="Your progress">
            <div className={styles.progressHead}>
              <div>
                <span className={styles.progressValue}>{totalRead}</span>
                <span className={styles.progressOf}>/ {docs.length} notes read</span>
              </div>
              <div className={styles.progressActions}>
                <span className={styles.progressPct}>{overallPct}%</span>
                {totalRead > 0 && (
                  <button
                    type="button"
                    className={styles.resetButton}
                    onClick={() => {
                      if (window.confirm('Clear all reading progress in this browser?')) {
                        clearRead();
                      }
                    }}>
                    Reset
                  </button>
                )}
              </div>
            </div>
            <div className={styles.progressTrack}>
              <div className={styles.progressFill} style={{width: `${overallPct}%`}} />
            </div>
            <p className={styles.progressNote}>
              Progress is stored in this browser only — nothing is uploaded.
            </p>
          </section>

          <div className={styles.controls}>
            <div className={styles.searchWrap}>
              <svg viewBox="0 0 24 24" className={styles.searchIcon} aria-hidden="true">
                <circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.7" />
                <path d="m16 16 4.5 4.5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
              </svg>
              <input
                ref={inputRef}
                type="search"
                className={styles.search}
                placeholder="Search notes, tags, topics…"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                aria-label="Search notes"
              />
              {query && (
                <button
                  type="button"
                  className={styles.clearSearch}
                  onClick={() => {
                    setQuery('');
                    inputRef.current?.focus();
                  }}
                  aria-label="Clear search">
                  ✕
                </button>
              )}
            </div>

            <label className={styles.sortWrap}>
              <span className={styles.sortLabel}>Sort</span>
              <select
                className={styles.sort}
                value={sort}
                onChange={(event) => setSort(event.target.value as typeof sort)}>
                <option value="order">Course order</option>
                <option value="title">A – Z</option>
                <option value="updated">Recently updated</option>
              </select>
            </label>

            <label className={styles.unreadToggle}>
              <input
                type="checkbox"
                checked={unreadOnly}
                onChange={(event) => setUnreadOnly(event.target.checked)}
              />
              Unread only
            </label>
          </div>

          <div className={styles.chips} role="group" aria-label="Filter by section">
            <button
              type="button"
              className={styles.chip}
              data-active={section === 'all'}
              onClick={() => setSection('all')}>
              All
              <span className={styles.chipCount}>{docs.length}</span>
            </button>
            {visibleSections.map((item) => {
              const counts = sectionCounts.get(item.key)!;
              return (
                <button
                  key={item.key}
                  type="button"
                  className={styles.chip}
                  data-tone={item.tone}
                  data-active={section === item.key}
                  onClick={() => setSection(section === item.key ? 'all' : item.key)}>
                  {item.label}
                  <span className={styles.chipCount}>
                    {counts.read > 0 ? `${counts.read}/${counts.total}` : counts.total}
                  </span>
                </button>
              );
            })}
          </div>

          <p className={styles.resultCount} role="status">
            {results.length} {results.length === 1 ? 'note' : 'notes'}
            {query && ` matching “${query}”`}
          </p>

          {results.length === 0 ? (
            <div className={styles.empty}>
              <p>Nothing matches that yet.</p>
              <button
                type="button"
                className={styles.resetButton}
                onClick={() => {
                  setQuery('');
                  setSection('all');
                  setUnreadOnly(false);
                }}>
                Clear filters
              </button>
            </div>
          ) : (
            <ul className={styles.results}>
              {results.map(({doc, section: docSection}) => (
                <li key={doc.permalink}>
                  <Link
                    to={doc.permalink}
                    className={styles.result}
                    data-tone={docSection.tone}
                    data-read={Boolean(read[doc.permalink])}>
                    <span className={styles.resultTick} aria-hidden="true">
                      <svg viewBox="0 0 24 24">
                        <path
                          d="m5 12.5 4.5 4.5L19 7.5"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.4"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </span>
                    <span className={styles.resultBody}>
                      <span className={styles.resultTitle}>{doc.title}</span>
                      {doc.description && (
                        <span className={styles.resultDesc}>{doc.description}</span>
                      )}
                    </span>
                    <span className={styles.resultSection}>{docSection.label}</span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>
      </main>
    </Layout>
  );
}
