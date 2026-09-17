import Link from '@docusaurus/Link';
import {useHistory} from '@docusaurus/router';
import Heading from '@theme/Heading';
import {useMemo, type ReactNode} from 'react';

import {SECTIONS, sectionOf, useDocsIndex} from '@site/src/lib/docsIndex';
import {useLastVisited, useReadDocs} from '@site/src/lib/progress';

import styles from './styles.module.css';

function timeAgo(timestamp: number): string {
  const seconds = Math.round((Date.now() - timestamp) / 1000);
  const table: [number, Intl.RelativeTimeFormatUnit][] = [
    [60, 'second'],
    [3600, 'minute'],
    [86400, 'hour'],
    [604800, 'day'],
    [2629800, 'week'],
    [31557600, 'month'],
  ];
  const formatter = new Intl.RelativeTimeFormat(undefined, {numeric: 'auto'});
  let previous = 1;
  for (const [limit, unit] of table) {
    if (seconds < limit) {
      return formatter.format(-Math.round(seconds / previous), unit);
    }
    previous = limit;
  }
  return formatter.format(-Math.round(seconds / 31557600), 'year');
}

export default function ContinueLearning(): ReactNode {
  const docs = useDocsIndex();
  const read = useReadDocs();
  const last = useLastVisited();
  const history = useHistory();

  const sectionProgress = useMemo(() => {
    const counts = new Map<string, {total: number; read: number}>();
    docs.forEach((doc) => {
      const {key} = sectionOf(doc);
      const entry = counts.get(key) ?? {total: 0, read: 0};
      entry.total += 1;
      if (read[doc.permalink]) {
        entry.read += 1;
      }
      counts.set(key, entry);
    });
    return SECTIONS.filter((section) => counts.has(section.key)).map((section) => ({
      ...section,
      ...counts.get(section.key)!,
    }));
  }, [docs, read]);

  const recent = useMemo(
    () =>
      [...docs]
        .filter((doc) => doc.updatedAt)
        .sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0))
        .slice(0, 5),
    [docs],
  );

  const totalRead = Object.keys(read).length;
  const pct = docs.length ? Math.round((totalRead / docs.length) * 100) : 0;

  const openRandom = () => {
    if (!docs.length) {
      return;
    }
    const unread = docs.filter((doc) => !read[doc.permalink]);
    const pool = unread.length ? unread : docs;
    history.push(pool[Math.floor(Math.random() * pool.length)].permalink);
  };

  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <Heading as="h2" className={styles.title}>
            Your shelf
          </Heading>
          <p className={styles.subtitle}>
            Reading progress lives in this browser — no account, nothing uploaded.
          </p>
        </div>

        <div className={styles.grid}>
          <div className={styles.primaryCard}>
            {last ? (
              <>
                <span className={styles.eyebrow}>Continue where you left off</span>
                <Link to={last.permalink} className={styles.continueTitle}>
                  {last.title}
                </Link>
                <span className={styles.continueMeta}>Opened {timeAgo(last.at)}</span>
              </>
            ) : (
              <>
                <span className={styles.eyebrow}>Nothing open yet</span>
                <Link to="/docs/intro" className={styles.continueTitle}>
                  Start with the AI/ML roadmap
                </Link>
                <span className={styles.continueMeta}>
                  Your most recent note will appear here.
                </span>
              </>
            )}

            <div className={styles.overall}>
              <div className={styles.overallHead}>
                <span>
                  <strong>{totalRead}</strong> of {docs.length} notes read
                </span>
                <span className={styles.overallPct}>{pct}%</span>
              </div>
              <div className={styles.track}>
                <div className={styles.fill} style={{width: `${pct}%`}} />
              </div>
            </div>

            <div className={styles.primaryActions}>
              <Link className={styles.ghostButton} to="/explore">
                Explore all notes
              </Link>
              <button type="button" className={styles.ghostButton} onClick={openRandom}>
                Surprise me
              </button>
            </div>
          </div>

          <div className={styles.card}>
            <span className={styles.eyebrow}>By section</span>
            <ul className={styles.sectionList}>
              {sectionProgress.map((section) => {
                const sectionPct = Math.round((section.read / section.total) * 100);
                return (
                  <li key={section.key} className={styles.sectionRow} data-tone={section.tone}>
                    <span className={styles.sectionLabel}>{section.label}</span>
                    <span className={styles.sectionTrack}>
                      <span className={styles.sectionFill} style={{width: `${sectionPct}%`}} />
                    </span>
                    <span className={styles.sectionCount}>
                      {section.read}/{section.total}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>

          <div className={styles.card}>
            <span className={styles.eyebrow}>Recently updated</span>
            {recent.length ? (
              <ul className={styles.recentList}>
                {recent.map((doc) => (
                  <li key={doc.permalink}>
                    <Link to={doc.permalink} className={styles.recentLink}>
                      <span className={styles.recentTitle}>{doc.title}</span>
                      <span className={styles.recentDate}>
                        {doc.updatedAt ? timeAgo(doc.updatedAt) : ''}
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <p className={styles.emptyNote}>Update timestamps appear after a git build.</p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
