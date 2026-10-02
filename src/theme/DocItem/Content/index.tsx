import Link from '@docusaurus/Link';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import Content from '@theme-original/DocItem/Content';
import type ContentType from '@theme/DocItem/Content';
import type {WrapperProps} from '@docusaurus/types';
import {JSX, useEffect, useMemo, useRef, useState} from 'react';

import {PATH_ROUTE, stageById} from '@site/src/data/learningPath';
import {useDocsIndex} from '@site/src/lib/docsIndex';
import {useIsRead, useReadDocs, useToggleRead} from '@site/src/lib/progress';
import {useReadingPrefs, type ReadingSize} from '@site/src/lib/readingPrefs';
import VoiceReader from '@site/src/components/VoiceReader';

import styles from './styles.module.css';

type Props = WrapperProps<typeof ContentType>;

const WORDS_PER_MINUTE = 220;

function ReadingTime({articleRef}: {articleRef: React.RefObject<HTMLDivElement | null>}) {
  const [minutes, setMinutes] = useState<number | null>(null);

  useEffect(() => {
    // Measured from the rendered article so code blocks, tables and math are
    // counted the way a reader actually meets them.
    const text = articleRef.current?.innerText ?? '';
    const words = text.trim().split(/\s+/).filter(Boolean).length;
    setMinutes(words > 0 ? Math.max(1, Math.round(words / WORDS_PER_MINUTE)) : null);
  }, [articleRef]);

  if (minutes === null) {
    return null;
  }

  return (
    <span className={styles.metaItem}>
      <svg viewBox="0 0 24 24" className={styles.metaIcon} aria-hidden="true">
        <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path
          d="M12 7.5V12l3 1.8"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
      {minutes} min read
    </span>
  );
}

const SIZE_STEPS: ReadingSize[] = ['cozy', 'default', 'large'];

function ReadingControls() {
  const [prefs, setPrefs] = useReadingPrefs();
  const index = SIZE_STEPS.indexOf(prefs.size);

  return (
    <div className={styles.controlGroup} role="group" aria-label="Reading comfort">
      <button
        type="button"
        className={styles.iconButton}
        onClick={() => setPrefs({size: SIZE_STEPS[Math.max(0, index - 1)]})}
        disabled={index === 0}
        title="Smaller text"
        aria-label="Decrease text size">
        <span className={styles.aaSmall}>A</span>
      </button>
      <button
        type="button"
        className={styles.iconButton}
        onClick={() => setPrefs({size: SIZE_STEPS[Math.min(SIZE_STEPS.length - 1, index + 1)]})}
        disabled={index === SIZE_STEPS.length - 1}
        title="Larger text"
        aria-label="Increase text size">
        <span className={styles.aaLarge}>A</span>
      </button>
      <button
        type="button"
        className={styles.iconButton}
        data-active={prefs.width === 'wide'}
        onClick={() => setPrefs({width: prefs.width === 'wide' ? 'default' : 'wide'})}
        title={prefs.width === 'wide' ? 'Comfortable line length' : 'Full width'}
        aria-pressed={prefs.width === 'wide'}
        aria-label="Toggle full width">
        <svg viewBox="0 0 24 24" className={styles.metaIcon} aria-hidden="true">
          <path
            d="M4 7v10M20 7v10M7 12h10m0 0-2.5-2.5M17 12l-2.5 2.5M7 12l2.5-2.5M7 12l2.5 2.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  );
}

function CopyLinkButton() {
  const [copied, setCopied] = useState(false);

  return (
    <button
      type="button"
      className={styles.iconButton}
      title="Copy link to this page"
      aria-label="Copy link to this page"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(window.location.href.split('#')[0]);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1600);
        } catch {
          /* clipboard blocked — nothing useful to show */
        }
      }}>
      {copied ? (
        <svg viewBox="0 0 24 24" className={styles.metaIcon} aria-hidden="true">
          <path
            d="m5 12.5 4.5 4.5L19 7.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" className={styles.metaIcon} aria-hidden="true">
          <path
            d="M10 13.5a3.5 3.5 0 0 0 5 0l3-3a3.54 3.54 0 0 0-5-5l-1 1M14 10.5a3.5 3.5 0 0 0-5 0l-3 3a3.54 3.54 0 0 0 5 5l1-1"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      )}
    </button>
  );
}

function StageChip({permalink}: {permalink: string}) {
  const docs = useDocsIndex();
  const read = useReadDocs();

  const info = useMemo(() => {
    const stageId = docs.find((doc) => doc.permalink === permalink)?.stage;
    const stage = stageById(stageId ?? undefined);
    if (!stage) {
      return null;
    }
    const inStage = docs.filter((doc) => doc.stage === stage.id);
    return {
      stage,
      total: inStage.length,
      read: inStage.filter((doc) => read[doc.permalink]).length,
    };
  }, [docs, read, permalink]);

  if (!info) {
    return null;
  }

  const {stage} = info;
  return (
    <Link
      to={`${PATH_ROUTE}#${stage.id}`}
      className={styles.stageChip}
      data-tone={stage.tone}
      title={`${info.read} of ${info.total} notes in this stage read — open the learning path`}>
      <span className={styles.stageNumber}>{stage.number ?? '·'}</span>
      <span className={styles.stageName}>{stage.title}</span>
      <span className={styles.stageCount}>
        {info.read}/{info.total}
      </span>
    </Link>
  );
}

export default function ContentWrapper(props: Props): JSX.Element {
  const {metadata} = useDoc();
  const articleRef = useRef<HTMLDivElement>(null);
  const isRead = useIsRead(metadata.permalink);
  const toggleRead = useToggleRead(metadata.permalink);

  return (
    <>
      <div className={styles.metaBar}>
        <div className={styles.metaGroup}>
          <StageChip permalink={metadata.permalink} />
          <ReadingTime articleRef={articleRef} />
          {metadata.lastUpdatedAt && (
            <span className={styles.metaItem}>
              Updated{' '}
              {new Date(metadata.lastUpdatedAt).toLocaleDateString(undefined, {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
              })}
            </span>
          )}
        </div>

        <div className={styles.metaActions}>
          <VoiceReader articleRef={articleRef} permalink={metadata.permalink} />
          <ReadingControls />
          <CopyLinkButton />
          <button
          type="button"
          className={styles.readToggle}
          data-read={isRead ? 'true' : 'false'}
          onClick={toggleRead}
          aria-pressed={isRead}>
          <svg viewBox="0 0 24 24" className={styles.metaIcon} aria-hidden="true">
            <path
              d="m5 12.5 4.5 4.5L19 7.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          {isRead ? 'Read' : 'Mark as read'}
          </button>
        </div>
      </div>

      <div ref={articleRef}>
        <Content {...props} />
      </div>
    </>
  );
}
