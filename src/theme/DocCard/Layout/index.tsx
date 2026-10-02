import Link from '@docusaurus/Link';
import isInternalUrl from '@docusaurus/isInternalUrl';
import {ThemeClassNames} from '@docusaurus/theme-common';
import type {PropSidebarItem} from '@docusaurus/plugin-content-docs';
import clsx from 'clsx';
import {useMemo, type ReactNode} from 'react';

import {useReadDocs} from '@site/src/lib/progress';

import styles from './styles.module.css';

type Props = {
  item: PropSidebarItem;
  className?: string;
  href: string;
  icon: ReactNode;
  title: string;
  description?: string;
};

const NUMBERED = /^\s*(\d{1,3})\s*[·.)\-:]\s*(.+)$/;
const LEADING_EMOJI = /^(\p{Extended_Pictographic}[️‍\p{Extended_Pictographic}]*)\s*(.+)$/u;

function parseLabel(label: string) {
  const numbered = label.match(NUMBERED);
  if (numbered) return {badge: numbered[1], title: numbered[2].trim(), emoji: null};
  const emoji = label.match(LEADING_EMOJI);
  if (emoji) return {badge: null, title: emoji[2].trim(), emoji: emoji[1]};
  return {badge: null, title: label.trim(), emoji: null};
}

function hrefsOf(item: PropSidebarItem): string[] {
  if (item.type === 'link') return isInternalUrl(item.href) ? [item.href] : [];
  if (item.type === 'category') {
    const own = item.href ? [item.href] : [];
    return [...own, ...item.items.flatMap(hrefsOf)];
  }
  return [];
}

function KindIcon({kind}: {kind: 'section' | 'note' | 'external'}) {
  if (kind === 'section') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2h7a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z" />
      </svg>
    );
  }
  if (kind === 'external') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M14 4h6v6M20 4l-9 9M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 3.5h7l4.5 4.5v11a1.5 1.5 0 0 1-1.5 1.5H7A1.5 1.5 0 0 1 5.5 19V5A1.5 1.5 0 0 1 7 3.5zM14 3.5V8h4.5M8.5 12.5h7M8.5 16h5" />
    </svg>
  );
}

export default function DocCardLayout({item, className, href, title, description}: Props): ReactNode {
  const read = useReadDocs();
  const label = parseLabel('label' in item ? item.label : title);
  const summary = description && !/^\d+ items?$/.test(description.trim()) ? description : undefined;
  const kind = item.type === 'category' ? 'section' : isInternalUrl(href) ? 'note' : 'external';

  const progress = useMemo(() => {
    const hrefs = [...new Set(hrefsOf(item))];
    return {total: hrefs.length, read: hrefs.filter((link) => read[link]).length};
  }, [item, read]);

  const isRead = kind === 'note' && Boolean(read[href]);
  const sectionPct = progress.total ? Math.round((progress.read / progress.total) * 100) : 0;

  return (
    <Link
      href={href}
      className={clsx('card', ThemeClassNames.docs.docCard.container, styles.card, className)}
      data-kind={kind}
      data-read={isRead || (kind === 'section' && progress.total > 0 && progress.read === progress.total) || undefined}>
      <span className={styles.badge} aria-hidden="true">
        {label.badge ?? label.emoji ?? <KindIcon kind={kind} />}
      </span>

      <span className={styles.body}>
        <span className={styles.title} title={label.title}>
          {label.title}
        </span>
        {summary && (
          <span className={styles.description} title={summary}>
            {summary}
          </span>
        )}
        <span className={styles.meta}>
          <span className={styles.kind}>
            <KindIcon kind={kind} />
            {kind === 'section' ? `${progress.total} notes` : kind === 'external' ? 'External link' : 'Note'}
          </span>
          {kind === 'section' && progress.read > 0 && (
            <span className={styles.sectionProgress}>
              <span className={styles.track} aria-hidden="true">
                <span style={{width: `${sectionPct}%`}} />
              </span>
              {progress.read}/{progress.total} read
            </span>
          )}
        </span>
      </span>

      {isRead && (
        <span className={styles.readMark} aria-label="Read">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m5 12.5 4.5 4.5L19 7.5" />
          </svg>
        </span>
      )}
      <span className={styles.arrow} aria-hidden="true">
        →
      </span>
    </Link>
  );
}
