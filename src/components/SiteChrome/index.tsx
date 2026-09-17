import {useHistory, useLocation} from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {useCallback, useEffect, useRef, useState} from 'react';

import {recordVisit} from '@site/src/lib/progress';

import styles from './styles.module.css';

/* ------------------------------------------------------------------ *
 * Reading progress bar (docs only)
 * ------------------------------------------------------------------ */

function ReadingProgress({active}: {active: boolean}) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!active) {
      setProgress(0);
      return undefined;
    }

    let frame = 0;
    const update = () => {
      frame = 0;
      const scrollable =
        document.documentElement.scrollHeight - window.innerHeight;
      setProgress(scrollable > 0 ? Math.min(1, window.scrollY / scrollable) : 0);
    };
    const onScroll = () => {
      if (!frame) {
        frame = window.requestAnimationFrame(update);
      }
    };

    update();
    window.addEventListener('scroll', onScroll, {passive: true});
    window.addEventListener('resize', onScroll);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, [active]);

  if (!active) {
    return null;
  }

  return (
    <div
      className={styles.progressTrack}
      role="progressbar"
      aria-label="Reading progress"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(progress * 100)}>
      <div className={styles.progressBar} style={{transform: `scaleX(${progress})`}} />
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Figure lightbox
 * ------------------------------------------------------------------ */

function Lightbox() {
  const [src, setSrc] = useState<string | null>(null);
  const [alt, setAlt] = useState('');

  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      if (!(target instanceof HTMLImageElement)) {
        return;
      }
      // Only content figures — not logos, icons or linked images.
      if (!target.closest('.markdown') || target.closest('a')) {
        return;
      }
      event.preventDefault();
      setSrc(target.currentSrc || target.src);
      setAlt(target.alt ?? '');
    };

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setSrc(null);
      }
    };

    document.addEventListener('click', onClick);
    window.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('click', onClick);
      window.removeEventListener('keydown', onKey);
    };
  }, []);

  useEffect(() => {
    document.body.style.overflow = src ? 'hidden' : '';
    return () => {
      document.body.style.overflow = '';
    };
  }, [src]);

  if (!src) {
    return null;
  }

  return (
    <div
      className="lightbox"
      role="dialog"
      aria-modal="true"
      aria-label={alt || 'Figure'}
      onClick={() => setSrc(null)}>
      <img src={src} alt={alt} />
      <span className="lightbox__hint">Click anywhere or press Esc to close</span>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Keyboard shortcuts
 * ------------------------------------------------------------------ */

type Shortcut = {keys: string[]; label: string};

const SHORTCUTS: {group: string; items: Shortcut[]}[] = [
  {
    group: 'Search',
    items: [
      {keys: ['/'], label: 'Focus search'},
      {keys: ['⌘', 'K'], label: 'Open search'},
    ],
  },
  {
    group: 'Go to',
    items: [
      {keys: ['g', 'h'], label: 'Home'},
      {keys: ['g', 'e'], label: 'Explore all notes'},
      {keys: ['g', 'r'], label: 'Roadmap'},
      {keys: ['g', 'c'], label: 'Cheatsheets'},
      {keys: ['g', 'i'], label: 'Interview prep'},
    ],
  },
  {
    group: 'View',
    items: [
      {keys: ['t'], label: 'Toggle light / dark'},
      {keys: ['?'], label: 'Show this dialog'},
      {keys: ['Esc'], label: 'Close'},
    ],
  },
];

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName;
  return (
    tag === 'INPUT' ||
    tag === 'TEXTAREA' ||
    tag === 'SELECT' ||
    target.isContentEditable
  );
}

function focusSearch() {
  const input = document.querySelector<HTMLInputElement>('.navbar__search-input');
  if (input) {
    input.focus();
    input.select();
    return;
  }
  document
    .querySelector<HTMLElement>('[class*="searchButton"], .DocSearch-Button')
    ?.click();
}

function toggleColorMode() {
  // Root renders above the color-mode provider, so drive the navbar control.
  document
    .querySelector<HTMLElement>(
      '.navbar [class*="colorModeToggle"] button, .navbar button[class*="toggleButton"]',
    )
    ?.click();
}

function ShortcutsDialog({open, onClose}: {open: boolean; onClose: () => void}) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (open) {
      closeRef.current?.focus();
    }
  }, [open]);

  if (!open) {
    return null;
  }

  return (
    <div
      className={styles.overlay}
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard shortcuts"
      onClick={onClose}>
      <div className={styles.dialog} onClick={(event) => event.stopPropagation()}>
        <div className={styles.dialogHead}>
          <h2 className={styles.dialogTitle}>Keyboard shortcuts</h2>
          <button
            ref={closeRef}
            type="button"
            className={styles.dialogClose}
            onClick={onClose}
            aria-label="Close">
            ✕
          </button>
        </div>
        <div className={styles.dialogBody}>
          {SHORTCUTS.map(({group, items}) => (
            <section key={group} className={styles.shortcutGroup}>
              <h3 className={styles.shortcutGroupTitle}>{group}</h3>
              <ul className={styles.shortcutList}>
                {items.map(({keys, label}) => (
                  <li key={label} className={styles.shortcutRow}>
                    <span>{label}</span>
                    <span className={styles.keys}>
                      {keys.map((key) => (
                        <kbd key={key} className={styles.key}>
                          {key}
                        </kbd>
                      ))}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Root chrome
 * ------------------------------------------------------------------ */

export default function SiteChrome() {
  const location = useLocation();
  const history = useHistory();
  const [helpOpen, setHelpOpen] = useState(false);
  const docsBase = useBaseUrl('/docs/');
  const isDoc = location.pathname.startsWith(docsBase);

  const go = useCallback((to: string) => history.push(to), [history]);

  // Remember the last note read, for the homepage "continue" card.
  useEffect(() => {
    if (!isDoc) {
      return undefined;
    }
    const timer = window.setTimeout(() => {
      const heading = document.querySelector('.markdown h1, article h1');
      const title = heading?.textContent?.trim() || document.title.split('|')[0].trim();
      if (title) {
        recordVisit(location.pathname, title);
      }
    }, 400);
    return () => window.clearTimeout(timer);
  }, [isDoc, location.pathname]);

  useEffect(() => {
    let pendingChord = '';
    let chordTimer = 0;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setHelpOpen(false);
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey || isTypingTarget(event.target)) {
        return;
      }

      if (pendingChord === 'g') {
        const targets: Record<string, string> = {
          h: '/',
          e: '/explore',
          r: '/docs/intro',
          c: '/docs/category/cheetsheet',
          i: '/docs/category/interview',
        };
        const target = targets[event.key.toLowerCase()];
        pendingChord = '';
        window.clearTimeout(chordTimer);
        if (target) {
          event.preventDefault();
          go(target);
        }
        return;
      }

      switch (event.key) {
        case '/':
          event.preventDefault();
          focusSearch();
          break;
        case '?':
          event.preventDefault();
          setHelpOpen((open) => !open);
          break;
        case 't':
        case 'T':
          event.preventDefault();
          toggleColorMode();
          break;
        case 'g':
        case 'G':
          pendingChord = 'g';
          chordTimer = window.setTimeout(() => {
            pendingChord = '';
          }, 1200);
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.clearTimeout(chordTimer);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [go]);

  return (
    <>
      <ReadingProgress active={isDoc} />
      <Lightbox />
      <ShortcutsDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
      <button
        type="button"
        className={styles.helpButton}
        onClick={() => setHelpOpen(true)}
        aria-label="Keyboard shortcuts">
        <kbd className={styles.key}>?</kbd>
      </button>
    </>
  );
}
