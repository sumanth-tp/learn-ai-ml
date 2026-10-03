import clsx from 'clsx';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import {useWindowSize} from '@docusaurus/theme-common';
import DocBreadcrumbs from '@theme/DocBreadcrumbs';
import ContentVisibility from '@theme/ContentVisibility';
import DocItemContent from '@theme/DocItem/Content';
import DocItemFooter from '@theme/DocItem/Footer';
import DocItemPaginator from '@theme/DocItem/Paginator';
import DocItemTOCDesktop from '@theme/DocItem/TOC/Desktop';
import DocItemTOCMobile from '@theme/DocItem/TOC/Mobile';
import DocVersionBadge from '@theme/DocVersionBadge';
import DocVersionBanner from '@theme/DocVersionBanner';
import type {Props} from '@theme/DocItem/Layout';
import {useEffect, useRef, useState, type ReactNode} from 'react';

import styles from './styles.module.css';

const TOC_STORAGE_KEY = 'learn-ai-ml-toc-collapsed';

export default function DocItemLayout({children}: Props): ReactNode {
  const {frontMatter, metadata, toc} = useDoc();
  const windowSize = useWindowSize();
  const [tocCollapsed, setTocCollapsed] = useState(false);
  const tocBodyRef = useRef<HTMLDivElement>(null);

  // Start with the same markup on the server and first client render.
  // Restore the reader's choice only after hydration.
  useEffect(() => {
    try {
      setTocCollapsed(window.localStorage.getItem(TOC_STORAGE_KEY) === 'true');
    } catch {
      // Private browsing can block storage; the control still works this visit.
    }
  }, []);

  const canRenderTOC = !frontMatter.hide_table_of_contents && toc.length > 0;
  const desktopTOC = canRenderTOC && (windowSize === 'desktop' || windowSize === 'ssr');

  useEffect(() => {
    const container = tocBodyRef.current;
    if (!container || !desktopTOC || tocCollapsed) return;

    let lastActiveLink: HTMLAnchorElement | null = null;
    let frame = 0;

    const revealActiveLink = () => {
      const activeLink = container.querySelector<HTMLAnchorElement>('.table-of-contents__link--active');
      if (!activeLink || container.scrollHeight <= container.clientHeight) return;

      const bounds = container.getBoundingClientRect();
      const linkBounds = activeLink.getBoundingClientRect();
      // Keep the section being read near the middle of the contents pane.
      // Scroll this container directly so the article stays in place.
      const target = container.scrollTop + linkBounds.top - bounds.top - container.clientTop
        - (container.clientHeight - linkBounds.height) / 2;
      const top = Math.max(0, Math.min(target, container.scrollHeight - container.clientHeight));

      if (Math.abs(top - container.scrollTop) > 1) {
        container.scrollTo({
          top,
          behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
        });
      }
    };

    const scheduleReveal = (force = false) => {
      const activeLink = container.querySelector<HTMLAnchorElement>('.table-of-contents__link--active');
      if (!force && activeLink === lastActiveLink) return;
      lastActiveLink = activeLink;
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(revealActiveLink);
    };

    const observer = new MutationObserver(() => scheduleReveal());
    observer.observe(container, {subtree: true, attributes: true, attributeFilter: ['class'], childList: true});
    const resizeObserver = new ResizeObserver(() => scheduleReveal(true));
    resizeObserver.observe(container);
    scheduleReveal(true);

    return () => {
      observer.disconnect();
      resizeObserver.disconnect();
      window.cancelAnimationFrame(frame);
    };
  }, [desktopTOC, tocCollapsed, metadata.permalink]);

  function toggleTOC() {
    const next = !tocCollapsed;
    setTocCollapsed(next);
    try {
      window.localStorage.setItem(TOC_STORAGE_KEY, String(next));
    } catch {
      // The current page can still collapse or expand the contents.
    }
  }

  return (
    <div className="row">
      <div
        className={clsx(
          'col',
          canRenderTOC && styles.docItemCol,
          desktopTOC && tocCollapsed && styles.docItemColExpanded,
        )}>
        <ContentVisibility metadata={metadata} />
        <DocVersionBanner />
        <div className={styles.docItemContainer}>
          <article>
            <DocBreadcrumbs />
            <DocVersionBadge />
            {canRenderTOC && <DocItemTOCMobile />}
            <DocItemContent>{children}</DocItemContent>
            <DocItemFooter />
          </article>
          <DocItemPaginator />
        </div>
      </div>

      {desktopTOC && (
        <div className={clsx('col', styles.tocColumn, tocCollapsed ? styles.tocColumnCollapsed : 'col--3')}>
          <div className={styles.tocShell}>
            <div className={clsx(styles.tocHeader, tocCollapsed && styles.tocHeaderCollapsed)}>
              {!tocCollapsed && <span className={styles.tocHeading}>On this page</span>}
              <button
                type="button"
                className={clsx(styles.tocToggle, tocCollapsed && styles.tocToggleCollapsed)}
                onClick={toggleTOC}
                aria-label={tocCollapsed ? 'Show right contents pane' : 'Hide right contents pane'}
                aria-expanded={!tocCollapsed}
                aria-controls="doc-right-contents">
                <svg viewBox="0 0 20 20" aria-hidden="true">
                  <path d={tocCollapsed ? 'm7 4 6 6-6 6' : 'm13 4-6 6 6 6'} />
                </svg>
                <span>{tocCollapsed ? 'Show contents' : 'Collapse'}</span>
              </button>
            </div>
            <div id="doc-right-contents" ref={tocBodyRef} className={styles.tocBody} hidden={tocCollapsed}>
              {!tocCollapsed && <DocItemTOCDesktop />}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
