import DOMPurify from 'dompurify';
import {marked} from 'marked';
import {useEffect, useMemo, useRef} from 'react';

import styles from './styles.module.css';

/** Render model output as sanitised GitHub-flavoured Markdown. */
export default function MarkdownLite({children}: {children: string}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const html = useMemo(() => {
    if (typeof window === 'undefined') return '';
    const rendered = marked.parse(children, {
      async: false,
      breaks: true,
      gfm: true,
    }) as string;
    return DOMPurify.sanitize(rendered, {USE_PROFILES: {html: true}});
  }, [children]);

  useEffect(() => {
    const root = rootRef.current;
    root?.querySelectorAll<HTMLAnchorElement>('a[href]').forEach((link) => {
      if (link.origin !== window.location.origin) {
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
      }
    });
    root?.querySelectorAll<HTMLTableElement>('table').forEach((table) => {
      if (table.parentElement?.classList.contains(styles.tableScroll)) return;
      const wrapper = document.createElement('div');
      wrapper.className = styles.tableScroll;
      table.parentNode?.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
    root?.querySelectorAll<HTMLElement>('pre > code').forEach((code) => {
      const language = [...code.classList]
        .find((name) => name.startsWith('language-'))
        ?.replace('language-', '');
      if (language && code.parentElement) code.parentElement.dataset.language = language;
    });
  }, [html]);

  return (
    <div
      ref={rootRef}
      className={styles.markdown}
      // Model HTML is sanitised above before it reaches the DOM.
      dangerouslySetInnerHTML={{__html: html}}
    />
  );
}
