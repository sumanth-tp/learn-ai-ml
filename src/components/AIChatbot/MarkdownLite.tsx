import DOMPurify from 'dompurify';
import {marked} from 'marked';
import {useEffect, useMemo, useRef} from 'react';

import styles from './styles.module.css';

const CODE_LANGUAGE_ALIASES: Record<string, string> = {
  bash: 'bash',
  c: 'c',
  'c++': 'cpp',
  cpp: 'cpp',
  csharp: 'csharp',
  css: 'css',
  go: 'go',
  html: 'html',
  java: 'java',
  javascript: 'javascript',
  js: 'javascript',
  json: 'json',
  plaintext: 'text',
  py: 'python',
  python: 'python',
  r: 'r',
  rust: 'rust',
  sh: 'bash',
  shell: 'bash',
  sql: 'sql',
  text: 'text',
  ts: 'typescript',
  typescript: 'typescript',
  yaml: 'yaml',
  yml: 'yaml',
};

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
    // Some models encode multiline snippets inside a GFM table as
    // `python<br>print(...)`. Backticks intentionally escape HTML, so repair
    // that narrow shape into a real code block after the HTML is sanitised.
    root?.querySelectorAll<HTMLElement>('td code, th code').forEach((inlineCode) => {
      const source = inlineCode.textContent ?? '';
      if (!/<br\s*\/?>/i.test(source)) return;

      const lines = source.split(/<br\s*\/?>/i);
      const language = CODE_LANGUAGE_ALIASES[lines[0]?.trim().toLowerCase()];
      const code = document.createElement('code');
      code.textContent = (language ? lines.slice(1) : lines).join('\n').trim();
      if (language) code.className = `language-${language}`;

      const pre = document.createElement('pre');
      if (language) pre.dataset.language = language;
      pre.appendChild(code);
      inlineCode.replaceWith(pre);
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
