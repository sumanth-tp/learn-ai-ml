export type PageContext = {
  title: string;
  url: string;
  content: string;
  truncated: boolean;
  headings: string[];
};

const MAX_CONTEXT_CHARS = 36_000;

export function extractCurrentPageContext(): PageContext {
  const article =
    document.querySelector<HTMLElement>('.theme-doc-markdown.markdown') ??
    document.querySelector<HTMLElement>('article .markdown') ??
    document.querySelector<HTMLElement>('main article') ??
    document.querySelector<HTMLElement>('main');
  const heading = article?.querySelector<HTMLElement>('h1')?.innerText.replace(/\s+/g, ' ').trim();
  const raw = article?.innerText?.replace(/\n{3,}/g, '\n\n').trim() ?? '';
  const truncated = raw.length > MAX_CONTEXT_CHARS;
  const headings = [...(article?.querySelectorAll<HTMLElement>('h2') ?? [])]
    .map((element) => element.innerText.replace(/[\u200b#]/g, '').replace(/\s+/g, ' ').trim())
    .filter((text) => text.length > 2 && text.length < 70 && !/^(checklist|recap|summary|what comes next|references)$/i.test(text));

  return {
    title: heading || document.title.split('|')[0].trim() || 'Current note',
    url: window.location.href.split('#')[0],
    content: truncated ? `${raw.slice(0, MAX_CONTEXT_CHARS)}\n\n[Page context truncated]` : raw,
    truncated,
    headings,
  };
}
