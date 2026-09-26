import {useEffect, useRef, useState} from 'react';

import AISettingsDialog from '@site/src/components/AISettingsDialog';
import {loadAISettings, type AISettings} from '@site/src/lib/aiSettings';
import {generateWithAI, type GeminiMessage} from '@site/src/lib/gemini';

import {extractCurrentPageContext, type PageContext} from './ContextExtractor';
import MarkdownLite from './MarkdownLite';
import styles from './styles.module.css';

type ChatMessage = {id: string; role: 'user' | 'model'; text: string; provider?: string};

const STARTERS = [
  'Explain the main idea simply',
  'Give me a concrete example',
  'Quiz me on this page',
];

function id() {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random()}`;
}

export default function AIChatbot({pageKey}: {pageKey: string}) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<AISettings>(() => loadAISettings());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [context, setContext] = useState<PageContext | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setMessages([]);
    setContext(null);
    setError('');
    setExpanded(false);
    abortRef.current?.abort();
    setLoading(false);
  }, [pageKey]);

  useEffect(() => {
    if (!open) return;
    const timer = window.setTimeout(() => {
      setContext(extractCurrentPageContext());
      inputRef.current?.focus();
    }, 80);
    return () => window.clearTimeout(timer);
  }, [open, pageKey]);

  useEffect(() => {
    endRef.current?.scrollIntoView({behavior: 'smooth'});
  }, [messages, loading]);

  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    if (!open || !expanded) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setExpanded(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [expanded, open]);

  async function send(text: string) {
    const question = text.trim();
    if (!question || loading) return;
    const liveContext = context ?? extractCurrentPageContext();
    setContext(liveContext);
    setDraft('');
    setError('');

    if (!settings.geminiApiKey && !settings.groqApiKey) {
      setDraft(question);
      setSettingsOpen(true);
      return;
    }

    const userMessage: ChatMessage = {id: id(), role: 'user', text: question};
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setLoading(true);
    const controller = new AbortController();
    abortRef.current = controller;

    const history: GeminiMessage[] = nextMessages.slice(-9).map((message) => ({
      role: message.role,
      parts: [{text: message.text}],
    }));

    try {
      const answer = await generateWithAI(
        settings,
        {
          systemInstruction: `You are a patient AI tutor embedded in a learning website. Answer primarily from the supplied page. If the page does not contain enough information, say so clearly before giving brief general knowledge. Never invent a quote or source. Use valid GitHub-flavoured Markdown. Put code in fenced blocks with a language. Never put multiline code inside a Markdown table; place each code block after the table and refer to it by name. Format comparisons as proper Markdown tables with a header separator row. Prefer short sections, lists and concrete examples over dense paragraphs.\n\nPAGE TITLE: ${liveContext.title}\nPAGE URL: ${liveContext.url}\n\nPAGE CONTENT:\n${liveContext.content}`,
          messages: history,
          temperature: 0.25,
        },
        controller.signal,
      );
      setMessages((current) => [...current, {id: id(), role: 'model', text: answer.text, provider: answer.provider}]);
    } catch (caught) {
      if ((caught as Error).name !== 'AbortError') {
        setError((caught as Error).message);
        setDraft(question);
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  return (
    <>
      {open && (
        <aside className={styles.panel} data-expanded={expanded} role="dialog" aria-label="AI tutor" aria-modal={expanded}>
          <header className={styles.head}>
            <div className={styles.identity}>
              <span className={styles.spark} aria-hidden="true">✦</span>
              <div><strong>AI Tutor</strong><span>{context?.title || 'Reading this page…'}</span></div>
            </div>
            <div className={styles.headActions}>
              <button
                type="button"
                onClick={() => setExpanded((current) => !current)}
                aria-label={expanded ? 'Collapse AI tutor' : 'Expand AI tutor'}
                title={expanded ? 'Collapse tutor' : 'Expand tutor'}>
                {expanded ? (
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" /></svg>
                ) : (
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4H4v5M15 4h5v5M9 20H4v-5M15 20h5v-5" /></svg>
                )}
              </button>
              <button type="button" onClick={() => setSettingsOpen(true)} aria-label="AI settings" title="AI settings">⚙</button>
              <button type="button" onClick={() => {setOpen(false); setExpanded(false);}} aria-label="Close tutor">✕</button>
            </div>
          </header>

          <div className={styles.messages} aria-live="polite">
            {messages.length === 0 && (
              <div className={styles.welcome}>
                <span className={styles.welcomeIcon}>✦</span>
                <h3>Ask about this page</h3>
                <p>I use the visible page as context. Try an explanation, an example, or a quick knowledge check.</p>
                <div className={styles.starters}>
                  {STARTERS.map((starter) => <button type="button" key={starter} onClick={() => send(starter)}>{starter}</button>)}
                </div>
              </div>
            )}
            {messages.map((message) => (
              <div key={message.id} className={styles.message} data-role={message.role}>
                {message.role === 'model' ? <><MarkdownLite>{message.text}</MarkdownLite><span className={styles.provider}>{message.provider}</span></> : <p>{message.text}</p>}
              </div>
            ))}
            {loading && <div className={styles.thinking}><i /><i /><i /><span>Thinking from this page</span></div>}
            {error && <div className={styles.error} role="alert">{error}<button type="button" onClick={() => setSettingsOpen(true)}>Check settings</button></div>}
            <div ref={endRef} />
          </div>

          <form className={styles.composer} onSubmit={(event) => {event.preventDefault(); send(draft);}}>
            <textarea
              ref={inputRef}
              rows={1}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  send(draft);
                }
              }}
              placeholder="Ask about this page…"
              aria-label="Question for AI tutor"
            />
            <button type="submit" disabled={!draft.trim() || loading} aria-label="Send question">↑</button>
          </form>
          <footer className={styles.source}>Context: <a href={context?.url || pageKey}>{context?.title || 'current page'}</a>{context?.truncated ? ' · long page shortened' : ''}</footer>
        </aside>
      )}

      <button
        type="button"
        className={styles.launcher}
        data-open={open}
        data-expanded={expanded}
        onClick={() => setOpen((current) => !current)}
        aria-label={open ? 'Close AI tutor' : 'Ask AI tutor'}
        aria-expanded={open}>
        <span aria-hidden="true">{open ? '✕' : '✦'}</span>
        {!open && <b>Ask AI</b>}
      </button>

      <AISettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} onSave={setSettings} />
    </>
  );
}
