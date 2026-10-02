import {useCallback, useEffect, useRef, useState} from 'react';

import AISettingsDialog from '@site/src/components/AISettingsDialog';
import {loadAISettings, type AISettings} from '@site/src/lib/aiSettings';
import {generateWithAI, type GeminiMessage} from '@site/src/lib/gemini';

import {extractCurrentPageContext, type PageContext} from './ContextExtractor';
import MarkdownLite from './MarkdownLite';
import styles from './styles.module.css';

type ChatMessage = {
  id: string;
  role: 'user' | 'model';
  text: string;
  quote?: string;
  provider?: string;
  fallbackReason?: string;
};

export const ASK_AI_EVENT = 'learn-ai-ml:ask-ai';

const STARTERS = [
  'Explain the main idea simply',
  'Give me a concrete example',
  'Quiz me on this page',
];

const QUOTE_STARTERS = [
  'Explain this passage simply',
  'Give me an example of this',
  'Why does this matter?',
];

const MAX_QUOTE_CHARS = 1_200;
const MAX_STORED_MESSAGES = 30;

function id() {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random()}`;
}

function storageKey(pageKey: string) {
  return `learn-ai-ml:chat:${pageKey}`;
}

function loadConversation(pageKey: string): ChatMessage[] {
  try {
    const raw = window.sessionStorage.getItem(storageKey(pageKey));
    const parsed = raw ? (JSON.parse(raw) as ChatMessage[]) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveConversation(pageKey: string, messages: ChatMessage[]) {
  try {
    if (messages.length === 0) {
      window.sessionStorage.removeItem(storageKey(pageKey));
    } else {
      window.sessionStorage.setItem(storageKey(pageKey), JSON.stringify(messages.slice(-MAX_STORED_MESSAGES)));
    }
  } catch {
    return;
  }
}

function promptText(message: ChatMessage) {
  return message.quote
    ? `About this passage from the page:\n"""\n${message.quote}\n"""\n\n${message.text}`
    : message.text;
}

function clip(text: string) {
  const clean = text.replace(/\s+/g, ' ').trim();
  return clean.length > MAX_QUOTE_CHARS ? `${clean.slice(0, MAX_QUOTE_CHARS)}…` : clean;
}

type SelectionPill = {top: number; left: number; text: string};

function useSelectionPill(enabled: boolean) {
  const [pill, setPill] = useState<SelectionPill | null>(null);

  useEffect(() => {
    if (!enabled) {
      setPill(null);
      return undefined;
    }
    let frame = 0;
    const update = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        const selection = window.getSelection();
        const text = selection?.toString().trim() ?? '';
        if (!selection || selection.rangeCount === 0 || text.length < 3) {
          setPill(null);
          return;
        }
        const range = selection.getRangeAt(0);
        const node = range.commonAncestorContainer;
        const element = node instanceof Element ? node : node.parentElement;
        if (!element?.closest('.theme-doc-markdown, article .markdown')) {
          setPill(null);
          return;
        }
        const rect = range.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) {
          setPill(null);
          return;
        }
        setPill({
          top: Math.max(8, rect.top - 44),
          left: Math.min(window.innerWidth - 190, Math.max(8, rect.left + rect.width / 2 - 90)),
          text: clip(text),
        });
      });
    };
    const hide = () => setPill(null);
    document.addEventListener('selectionchange', update);
    window.addEventListener('scroll', hide, {passive: true});
    window.addEventListener('resize', hide);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener('selectionchange', update);
      window.removeEventListener('scroll', hide);
      window.removeEventListener('resize', hide);
    };
  }, [enabled]);

  return [pill, setPill] as const;
}

function CopyButton({text, label = 'Copy'}: {text: string; label?: string}) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className={styles.action}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        } catch {
          setCopied(false);
        }
      }}>
      <svg viewBox="0 0 24 24" aria-hidden="true">
        {copied ? (
          <path d="m5 12.5 4.5 4.5L19 7.5" />
        ) : (
          <path d="M9 9h10v10H9zM5 15V5h10" />
        )}
      </svg>
      {copied ? 'Copied' : label}
    </button>
  );
}

export default function AIChatbot({pageKey}: {pageKey: string}) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<AISettings>(() => loadAISettings());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [quote, setQuote] = useState('');
  const [context, setContext] = useState<PageContext | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const loadedKeyRef = useRef('');
  const [pill, setPill] = useSelectionPill(pageKey.startsWith('/docs/'));

  useEffect(() => {
    abortRef.current?.abort();
    setLoading(false);
    setMessages(loadConversation(pageKey));
    loadedKeyRef.current = pageKey;
    setContext(null);
    setError('');
    setQuote('');
    setExpanded(false);
  }, [pageKey]);

  useEffect(() => {
    if (loadedKeyRef.current === pageKey) {
      saveConversation(pageKey, messages);
    }
  }, [messages, pageKey]);

  useEffect(() => {
    if (!open) return;
    const timer = window.setTimeout(() => {
      setContext(extractCurrentPageContext());
      inputRef.current?.focus();
    }, 80);
    return () => window.clearTimeout(timer);
  }, [open, pageKey]);

  useEffect(() => {
    endRef.current?.scrollIntoView({behavior: 'smooth', block: 'end'});
  }, [messages, loading]);

  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    const textarea = inputRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${textarea.scrollHeight}px`;
  }, [draft, open]);

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

  const openWithQuote = useCallback((text?: string) => {
    if (text) setQuote(clip(text));
    setOpen(true);
    window.setTimeout(() => inputRef.current?.focus(), 120);
  }, []);

  useEffect(() => {
    const onAsk = (event: Event) => {
      const detail = (event as CustomEvent<{quote?: string; toggle?: boolean}>).detail ?? {};
      if (detail.toggle && !detail.quote) {
        setOpen((current) => !current);
        return;
      }
      openWithQuote(detail.quote);
    };
    window.addEventListener(ASK_AI_EVENT, onAsk);
    return () => window.removeEventListener(ASK_AI_EVENT, onAsk);
  }, [openWithQuote]);

  async function ask(conversation: ChatMessage[]) {
    const liveContext = context ?? extractCurrentPageContext();
    setContext(liveContext);
    setError('');
    setMessages(conversation);
    setLoading(true);
    const controller = new AbortController();
    abortRef.current = controller;

    const history: GeminiMessage[] = conversation.slice(-9).map((message) => ({
      role: message.role,
      parts: [{text: message.role === 'user' ? promptText(message) : message.text}],
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
      setMessages((current) => [
        ...current,
        {id: id(), role: 'model', text: answer.text, provider: answer.provider, fallbackReason: answer.fallbackReason},
      ]);
    } catch (caught) {
      if ((caught as Error).name !== 'AbortError') {
        setError((caught as Error).message);
      }
    } finally {
      if (abortRef.current === controller) {
        setLoading(false);
        abortRef.current = null;
      }
    }
  }

  function send(text: string) {
    const question = text.trim();
    if (!question || loading) return;

    if (!settings.geminiApiKey && !settings.groqApiKey) {
      setDraft(question);
      setSettingsOpen(true);
      return;
    }

    const userMessage: ChatMessage = {id: id(), role: 'user', text: question, ...(quote ? {quote} : {})};
    setDraft('');
    setQuote('');
    void ask([...messages, userMessage]);
  }

  function regenerate() {
    if (loading) return;
    const lastUser = messages.map((message) => message.role).lastIndexOf('user');
    if (lastUser < 0) return;
    void ask(messages.slice(0, lastUser + 1));
  }

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  }

  function clearConversation() {
    stop();
    setMessages([]);
    setError('');
    setQuote('');
  }

  const lastModelIndex = messages.map((message) => message.role).lastIndexOf('model');
  const headingStarters = (context?.headings ?? []).slice(0, 2).map((heading) => `Explain “${heading}”`);
  const starters = quote ? QUOTE_STARTERS : [...STARTERS, ...headingStarters];

  return (
    <>
      {pill && !loading && (
        <button
          type="button"
          className={styles.selectionPill}
          style={{top: pill.top, left: pill.left}}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => {
            openWithQuote(pill.text);
            setPill(null);
          }}>
          <span aria-hidden="true">✦</span> Ask AI about this
        </button>
      )}

      {open && (
        <aside className={styles.panel} data-expanded={expanded} role="dialog" aria-label="AI tutor" aria-modal={expanded}>
          <header className={styles.head}>
            <div className={styles.identity}>
              <span className={styles.spark} aria-hidden="true">✦</span>
              <div><strong>AI Tutor</strong><span>{context?.title || 'Reading this page…'}</span></div>
            </div>
            <div className={styles.headActions}>
              {messages.length > 0 && (
                <button type="button" onClick={clearConversation} aria-label="Clear conversation" title="Clear conversation">
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></svg>
                </button>
              )}
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
              <button type="button" onClick={() => setSettingsOpen(true)} aria-label="AI settings" title="AI settings">
                <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z" /></svg>
              </button>
              <button type="button" onClick={() => {setOpen(false); setExpanded(false);}} aria-label="Close tutor" title="Close">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" /></svg>
              </button>
            </div>
          </header>

          <div className={styles.messages} aria-live="polite">
            {messages.length === 0 && (
              <div className={styles.welcome}>
                <span className={styles.welcomeIcon}>✦</span>
                <h3>{quote ? 'Ask about the passage' : 'Ask about this page'}</h3>
                <p>
                  {quote
                    ? 'Your selection is attached to the question, with the rest of the page as context.'
                    : 'I use the page you are reading as context. Select any text in the note to ask about just that part.'}
                </p>
                <div className={styles.starters}>
                  {starters.map((starter) => <button type="button" key={starter} onClick={() => send(starter)}>{starter}</button>)}
                </div>
              </div>
            )}
            {messages.map((message, index) => (
              <div key={message.id} className={styles.message} data-role={message.role}>
                {message.role === 'model' ? (
                  <>
                    <MarkdownLite>{message.text}</MarkdownLite>
                    <div className={styles.messageActions}>
                      <CopyButton text={message.text} />
                      {index === lastModelIndex && !loading && (
                        <button type="button" className={styles.action} onClick={regenerate}>
                          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6" /></svg>
                          Regenerate
                        </button>
                      )}
                      <span className={styles.provider} title={message.fallbackReason ? `Gemini failed: ${message.fallbackReason}` : undefined}>
                        {message.provider}
                        {message.fallbackReason ? ' · fallback' : ''}
                      </span>
                    </div>
                  </>
                ) : (
                  <>
                    {message.quote && <blockquote className={styles.quote}>{message.quote}</blockquote>}
                    <p>{message.text}</p>
                  </>
                )}
              </div>
            ))}
            {loading && (
              <div className={styles.thinking}>
                <i /><i /><i /><span>Reading the page and thinking</span>
              </div>
            )}
            {error && (
              <div className={styles.error} role="alert">
                {error}
                <div className={styles.errorActions}>
                  <button type="button" onClick={regenerate}>Try again</button>
                  <button type="button" onClick={() => setSettingsOpen(true)}>Check settings</button>
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>

          <form className={styles.composer} onSubmit={(event) => {event.preventDefault(); send(draft);}}>
            {quote && (
              <div className={styles.quoteChip}>
                <span className={styles.quoteChipLabel}>Asking about</span>
                <span className={styles.quoteChipText}>{quote}</span>
                <button type="button" onClick={() => setQuote('')} aria-label="Remove selected passage">
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" /></svg>
                </button>
              </div>
            )}
            <div className={styles.composerRow}>
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
                placeholder={quote ? 'Ask about the selected passage…' : 'Ask about this page…'}
                aria-label="Question for AI tutor"
              />
              {loading ? (
                <button type="button" className={styles.stopButton} onClick={stop} aria-label="Stop generating" title="Stop">
                  <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1.5" /></svg>
                </button>
              ) : (
                <button type="submit" disabled={!draft.trim()} aria-label="Send question" title="Send (Enter)">
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7" /></svg>
                </button>
              )}
            </div>
          </form>
          <footer className={styles.source}>
            <span>Context: <a href={context?.url || pageKey}>{context?.title || 'current page'}</a>{context?.truncated ? ' · long page shortened' : ''}</span>
            <span className={styles.hint}><kbd>Enter</kbd> send · <kbd>Shift</kbd>+<kbd>Enter</kbd> new line</span>
          </footer>
        </aside>
      )}

      <button
        type="button"
        className={styles.launcher}
        data-open={open}
        data-expanded={expanded}
        onClick={() => setOpen((current) => !current)}
        aria-label={open ? 'Close AI tutor' : 'Ask AI tutor'}
        aria-expanded={open}
        title={open ? 'Close AI tutor' : 'Ask AI (a)'}>
        <span aria-hidden="true">{open ? '✕' : '✦'}</span>
        {!open && <b>Ask AI</b>}
      </button>

      <AISettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} onSave={setSettings} />
    </>
  );
}
