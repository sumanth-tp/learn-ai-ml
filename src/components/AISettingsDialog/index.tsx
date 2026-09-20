import {useEffect, useState} from 'react';

import {
  clearAISettings,
  DEFAULT_AI_SETTINGS,
  loadAISettings,
  saveAISettings,
  type AISettings,
} from '@site/src/lib/aiSettings';

import styles from './styles.module.css';

type Props = {
  open: boolean;
  onClose: () => void;
  onSave?: (settings: AISettings) => void;
};

export default function AISettingsDialog({open, onClose, onSave}: Props) {
  const [settings, setSettings] = useState<AISettings>(DEFAULT_AI_SETTINGS);
  const [showSecrets, setShowSecrets] = useState(false);

  useEffect(() => {
    if (open) setSettings(loadAISettings());
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const update = (patch: Partial<AISettings>) => setSettings((current) => ({...current, ...patch}));

  return (
    <div className={styles.overlay} role="dialog" aria-modal="true" aria-labelledby="ai-settings-title" onMouseDown={onClose}>
      <section className={styles.dialog} onMouseDown={(event) => event.stopPropagation()}>
        <header className={styles.head}>
          <div>
            <span className={styles.eyebrow}>Browser-only configuration</span>
            <h2 id="ai-settings-title" className={styles.title}>AI settings</h2>
          </div>
          <button type="button" className={styles.close} onClick={onClose} aria-label="Close settings">✕</button>
        </header>

        <div className={styles.body}>
          <div className={styles.notice}>
            Keys are sent directly from this browser to their API provider. They are never sent to this site, but any key used in frontend code can be inspected. Use restricted, low-quota keys.
          </div>

          <label className={styles.field}>
            <span>Gemini API key <b>Required for AI answers</b></span>
            <input
              type={showSecrets ? 'text' : 'password'}
              value={settings.geminiApiKey}
              onChange={(event) => update({geminiApiKey: event.target.value})}
              placeholder="AIza…"
              autoComplete="off"
            />
          </label>

          <label className={styles.field}>
            <span>Gemini model</span>
            <input
              type="text"
              value={settings.geminiModel}
              onChange={(event) => update({geminiModel: event.target.value})}
              placeholder="gemini-3.6-flash"
            />
          </label>

          <div className={styles.optionalHead}>Groq fallback</div>
          <p className={styles.hint}>If Gemini is missing or fails, the tutor and news editor automatically retry with Groq.</p>

          <label className={styles.field}>
            <span>Groq API key <b>Fallback</b></span>
            <input
              type={showSecrets ? 'text' : 'password'}
              value={settings.groqApiKey}
              onChange={(event) => update({groqApiKey: event.target.value})}
              placeholder="gsk_…"
              autoComplete="off"
            />
          </label>

          <label className={styles.field}>
            <span>Groq model</span>
            <input
              type="text"
              value={settings.groqModel}
              onChange={(event) => update({groqModel: event.target.value})}
              placeholder="openai/gpt-oss-20b"
            />
          </label>

          <div className={styles.optionalHead}>Optional discovery keys</div>
          <p className={styles.hint}>Everything works without these. Add a YouTube key to use Google’s official, more reliable search instead of the community fallback, or a GitHub token for higher rate limits.</p>

          <label className={styles.field}>
            <span>YouTube Data API key <b>Optional</b></span>
            <input
              type={showSecrets ? 'text' : 'password'}
              value={settings.youtubeApiKey}
              onChange={(event) => update({youtubeApiKey: event.target.value})}
              placeholder="Optional"
              autoComplete="off"
            />
          </label>

          <label className={styles.field}>
            <span>GitHub fine-grained token</span>
            <input
              type={showSecrets ? 'text' : 'password'}
              value={settings.githubToken}
              onChange={(event) => update({githubToken: event.target.value})}
              placeholder="Optional"
              autoComplete="off"
            />
          </label>

          <div className={styles.checks}>
            <label><input type="checkbox" checked={showSecrets} onChange={(event) => setShowSecrets(event.target.checked)} /> Show keys</label>
            <label><input type="checkbox" checked={settings.remember} onChange={(event) => update({remember: event.target.checked})} /> Remember on this device</label>
          </div>
          <p className={styles.storageHint}>
            {settings.remember ? 'Saved in localStorage until you clear it.' : 'Saved in sessionStorage and cleared when the browser session ends.'}
          </p>
        </div>

        <footer className={styles.actions}>
          <button
            type="button"
            className={styles.clear}
            onClick={() => {
              clearAISettings();
              setSettings(DEFAULT_AI_SETTINGS);
              onSave?.(DEFAULT_AI_SETTINGS);
            }}>
            Clear keys
          </button>
          <div className={styles.actionRight}>
            <button type="button" className={styles.cancel} onClick={onClose}>Cancel</button>
            <button
              type="button"
              className={styles.save}
              onClick={() => {
                const cleaned = {
                  ...settings,
                  geminiApiKey: settings.geminiApiKey.trim(),
                  geminiModel: settings.geminiModel.trim() || DEFAULT_AI_SETTINGS.geminiModel,
                  groqApiKey: settings.groqApiKey.trim(),
                  groqModel: settings.groqModel.trim() || DEFAULT_AI_SETTINGS.groqModel,
                  youtubeApiKey: settings.youtubeApiKey.trim(),
                  githubToken: settings.githubToken.trim(),
                };
                saveAISettings(cleaned);
                onSave?.(cleaned);
                onClose();
              }}>
              Save settings
            </button>
          </div>
        </footer>
      </section>
    </div>
  );
}
