export type AISettings = {
  geminiApiKey: string;
  geminiModel: string;
  groqApiKey: string;
  groqModel: string;
  youtubeApiKey: string;
  githubToken: string;
  remember: boolean;
};

export const DEFAULT_AI_SETTINGS: AISettings = {
  geminiApiKey: '',
  geminiModel: 'gemini-3.6-flash',
  groqApiKey: '',
  groqModel: 'openai/gpt-oss-20b',
  youtubeApiKey: '',
  githubToken: '',
  remember: false,
};

const STORAGE_KEY = 'learn-ai-ml:ai-settings:v1';

function storageFor(remember: boolean): Storage | null {
  if (typeof window === 'undefined') return null;
  return remember ? window.localStorage : window.sessionStorage;
}

export function loadAISettings(): AISettings {
  if (typeof window === 'undefined') return DEFAULT_AI_SETTINGS;

  for (const storage of [window.sessionStorage, window.localStorage]) {
    try {
      const raw = storage.getItem(STORAGE_KEY);
      if (raw) {
        const loaded = {...DEFAULT_AI_SETTINGS, ...JSON.parse(raw)};
        // Migrate the retired default while preserving intentional custom models.
        if (loaded.geminiModel === 'gemini-2.5-flash') {
          loaded.geminiModel = DEFAULT_AI_SETTINGS.geminiModel;
        }
        return loaded;
      }
    } catch {
      // Storage may be disabled. The UI still works for this page view.
    }
  }
  return DEFAULT_AI_SETTINGS;
}

export function saveAISettings(settings: AISettings): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
    window.sessionStorage.removeItem(STORAGE_KEY);
    storageFor(settings.remember)?.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Keep the in-memory form state when browser storage is unavailable.
  }
  window.dispatchEvent(new CustomEvent('learn-ai-ml:ai-settings', {detail: settings}));
}

export function clearAISettings(): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
    window.sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing else to clear.
  }
  window.dispatchEvent(
    new CustomEvent('learn-ai-ml:ai-settings', {detail: DEFAULT_AI_SETTINGS}),
  );
}
