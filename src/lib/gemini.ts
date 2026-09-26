import type {AISettings} from './aiSettings';

type GeminiPart = {text: string};
export type GeminiMessage = {
  role: 'user' | 'model';
  parts: GeminiPart[];
};

export type GenerateOptions = {
  systemInstruction: string;
  messages: GeminiMessage[];
  temperature?: number;
  responseJson?: boolean;
};

type GeminiResponse = {
  candidates?: Array<{
    content?: {parts?: GeminiPart[]};
    finishReason?: string;
  }>;
  promptFeedback?: {blockReason?: string};
  error?: {message?: string};
};

function readableGeminiError(status: number, body: GeminiResponse): string {
  const detail = body.error?.message;
  if (status === 400) return detail || 'Gemini rejected this request. Check the model name and API key.';
  if (status === 403) return detail || 'This Gemini key cannot use the selected model.';
  if (status === 429) return 'Gemini is rate-limited right now. Wait a moment and try again.';
  return detail || `Gemini request failed (${status}).`;
}

export async function generateWithGemini(
  settings: Pick<AISettings, 'geminiApiKey' | 'geminiModel'>,
  options: GenerateOptions,
  signal?: AbortSignal,
): Promise<string> {
  if (!settings.geminiApiKey.trim()) {
    throw new Error('Add a Gemini API key in AI settings first.');
  }

  const model = settings.geminiModel.trim() || 'gemini-3.6-flash';
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`,
    {
      method: 'POST',
      signal,
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': settings.geminiApiKey.trim(),
      },
      body: JSON.stringify({
        systemInstruction: {parts: [{text: options.systemInstruction}]},
        contents: options.messages,
        generationConfig: {
          temperature: options.temperature ?? 0.3,
          ...(options.responseJson ? {responseMimeType: 'application/json'} : {}),
        },
      }),
    },
  );

  const body = (await response.json().catch(() => ({}))) as GeminiResponse;
  if (!response.ok) throw new Error(readableGeminiError(response.status, body));

  const text = body.candidates?.[0]?.content?.parts
    ?.map((part) => part.text)
    .filter(Boolean)
    .join('\n')
    .trim();

  if (!text) {
    const reason = body.promptFeedback?.blockReason || body.candidates?.[0]?.finishReason;
    throw new Error(reason ? `Gemini returned no answer (${reason}).` : 'Gemini returned an empty answer.');
  }
  return text;
}

type GroqResponse = {
  choices?: Array<{message?: {content?: string}}>;
  error?: {message?: string; code?: string; failed_generation?: string};
};

const GROQ_PAGE_CHARS = 8_000;
const GROQ_HISTORY_CHARS = 3_600;
const GROQ_JSON_INPUT_CHARS = 8_000;

function relevantPageExcerpt(content: string, question: string, limit: number): string {
  if (content.length <= limit) return content;

  const words = new Set((question.toLowerCase().match(/[\p{L}\p{N}]{4,}/gu) ?? [])
    .filter((word) => !['about', 'explain', 'give', 'what', 'which', 'this', 'that', 'with', 'from'].includes(word)));
  const blocks = content.split(/\n{2,}/).map((text, index) => ({text, index}));
  const first = blocks.slice(0, 2);
  const scored = blocks.slice(2).map((block) => ({
    ...block,
    score: [...words].reduce((total, word) => total + (block.text.toLowerCase().includes(word) ? 1 : 0), 0),
  }));
  scored.sort((a, b) => b.score - a.score || a.index - b.index);

  const chosen = [...first];
  let remaining = limit - first.reduce((total, block) => total + Math.min(block.text.length, 1_200) + 2, 0);
  for (const block of scored) {
    if (remaining < 120) break;
    if (block.score === 0 && chosen.length >= 5) break;
    const text = block.text.slice(0, Math.min(block.text.length, remaining - 2, 1_500));
    chosen.push({text, index: block.index});
    remaining -= text.length + 2;
  }
  return chosen.sort((a, b) => a.index - b.index)
    .map((block) => block.text.slice(0, 1_200))
    .join('\n\n')
    .slice(0, limit);
}

function groqMessages(options: GenerateOptions): Array<{role: 'system' | 'user' | 'assistant'; content: string}> {
  const question = options.messages.at(-1)?.parts.map((part) => part.text).join('\n') ?? '';
  const marker = '\n\nPAGE CONTENT:\n';
  const markerAt = options.systemInstruction.indexOf(marker);
  const system = markerAt < 0
    ? options.systemInstruction.slice(0, GROQ_PAGE_CHARS)
    : `${options.systemInstruction.slice(0, markerAt + marker.length)}${relevantPageExcerpt(
      options.systemInstruction.slice(markerAt + marker.length), question, GROQ_PAGE_CHARS,
    )}`;

  const limit = options.responseJson ? GROQ_JSON_INPUT_CHARS : GROQ_HISTORY_CHARS;
  let remaining = limit;
  const selected: Array<{role: 'user' | 'assistant'; content: string}> = [];
  for (const message of [...options.messages].reverse()) {
    if (remaining <= 0) break;
    const full = message.parts.map((part) => part.text).join('\n');
    if (options.responseJson && full.length > limit) {
      throw new Error('The AI editor batch is too large. Reduce the number of records and retry.');
    }
    const content = full.slice(0, Math.min(full.length, remaining, message === options.messages.at(-1) ? limit : 1_200));
    selected.unshift({role: message.role === 'model' ? 'assistant' : 'user', content});
    remaining -= content.length;
  }
  return [{role: 'system', content: system}, ...selected];
}

async function generateWithGroq(
  settings: Pick<AISettings, 'groqApiKey' | 'groqModel'>,
  options: GenerateOptions,
  signal?: AbortSignal,
): Promise<string> {
  if (!settings.groqApiKey.trim()) throw new Error('Add a Groq API key in AI settings first.');
  const model = settings.groqModel.trim() || 'openai/gpt-oss-20b';
  const messages = groqMessages(options);
  const request = async (strictJson: boolean) => {
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      signal,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${settings.groqApiKey.trim()}`,
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: options.temperature ?? 0.3,
        max_completion_tokens: options.responseJson ? 2048 : 1536,
        ...(strictJson ? {response_format: {type: 'json_object'}} : {}),
      }),
    });
    const body = (await response.json().catch(() => ({}))) as GroqResponse;
    return {response, body};
  };

  // GPT-OSS occasionally fails Groq's server-side JSON validator even when it
  // produces recoverable JSON. Use prompt-enforced JSON for those models.
  const strictJson = Boolean(options.responseJson) && !model.startsWith('openai/gpt-oss');
  let {response, body} = await request(strictJson);
  const strictError = `${body.error?.message ?? ''} ${body.error?.code ?? ''}`;
  if (
    !response.ok &&
    strictJson &&
    response.status === 400 &&
    /json|validat|failed.?generation/i.test(strictError)
  ) {
    // Groq's server-side JSON validator can reject a useful generation. The
    // prompt still requires JSON, so retry once without strict response mode.
    ({response, body} = await request(false));
  }
  if (!response.ok) {
    const message = body.error?.message;
    if (response.status === 429) {
      throw new Error(/request too large|requested \d+/i.test(message ?? '')
        ? 'Groq rejected the prompt under this account’s token limit. Try a shorter question or a model with a higher limit.'
        : 'Groq has reached its token-per-minute limit. Wait a minute and try again.');
    }
    throw new Error(message || `Groq request failed (${response.status}).`);
  }
  const text = body.choices?.[0]?.message?.content?.trim();
  if (!text) throw new Error('Groq returned an empty answer.');
  return text;
}

export type AIResult = {text: string; provider: 'Gemini' | 'Groq'; fallbackReason?: string};

export async function generateWithAI(
  settings: AISettings,
  options: GenerateOptions,
  signal?: AbortSignal,
): Promise<AIResult> {
  if (!settings.geminiApiKey && !settings.groqApiKey) {
    throw new Error('Add a Gemini or Groq API key in AI settings first.');
  }

  let geminiError = '';
  if (settings.geminiApiKey) {
    try {
      return {text: await generateWithGemini(settings, options, signal), provider: 'Gemini'};
    } catch (error) {
      if ((error as Error).name === 'AbortError') throw error;
      geminiError = (error as Error).message;
      if (!settings.groqApiKey) throw error;
    }
  }

  try {
    return {
      text: await generateWithGroq(settings, options, signal),
      provider: 'Groq',
      ...(geminiError ? {fallbackReason: geminiError} : {}),
    };
  } catch (error) {
    if ((error as Error).name === 'AbortError') throw error;
    if (geminiError) {
      throw new Error(`Gemini failed: ${geminiError} Groq fallback failed: ${(error as Error).message}`);
    }
    throw error;
  }
}
