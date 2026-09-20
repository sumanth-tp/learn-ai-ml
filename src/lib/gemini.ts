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

async function generateWithGroq(
  settings: Pick<AISettings, 'groqApiKey' | 'groqModel'>,
  options: GenerateOptions,
  signal?: AbortSignal,
): Promise<string> {
  if (!settings.groqApiKey.trim()) throw new Error('Add a Groq API key in AI settings first.');
  const model = settings.groqModel.trim() || 'openai/gpt-oss-20b';
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
        messages: [
          {role: 'system', content: options.systemInstruction},
          ...options.messages.map((message) => ({
            role: message.role === 'model' ? 'assistant' : 'user',
            content: message.parts.map((part) => part.text).join('\n'),
          })),
        ],
        temperature: options.temperature ?? 0.3,
        max_completion_tokens: options.responseJson ? 8192 : 4096,
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
    if (response.status === 429) throw new Error('Groq is rate-limited right now.');
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
