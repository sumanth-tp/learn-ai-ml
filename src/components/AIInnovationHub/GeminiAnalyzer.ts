import type {AISettings} from '@site/src/lib/aiSettings';
import {generateWithAI} from '@site/src/lib/gemini';

import type {Discovery, RawDiscovery} from './types';

function fallbackWhy(item: RawDiscovery): string {
  const metric = item.metrics?.[0];
  const shortTitle = item.title.length > 105 ? `${item.title.slice(0, 102).replace(/\s+\S*$/, '')}…` : item.title;
  switch (item.category) {
    case 'papers': {
      const lower = item.title.toLowerCase();
      if (/survey|review|taxonomy/.test(lower)) {
        return `“${shortTitle}” can compress a scattered research area into a useful map. Check its inclusion criteria and publication cut-off before treating that map as complete.`;
      }
      if (/agent|agentic|multi-agent/.test(lower)) {
        return `“${shortTitle}” addresses agent behaviour, where planning quality, tool errors and evaluation design often decide whether a system works outside a demo.`;
      }
      if (/framework|system|architecture/.test(lower)) {
        return `“${shortTitle}” proposes a reusable structure rather than a single result. Its practical value depends on the assumptions, comparison baselines and implementation evidence.`;
      }
      return `“${shortTitle}” adds current evidence to the field. The important check is whether its dataset, baselines and evaluation match the conditions you care about${metric ? `; it has ${metric}` : ''}.`;
    }
    case 'models': {
      const pipeline = item.details?.pipeline?.replace(/-/g, ' ') || 'AI';
      const base = item.details?.baseModel;
      const licence = item.details?.licence;
      const usage = item.metrics?.find((value) => /downloads|likes/.test(value));
      return `${base ? `As a ${pipeline} derivative of ${base},` : `As a new ${pipeline} release,`} ${shortTitle} is relevant for comparing deployable alternatives${licence ? ` under its ${licence.toUpperCase()} licence` : ''}. Check the training data and evaluation results before choosing it over the base model${usage ? `; early usage is ${usage}` : ''}.`;
    }
    case 'tools': {
      const lower = item.description.toLowerCase();
      const impact = /documentation|docs|wiki|codebase/.test(lower)
        ? 'It could reduce documentation drift and help new contributors understand a codebase faster.'
        : /mcp|model context protocol/.test(lower)
          ? 'It can make an external capability directly usable by MCP-compatible agents and assistants.'
          : /agent/.test(lower)
            ? 'It may remove orchestration work when prototyping or operating an agent workflow.'
            : /search|retriev|rag/.test(lower)
              ? 'It could improve how an application finds and grounds information before generation.'
              : 'Its value is practical: you can inspect the implementation and test it against a real workflow.';
      const adoption = item.details?.stars
        ? ` ${item.details.stars.toLocaleString()} stars show strong interest, though not production readiness.`
        : '';
      return `${impact}${adoption}`;
    }
    case 'videos': {
      const channel = item.source.replace(' · community index', '');
      const minutes = item.details?.durationSeconds
        ? Math.max(1, Math.round(item.details.durationSeconds / 60))
        : null;
      const traction = item.details?.views
        ? ` Its ${item.details.views.toLocaleString()} views indicate interest, not accuracy, so verify claims against the linked primary source.`
        : ' Use it to orient yourself, then verify technical claims against the primary source.';
      return `${minutes ? `This ${minutes}-minute walkthrough` : 'This walkthrough'} from ${channel} is a quick way to assess “${shortTitle}” before committing to a deeper read.${traction}`;
    }
  }
}

function fallbackUseCases(item: RawDiscovery): string[] {
  const text = `${item.title} ${item.description}`.toLowerCase();
  switch (item.category) {
    case 'papers':
      if (/survey|review|taxonomy/.test(text)) return ['Building a learning roadmap', 'Finding methods and baselines'];
      if (/agent|agentic|multi-agent/.test(text)) return ['Designing agent workflows', 'Planning agent evaluations'];
      if (/framework|system|architecture/.test(text)) return ['Comparing system designs', 'Implementing a research prototype'];
      return ['Literature review', 'Reproducing or extending the experiment'];
    case 'models': {
      const pipeline = item.details?.pipeline || '';
      if (/text-generation|conversational/.test(pipeline)) return ['Chat or RAG prototypes', 'Fine-tuning and model evaluation'];
      if (/image|vision/.test(pipeline)) return ['Vision experiments', 'Transfer-learning prototypes'];
      if (/audio|speech/.test(pipeline)) return ['Speech or audio prototypes', 'Domain-specific evaluation'];
      return ['Model comparison', 'Task-specific experimentation'];
    }
    case 'tools':
      if (/documentation|docs|wiki|codebase/.test(text)) return ['Generating codebase documentation', 'Developer and agent onboarding'];
      if (/mcp|model context protocol/.test(text)) return ['Connecting tools to AI assistants', 'Building MCP agent prototypes'];
      if (/agent/.test(text)) return ['Prototyping agent workflows', 'Testing orchestration patterns'];
      if (/search|retriev|rag/.test(text)) return ['Building grounded assistants', 'Evaluating retrieval quality'];
      return ['Rapid prototyping', 'Studying a working implementation'];
    case 'videos':
      if (/tutorial|how to|build|hands-on/.test(text)) return ['Following a hands-on implementation', 'Adapting the workflow for a small project'];
      if (/local|computer|laptop|offline/.test(text)) return ['Planning a local-AI setup', 'Comparing hardware and model options'];
      return ['Getting a quick conceptual overview', 'Deciding whether to study the primary source'];
  }
}

const fallback = (item: RawDiscovery): Discovery => ({
  ...item,
  simpleExplanation: item.description,
  whyImportant: fallbackWhy(item),
  useCases: fallbackUseCases(item),
  prerequisites: [],
  projectIdeas: [],
  aiEdited: false,
});

type Edited = {
  sourceId: string;
  simpleExplanation?: string;
  whyImportant?: string;
  useCases?: string[];
  prerequisites?: string[];
  projectIdeas?: string[];
};

function parseEditorResponse(text: string): {items?: Edited[]} {
  const withoutFence = text
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '');
  try {
    return JSON.parse(withoutFence) as {items?: Edited[]};
  } catch {
    // Loose Groq retries sometimes add one sentence around otherwise valid
    // JSON. Recover the outer object rather than discarding the whole batch.
    const start = withoutFence.indexOf('{');
    const end = withoutFence.lastIndexOf('}');
    if (start >= 0 && end > start) {
      return JSON.parse(withoutFence.slice(start, end + 1)) as {items?: Edited[]};
    }
    throw new Error('The AI editor returned text instead of JSON.');
  }
}

async function analyseBatch(
  items: RawDiscovery[],
  settings: AISettings,
  signal?: AbortSignal,
): Promise<Discovery[]> {
  const compact = items.map(({id, category, title, description, source, publishedAt, metrics, details}) => ({
    id, category, title, description, source, publishedAt, metrics, details,
  }));
  const response = await generateWithAI(
    settings,
    {
      systemInstruction: 'You are a careful AI news editor. Work only from the supplied records. Do not invent facts, benchmarks, links or dates. Return one valid JSON object only, with no Markdown fence or commentary.',
      messages: [{
        role: 'user',
        parts: [{text: `Edit every record for an AI/ML learner. Keep simpleExplanation under 45 words and whyImportant under 35 words. Add exactly 2 concrete short use cases answering "what can I use this for?" Add at most 2 short prerequisites and 2 feasible project ideas. Preserve each id exactly. Return this shape: {"items":[{"sourceId":"...","simpleExplanation":"...","whyImportant":"...","useCases":["...","..."],"prerequisites":["..."],"projectIdeas":["..."]}]}.\n\nRECORDS:\n${JSON.stringify(compact)}`}],
      }],
      temperature: 0.1,
      responseJson: true,
    },
    signal,
  );

  const parsed = parseEditorResponse(response.text);
  if (!Array.isArray(parsed.items)) throw new Error('The AI editor JSON did not contain an items array.');
  const byId = new Map(parsed.items.map((item) => [item.sourceId, item]));
  return items.map((item) => {
    const edit = byId.get(item.id);
    if (!edit) return fallback(item);
    return {
      ...item,
      simpleExplanation: edit.simpleExplanation?.trim() || item.description,
      whyImportant: edit.whyImportant?.trim() || fallback(item).whyImportant,
      useCases: (edit.useCases?.filter(Boolean) ?? fallbackUseCases(item)).slice(0, 2),
      prerequisites: (edit.prerequisites ?? []).filter(Boolean).slice(0, 2),
      projectIdeas: (edit.projectIdeas ?? []).filter(Boolean).slice(0, 2),
      aiEdited: true,
    };
  });
}

export async function analyseDiscoveries(
  items: RawDiscovery[],
  settings: AISettings,
  signal?: AbortSignal,
): Promise<Discovery[]> {
  if ((!settings.geminiApiKey && !settings.groqApiKey) || items.length === 0) return items.map(fallback);

  const BATCH_SIZE = 8;
  const discoveries: Discovery[] = [];
  const errors: string[] = [];
  for (let start = 0; start < items.length; start += BATCH_SIZE) {
    const batch = items.slice(start, start + BATCH_SIZE);
    try {
      discoveries.push(...await analyseBatch(batch, settings, signal));
    } catch (error) {
      if ((error as Error).name === 'AbortError') throw error;
      errors.push((error as Error).message);
      discoveries.push(...batch.map(fallback));
    }
  }

  if (!discoveries.some((item) => item.aiEdited) && errors.length > 0) {
    throw new Error(errors[0]);
  }
  return discoveries;
}
