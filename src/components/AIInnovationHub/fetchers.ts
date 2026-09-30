import type {AISettings} from '@site/src/lib/aiSettings';

import type {RawDiscovery, SourceReport} from './types';
import {discoveryKeys} from './history';
import {sourceJSON} from './sourceRequest';

const CANDIDATE_LIMIT = 50;
export const ITEMS_PER_SOURCE = 10;
const VIDEO_DAYS = 90;

const cutoff = (days: number) => {
  const date = new Date();
  date.setUTCDate(date.getUTCDate() - days);
  return date.toISOString().slice(0, 10);
};

const normalise = (value = '') => value.replace(/\s+/g, ' ').trim();
const excerpt = (value = '', limit = 420) => {
  const text = normalise(value);
  return text.length > limit ? `${text.slice(0, limit).replace(/\s+\S*$/, '')}…` : text;
};

function reconstructAbstract(index?: Record<string, number[]>): string {
  if (!index) return '';
  const words: Array<[number, string]> = [];
  Object.entries(index).forEach(([word, positions]) => positions.forEach((position) => words.push([position, word])));
  return words.sort((a, b) => a[0] - b[0]).map(([, word]) => word).join(' ');
}

export async function fetchPapers(signal?: AbortSignal): Promise<RawDiscovery[]> {
  // arXiv's Atom endpoint does not expose CORS headers. OpenAlex is used as a
  // browser-safe scholarly index; original arXiv landing links are retained.
  const params = new URLSearchParams({
    search: 'large language models generative artificial intelligence agents',
    filter: `from_publication_date:${cutoff(30)},to_publication_date:${cutoff(0)},type:article|preprint`,
    sort: 'publication_date:desc',
    'per-page': String(CANDIDATE_LIMIT),
  });
  const body = await sourceJSON<{results?: Array<Record<string, any>>}>(`https://api.openalex.org/works?${params}`, 'OpenAlex', signal);
  return (body.results ?? []).map((work) => {
    const location = work.primary_location ?? {};
    const url = location.landing_page_url || work.doi || work.id;
    return {
      id: `paper:${work.id}`,
      category: 'papers' as const,
      title: normalise(work.display_name || work.title || 'Untitled paper'),
      url,
      source: String(url).includes('arxiv.org') ? 'arXiv via OpenAlex' : (location.source?.display_name || 'OpenAlex'),
      publishedAt: work.publication_date || '',
      description: excerpt(reconstructAbstract(work.abstract_inverted_index), 520) || 'Abstract unavailable from the public index.',
      authors: (work.authorships ?? []).slice(0, 4).map((entry: any) => entry.author?.display_name).filter(Boolean),
      metrics: typeof work.cited_by_count === 'number' && work.cited_by_count > 0 ? [`${work.cited_by_count.toLocaleString()} citations`] : [],
    };
  });
}

export async function fetchModels(signal?: AbortSignal): Promise<RawDiscovery[]> {
  const params = new URLSearchParams({filter: 'text-generation', sort: 'createdAt', direction: '-1', limit: String(CANDIDATE_LIMIT)});
  const models = await sourceJSON<Array<Record<string, any>>>(`https://huggingface.co/api/models?${params}`, 'Hugging Face', signal);
  return models.map((model) => {
    const tags = (model.tags ?? []) as string[];
    const baseModel = tags.find((tag) => tag.startsWith('base_model:') && !tag.startsWith('base_model:finetune:'))?.slice('base_model:'.length);
    const licence = model.cardData?.license || tags.find((tag) => tag.startsWith('license:'))?.slice('license:'.length);
    const pipeline = model.pipeline_tag || 'AI';
    const library = model.library_name || (tags.includes('transformers') ? 'transformers' : undefined);
    const facts = [
      baseModel ? `based on ${baseModel}` : '',
      library ? `built with ${library}` : '',
      licence ? `licensed ${licence}` : '',
    ].filter(Boolean).join(', ');
    return {
      id: `model:${model.id}`,
      category: 'models' as const,
      title: model.id,
      url: `https://huggingface.co/${model.id}`,
      source: 'Hugging Face',
      publishedAt: model.createdAt || model.lastModified || '',
      description: excerpt(model.cardData?.description || `A recently published ${String(pipeline).replace(/-/g, ' ')} model${facts ? `, ${facts}` : ''}.`),
      metrics: [
        typeof model.downloads === 'number' && model.downloads > 0 ? `${model.downloads.toLocaleString()} downloads` : '',
        typeof model.likes === 'number' && model.likes > 0 ? `${model.likes.toLocaleString()} likes` : '',
        licence ? licence.toUpperCase() : '',
      ].filter(Boolean),
      details: {pipeline, baseModel, licence, library},
    };
  });
}

export async function fetchTools(settings: AISettings, signal?: AbortSignal): Promise<RawDiscovery[]> {
  // Repository search applies date/star qualifiers reliably to plain OR terms;
  // topic-qualified OR groups can unexpectedly collapse to zero results.
  const query = `llm OR mcp OR agent created:>=${cutoff(90)} stars:>5`;
  const params = new URLSearchParams({q: query, sort: 'updated', order: 'desc', per_page: String(CANDIDATE_LIMIT)});
  const headers: HeadersInit = {
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
  };
  if (settings.githubToken) headers.Authorization = `Bearer ${settings.githubToken}`;
  const body = await sourceJSON<{items?: Array<Record<string, any>>}>(`https://api.github.com/search/repositories?${params}`, 'GitHub', signal, headers);
  return (body.items ?? []).map((repo) => ({
    id: `tool:${repo.id}`,
    category: 'tools' as const,
    title: repo.full_name,
    url: repo.html_url,
    source: 'GitHub',
    publishedAt: repo.created_at,
    description: excerpt(repo.description || 'A newly published open-source AI project.'),
    authors: repo.owner?.login ? [repo.owner.login] : [],
    metrics: [
      `${Number(repo.stargazers_count || 0).toLocaleString()} stars`,
      repo.language || '',
    ].filter(Boolean),
    details: {
      repoLanguage: repo.language || undefined,
      stars: Number(repo.stargazers_count || 0),
    },
  }));
}

function decodeEntities(value: string): string {
  const textarea = document.createElement('textarea');
  textarea.innerHTML = value;
  return textarea.value;
}

export async function fetchVideos(settings: AISettings, signal?: AbortSignal): Promise<RawDiscovery[]> {
  if (!settings.youtubeApiKey) return fetchCommunityVideos(signal);
  const params = new URLSearchParams({
    part: 'snippet',
    q: 'new AI model OR LLM paper OR AI agent explained',
    type: 'video',
    order: 'date',
    publishedAfter: `${cutoff(VIDEO_DAYS)}T00:00:00Z`,
    maxResults: String(CANDIDATE_LIMIT),
    relevanceLanguage: 'en',
    safeSearch: 'moderate',
    key: settings.youtubeApiKey,
  });
  const body = await sourceJSON<{items?: Array<Record<string, any>>}>(`https://www.googleapis.com/youtube/v3/search?${params}`, 'YouTube', signal);
  return (body.items ?? []).map((video) => ({
    id: `video:${video.id.videoId}`,
    category: 'videos' as const,
    title: decodeEntities(video.snippet.title),
    url: `https://www.youtube.com/watch?v=${video.id.videoId}`,
    source: decodeEntities(video.snippet.channelTitle || 'YouTube'),
    publishedAt: video.snippet.publishedAt,
    description: excerpt(decodeEntities(video.snippet.description || 'A recent AI video.')),
    image: video.snippet.thumbnails?.medium?.url || video.snippet.thumbnails?.default?.url,
  }));
}

type PipedVideo = {
  type?: 'stream' | string;
  title?: string;
  url?: string;
  uploaderName?: string;
  uploaded?: number;
  uploadedDate?: string;
  shortDescription?: string;
  views?: number;
  duration?: number;
  thumbnail?: string;
};

async function fetchCommunityVideos(signal?: AbortSignal): Promise<RawDiscovery[]> {
  // Piped exposes a keyless, CORS-enabled search over public YouTube videos.
  // Its public instance is a community dependency, so failure stays isolated.
  const params = new URLSearchParams({
    q: 'latest AI model LLM agent paper explained',
    filter: 'videos',
  });
  const now = Date.now();
  const collected = new Map<string, PipedVideo>();
  const visited = new Set<string>();
  let nextpage = '';
  // Search pages can contain only a few recent videos. Continue when needed.
  for (let page = 0; page < 5; page++) {
    if (nextpage) params.set('nextpage', nextpage);
    const endpoint = nextpage ? 'nextpage/search' : 'search';
    try {
      const body = await sourceJSON<{items?: PipedVideo[]; nextpage?: string | null}>(`https://api.piped.private.coffee/${endpoint}?${params}`, 'Community video index', signal);
      for (const video of body.items ?? []) {
        if (video.type === 'stream' && video.url?.includes('v=') && video.uploaded
          && video.uploaded >= now - VIDEO_DAYS * 24 * 60 * 60 * 1000 && video.uploaded <= now) {
          collected.set(video.url, video);
        }
      }
      nextpage = body.nextpage || '';
      if (collected.size >= ITEMS_PER_SOURCE || !nextpage || visited.has(nextpage)) break;
      visited.add(nextpage);
    } catch (error) {
      signal?.throwIfAborted();
      if (collected.size === 0) throw error;
      break;
    }
  }
  const videos = [...collected.values()].sort((a, b) => (b.uploaded ?? 0) - (a.uploaded ?? 0)).slice(0, CANDIDATE_LIMIT);
  if (videos.length === 0) throw new Error('The community index returned no videos.');

  return videos.map((video) => {
    const videoId = new URLSearchParams(video.url?.split('?')[1] || '').get('v') || video.url;
    return {
      id: `video:${videoId}`,
      category: 'videos' as const,
      title: normalise(video.title || 'Recent AI video'),
      url: `https://www.youtube.com/watch?v=${videoId}`,
      source: `${video.uploaderName || 'YouTube'} · community index`,
      publishedAt: video.uploaded && video.uploaded > 0 ? new Date(video.uploaded).toISOString() : '',
      description: excerpt(video.shortDescription || `An AI video from ${video.uploaderName || 'YouTube'}.`),
      image: video.thumbnail || `https://i.ytimg.com/vi/${videoId}/mqdefault.jpg`,
      metrics: [
        typeof video.views === 'number' ? `${video.views.toLocaleString()} views` : '',
        typeof video.duration === 'number' ? `${Math.max(1, Math.round(video.duration / 60))} min` : '',
      ].filter(Boolean),
      details: {
        views: video.views,
        durationSeconds: video.duration,
      },
    };
  });
}

async function fetchHuggingFacePapers(signal?: AbortSignal): Promise<RawDiscovery[]> {
  const records = await sourceJSON<Array<Record<string, any>>>(`https://huggingface.co/api/daily_papers?limit=${CANDIDATE_LIMIT}`, 'Hugging Face Papers', signal);
  return records.filter((record) => record.paper?.id && record.paper?.title).map(({paper}) => ({
    id: `paper:arxiv:${paper.id}`,
    category: 'papers',
    title: normalise(paper.title),
    url: `https://arxiv.org/abs/${paper.id}`,
    source: 'arXiv via Hugging Face Papers',
    publishedAt: paper.publishedAt || '',
    description: excerpt(paper.summary, 520) || 'Read the paper for its methods and findings.',
    authors: (paper.authors ?? []).slice(0, 4).map((author: {name: string}) => author.name).filter(Boolean),
    metrics: typeof paper.upvotes === 'number' ? [`${paper.upvotes} community votes`] : [],
  }));
}

async function fetchHackerNewsVideos(signal?: AbortSignal): Promise<RawDiscovery[]> {
  const videos = new Map<string, RawDiscovery>();
  const params = new URLSearchParams({
    query: 'youtube', restrictSearchableAttributes: 'url', tags: 'story', hitsPerPage: '200',
    numericFilters: `created_at_i>=${Math.floor(Date.now() / 1000) - VIDEO_DAYS * 86400}`,
  });
  const aiTopic = /\b(ai|llms?|gpt\w*|chatgpt|claude|gemini|openai|anthropic|deepseek|qwen|llama|transformers?|neural|diffusion|rag)\b|machine learning|deep learning|artificial intelligence|reinforcement learning/i;
  for (let page = 0; page < 5; page++) {
    params.set('page', String(page));
    const body = await sourceJSON<{hits?: Array<{url?: string; title?: string; created_at?: string; points?: number}>; nbPages?: number}>(`https://hn.algolia.com/api/v1/search_by_date?${params}`, 'Hacker News video index', signal);
    for (const hit of body.hits ?? []) {
      if (!hit.url || !hit.title || !aiTopic.test(hit.title)) continue;
      let videoId: string | null = null;
      try {
        const url = new URL(hit.url);
        if (['youtube.com', 'www.youtube.com', 'm.youtube.com'].includes(url.hostname)) {
          videoId = url.searchParams.get('v') || url.pathname.match(/^\/(?:shorts|embed)\/([^/]+)/)?.[1] || null;
        }
      } catch { continue; }
      if (!videoId || !/^[\w-]{11}$/.test(videoId)) continue;
      videos.set(videoId, {
        id: `video:${videoId}`, category: 'videos', title: normalise(hit.title),
        url: `https://www.youtube.com/watch?v=${videoId}`, source: 'YouTube via Hacker News',
        publishedAt: '', sharedAt: hit.created_at,
        description: 'An AI video shared by the Hacker News community. Open the original video for the full explanation.',
        image: `https://i.ytimg.com/vi/${videoId}/mqdefault.jpg`,
        metrics: typeof hit.points === 'number' ? [`${hit.points} Hacker News points`] : [],
      });
    }
    if (videos.size >= ITEMS_PER_SOURCE || !body.hits?.length || page + 1 >= (body.nbPages ?? 1)) break;
  }
  return [...videos.values()];
}

type SourceResult = {items: RawDiscovery[]; message?: string};

async function withFallbacks(providers: Array<{name: string; run: () => Promise<RawDiscovery[]>}>, signal?: AbortSignal): Promise<SourceResult> {
  const items: RawDiscovery[] = [];
  const keys = new Set<string>();
  const messages: string[] = [];
  for (const [index, provider] of providers.entries()) {
    signal?.throwIfAborted();
    try {
      const records = await provider.run();
      for (const item of records) {
        const identity = discoveryKeys(item);
        if (identity.some((key) => keys.has(key))) continue;
        identity.forEach((key) => keys.add(key));
        items.push(item);
      }
      if (index > 0 && records.length > 0) messages.push(`Using ${provider.name} to fill this category.`);
      if (items.length >= ITEMS_PER_SOURCE) break;
      messages.push(`${provider.name} did not provide ${ITEMS_PER_SOURCE} items.`);
    } catch (error) {
      signal?.throwIfAborted();
      messages.push((error as Error).message);
    }
  }
  if (items.length === 0) throw new Error(messages.join(' '));
  return {items, message: messages.join(' ') || undefined};
}

type FetchResult = {items: RawDiscovery[]; reports: SourceReport[]; newCount: number};

export async function fetchAllDiscoveries(settings: AISettings, signal?: AbortSignal, previouslySeen: readonly string[] = [], savedItems: readonly RawDiscovery[] = []): Promise<FetchResult> {
  const sources = [
    {name: 'Papers', category: 'papers', run: () => withFallbacks([
      {name: 'OpenAlex', run: () => fetchPapers(signal)},
      {name: 'Hugging Face Papers', run: () => fetchHuggingFacePapers(signal)},
    ], signal)},
    {name: 'Hugging Face', category: 'models', run: async (): Promise<SourceResult> => ({items: await fetchModels(signal)})},
    {name: 'GitHub', category: 'tools', run: async (): Promise<SourceResult> => ({items: await fetchTools(settings, signal)})},
    {
      name: 'Videos',
      category: 'videos',
      run: () => withFallbacks([
        ...(settings.youtubeApiKey ? [{name: 'YouTube', run: () => fetchVideos(settings, signal)}] : []),
        {name: 'Community video index', run: () => fetchCommunityVideos(signal)},
        {name: 'Hacker News video index', run: () => fetchHackerNewsVideos(signal)},
      ], signal),
    },
  ];
  const settled = await Promise.allSettled(sources.map((source) => source.run()));
  signal?.throwIfAborted();
  const seen = new Set(previouslySeen);
  const selectedKeys = new Set<string>();
  const items: RawDiscovery[] = [];
  let newCount = 0;
  const reports = settled.map((result, index): SourceReport => {
    const {name: source, category} = sources[index];
    if (result.status === 'rejected' && result.reason?.name === 'AbortError') throw result.reason;
    const candidates = result.status === 'fulfilled' ? result.value.items : [];
    const selected: RawDiscovery[] = [];
    const append = (records: readonly RawDiscovery[]) => {
      for (const item of records) {
        if (selected.length >= ITEMS_PER_SOURCE) break;
        const keys = discoveryKeys(item);
        if (keys.some((key) => selectedKeys.has(key))) continue;
        keys.forEach((key) => selectedKeys.add(key));
        selected.push(item);
      }
    };
    // Prefer unseen records, then repeat source results to fill the category.
    // Saved records can fill gaps if an API is unavailable or returns too few.
    append(candidates.filter((item) => !discoveryKeys(item).some((key) => seen.has(key))));
    newCount += selected.length;
    append(candidates);
    const liveCount = selected.length;
    append(savedItems.filter((item) => item.category === category && (category !== 'videos'
      || Date.parse(item.publishedAt || item.sharedAt || '') >= Date.now() - VIDEO_DAYS * 24 * 60 * 60 * 1000)));
    const cachedCount = selected.length - liveCount;
    items.push(...selected);
    const messages = [
      result.status === 'rejected' ? result.reason?.message || 'Source unavailable.' : result.value.message,
      cachedCount > 0 ? `${cachedCount} items restored from saved updates.` : '',
      category === 'videos' && selected.some((item) => !item.publishedAt)
        ? 'Some upload dates are unavailable. Those cards show when the video was shared on Hacker News within the last 90 days.' : '',
      selected.length < ITEMS_PER_SOURCE ? `Only ${selected.length} of ${ITEMS_PER_SOURCE} items are available from this source and your saved updates.` : '',
    ].filter(Boolean);
    return {source, status: result.status === 'fulfilled' ? selected.length >= ITEMS_PER_SOURCE ? 'ok' : 'partial' : 'error', count: selected.length, message: messages.join(' ') || undefined};
  });
  return {items, reports, newCount};
}
