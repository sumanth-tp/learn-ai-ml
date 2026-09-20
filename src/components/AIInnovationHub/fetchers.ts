import type {AISettings} from '@site/src/lib/aiSettings';

import type {RawDiscovery, SourceReport} from './types';

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
    'per-page': '8',
  });
  const response = await fetch(`https://api.openalex.org/works?${params}`, {signal});
  if (!response.ok) throw new Error(`OpenAlex returned ${response.status}`);
  const body = await response.json() as {results?: Array<Record<string, any>>};
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
  const params = new URLSearchParams({filter: 'text-generation', sort: 'createdAt', direction: '-1', limit: '8'});
  const response = await fetch(`https://huggingface.co/api/models?${params}`, {signal});
  if (!response.ok) throw new Error(`Hugging Face returned ${response.status}`);
  const models = await response.json() as Array<Record<string, any>>;
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
  const params = new URLSearchParams({q: query, sort: 'stars', order: 'desc', per_page: '8'});
  const headers: HeadersInit = {
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
  };
  if (settings.githubToken) headers.Authorization = `Bearer ${settings.githubToken}`;
  const response = await fetch(`https://api.github.com/search/repositories?${params}`, {headers, signal});
  if (!response.ok) {
    if (response.status === 403) throw new Error('GitHub search limit reached. Add an optional token or try later.');
    throw new Error(`GitHub returned ${response.status}`);
  }
  const body = await response.json() as {items?: Array<Record<string, any>>};
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
    publishedAfter: `${cutoff(21)}T00:00:00Z`,
    maxResults: '8',
    relevanceLanguage: 'en',
    safeSearch: 'moderate',
    key: settings.youtubeApiKey,
  });
  const response = await fetch(`https://www.googleapis.com/youtube/v3/search?${params}`, {signal});
  if (!response.ok) throw new Error(response.status === 403 ? 'YouTube rejected the key or its quota is exhausted.' : `YouTube returned ${response.status}`);
  const body = await response.json() as {items?: Array<Record<string, any>>};
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
  const response = await fetch(`https://api.piped.private.coffee/search?${params}`, {signal});
  if (!response.ok) throw new Error(`The community video index returned ${response.status}. Add an optional YouTube key for the official API.`);
  const body = await response.json() as {items?: PipedVideo[]};
  const now = Date.now();
  const videos = (body.items ?? [])
    .filter((video) => video.type === 'stream' && video.url?.includes('v='))
    .filter((video) => !video.uploaded || video.uploaded <= now)
    .sort((a, b) => (b.uploaded ?? 0) - (a.uploaded ?? 0))
    .slice(0, 8);
  if (videos.length === 0) throw new Error('The community index returned no recent videos. Add an optional YouTube key for the official API.');

  return videos.map((video) => {
    const videoId = new URLSearchParams(video.url?.split('?')[1] || '').get('v') || video.url;
    return {
      id: `video:${videoId}`,
      category: 'videos' as const,
      title: normalise(video.title || 'Recent AI video'),
      url: `https://www.youtube.com/watch?v=${videoId}`,
      source: `${video.uploaderName || 'YouTube'} · community index`,
      publishedAt: video.uploaded ? new Date(video.uploaded).toISOString() : '',
      description: excerpt(video.shortDescription || `A recent AI video published ${video.uploadedDate || 'recently'}.`),
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

type FetchResult = {items: RawDiscovery[]; reports: SourceReport[]};

export async function fetchAllDiscoveries(settings: AISettings, signal?: AbortSignal): Promise<FetchResult> {
  const sources = [
    {name: 'Papers', run: () => fetchPapers(signal)},
    {name: 'Hugging Face', run: () => fetchModels(signal)},
    {name: 'GitHub', run: () => fetchTools(settings, signal)},
    {
      name: settings.youtubeApiKey ? 'YouTube' : 'Videos (community)',
      run: () => fetchVideos(settings, signal),
    },
  ];
  const settled = await Promise.allSettled(sources.map((source) => source.run()));
  const items: RawDiscovery[] = [];
  const reports = settled.map((result, index): SourceReport => {
    const source = sources[index].name;
    if (result.status === 'fulfilled') {
      items.push(...result.value);
      return {source, status: 'ok', count: result.value.length};
    }
    if (result.reason?.name === 'AbortError') throw result.reason;
    return {source, status: 'error', count: 0, message: result.reason?.message || 'Source unavailable'};
  });
  return {items, reports};
}
