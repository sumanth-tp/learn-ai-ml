export type DiscoveryCategory = 'papers' | 'models' | 'tools' | 'videos';

export type RawDiscovery = {
  id: string;
  category: DiscoveryCategory;
  title: string;
  url: string;
  source: string;
  publishedAt: string;
  description: string;
  authors?: string[];
  image?: string;
  metrics?: string[];
  details?: {
    pipeline?: string;
    baseModel?: string;
    licence?: string;
    library?: string;
    repoLanguage?: string;
    stars?: number;
    views?: number;
    durationSeconds?: number;
  };
};

export type Discovery = RawDiscovery & {
  simpleExplanation: string;
  whyImportant: string;
  useCases: string[];
  prerequisites: string[];
  projectIdeas: string[];
  aiEdited: boolean;
};

export type SourceReport = {
  source: string;
  status: 'ok' | 'skipped' | 'error';
  count: number;
  message?: string;
};
