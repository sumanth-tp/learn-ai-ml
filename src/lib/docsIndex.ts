import {usePluginData} from '@docusaurus/useGlobalData';
import {useMemo} from 'react';

export type IndexedDoc = {
  id: string;
  title: string;
  description: string;
  permalink: string;
  dir: string;
  tags: string[];
  updatedAt: number | null;
  stage: string | null;
  unit: string | null;
  order: number | null;
};

export type StageUnit = {
  key: string;
  label: string | null;
  permalink: string | null;
  docCount: number;
};

export type StageSummary = {id: string; units: StageUnit[]};

export type DocSection = {
  key: string;
  label: string;
  /** Matched against the doc's source directory, longest prefix wins. */
  match: string | string[];
  tone: string;
};

/** Ordered longest-prefix-first so `theory/dnn` beats `theory`. */
export const SECTIONS: DocSection[] = [
  {key: 'daily', label: 'Daily', match: 'daily', tone: 'teal'},
  {key: 'research-papers', label: 'Research Papers', match: 'research-papers', tone: 'indigo'},
  {key: 'genai', label: 'Generative AI', match: 'genai', tone: 'violet'},
  {key: 'agentic-ai', label: 'Agentic AI', match: 'agentic-ai', tone: 'rose'},
  {key: 'mcp', label: 'MCP', match: 'mcp', tone: 'sky'},
  {key: 'llm-evals', label: 'LLM Evaluation', match: 'llm-evals', tone: 'amber'},
  {key: 'projects', label: 'Projects', match: 'projects', tone: 'teal'},
  {key: 'ml', label: 'Machine Learning', match: 'theory/ml', tone: 'sky'},
  {key: 'dnn', label: 'Deep Learning', match: 'theory/dnn', tone: 'indigo'},
  {key: 'cv', label: 'Computer Vision', match: 'theory/cv', tone: 'indigo'},
  {key: 'ir', label: 'Information Retrieval', match: 'theory/ir', tone: 'sky'},
  {key: 'specialised', label: 'Specialised ML', match: ['theory/timeseries', 'theory/recsys', 'theory/causal', 'theory/gnn', 'theory/speech', 'theory/udl', 'theory/va'], tone: 'teal'},
  {key: 'mlops', label: 'MLOps and Data', match: 'mlops', tone: 'amber'},
  {key: 'llm-engineering', label: 'LLM Engineering', match: 'llm-engineering', tone: 'violet'},
  {key: 'agentic-frontier', label: 'Agent Frontier', match: 'agentic-frontier', tone: 'rose'},
  {key: 'governance', label: 'Governance', match: 'governance', tone: 'amber'},
  {key: 'senior', label: 'Senior Craft', match: 'senior', tone: 'teal'},
  {key: 'drl', label: 'Reinforcement Learning', match: 'theory/drl', tone: 'violet'},
  {key: 'nlp', label: 'NLP', match: 'theory/nlp', tone: 'sky'},
  {key: 'seml', label: 'ML Engineering', match: 'theory/seml', tone: 'amber'},
  {key: 'stats', label: 'Statistics', match: 'theory/statistics', tone: 'teal'},
  {key: 'code', label: 'Code', match: 'code', tone: 'violet'},
  {key: 'cheatsheets', label: 'Cheatsheets', match: 'cheetsheet', tone: 'amber'},
  {key: 'interviews', label: 'Interviews', match: 'interviews', tone: 'rose'},
  {key: 'scaler', label: 'Engineering', match: 'scaler', tone: 'sky'},
  {key: 'misc', label: 'Miscellaneous', match: 'document-collection', tone: 'slate'},
  {key: 'start', label: 'Start here', match: '', tone: 'indigo'},
];

export function sectionOf(doc: IndexedDoc): DocSection {
  return (
    SECTIONS.find((section) =>
      [section.match].flat().some((prefix) => prefix !== '' && doc.dir.startsWith(prefix)),
    ) ?? SECTIONS[SECTIONS.length - 1]
  );
}

type PluginData = {docs?: IndexedDoc[]; count?: number; stages?: StageSummary[]};

export function useDocsIndex(): IndexedDoc[] {
  const data = usePluginData('learn-index') as PluginData | undefined;
  return data?.docs ?? [];
}

export function useStageSummaries(): StageSummary[] {
  const data = usePluginData('learn-index') as PluginData | undefined;
  return data?.stages ?? [];
}

export function useDocsInOrder(): IndexedDoc[] {
  const docs = useDocsIndex();
  return useMemo(
    () =>
      docs
        .filter((doc) => doc.order !== null)
        .sort((a, b) => (a.order ?? 0) - (b.order ?? 0)),
    [docs],
  );
}

export function useSectionCounts(): Map<string, number> {
  const docs = useDocsIndex();
  return useMemo(() => {
    const counts = new Map<string, number>();
    docs.forEach((doc) => {
      const {key} = sectionOf(doc);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
    return counts;
  }, [docs]);
}
