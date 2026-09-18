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
};

export type DocSection = {
  key: string;
  label: string;
  /** Matched against the doc's source directory, longest prefix wins. */
  match: string;
  tone: string;
};

/** Ordered longest-prefix-first so `theory/dnn` beats `theory`. */
export const SECTIONS: DocSection[] = [
  {key: 'dnn', label: 'Deep Learning', match: 'theory/dnn', tone: 'indigo'},
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
    SECTIONS.find((section) => section.match !== '' && doc.dir.startsWith(section.match)) ??
    SECTIONS[SECTIONS.length - 1]
  );
}

type PluginData = {docs?: IndexedDoc[]; count?: number};

export function useDocsIndex(): IndexedDoc[] {
  const data = usePluginData('learn-index') as PluginData | undefined;
  return data?.docs ?? [];
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
