import {useMemo} from 'react';

import {STAGES, UNIT_LABELS, type PathLink, type Stage} from '@site/src/data/learningPath';
import {useDocsIndex, useDocsInOrder, useStageSummaries, type IndexedDoc} from '@site/src/lib/docsIndex';
import {useReadDocs, type ReadMap} from '@site/src/lib/progress';

export type ResolvedLink = PathLink & {href: string | null; read: boolean};

export type UnitProgress = {
  key: string;
  label: string;
  href: string | null;
  total: number;
  read: number;
};

export type StageStatus = 'done' | 'active' | 'todo';

export type StageProgress = {
  stage: Stage;
  docs: IndexedDoc[];
  units: UnitProgress[];
  total: number;
  read: number;
  pct: number;
  status: StageStatus;
  nextUp: IndexedDoc | null;
  comingSoon: boolean;
  milestone: ResolvedLink | null;
  pairWith: ResolvedLink[];
};

export function pct(done: number, total: number) {
  return total ? Math.round((done / total) * 100) : 0;
}

function resolve(link: PathLink, byId: Map<string, IndexedDoc>, read: ReadMap): ResolvedLink {
  if (link.docId) {
    const doc = byId.get(link.docId);
    return {...link, href: doc?.permalink ?? null, read: Boolean(doc && read[doc.permalink])};
  }
  return {...link, href: link.to ?? null, read: false};
}

export function useStageProgress(): StageProgress[] {
  const allDocs = useDocsIndex();
  const ordered = useDocsInOrder();
  const summaries = useStageSummaries();
  const read = useReadDocs();

  return useMemo(() => {
    const byId = new Map(allDocs.map((doc) => [doc.id, doc]));

    return STAGES.map((stage) => {
      const docs = ordered.filter((doc) => doc.stage === stage.id);
      const readCount = docs.filter((doc) => read[doc.permalink]).length;
      const summary = summaries.find((item) => item.id === stage.id);

      const units: UnitProgress[] = (summary?.units ?? []).map((unit) => {
        const unitDocs = docs.filter((doc) => doc.unit === unit.key);
        return {
          key: unit.key,
          label: UNIT_LABELS[unit.key] ?? unit.label ?? unit.key,
          href: unit.permalink,
          total: unitDocs.length,
          read: unitDocs.filter((doc) => read[doc.permalink]).length,
        };
      });

      const status: StageStatus =
        docs.length > 0 && readCount === docs.length ? 'done' : readCount > 0 ? 'active' : 'todo';

      return {
        stage,
        docs,
        units,
        total: docs.length,
        read: readCount,
        pct: pct(readCount, docs.length),
        status,
        nextUp: docs.find((doc) => !read[doc.permalink]) ?? null,
        comingSoon: docs.length === 0,
        milestone: stage.milestone ? resolve(stage.milestone, byId, read) : null,
        pairWith: (stage.pairWith ?? []).map((link) => resolve(link, byId, read)),
      };
    });
  }, [allDocs, ordered, summaries, read]);
}

export function currentStageOf(progress: StageProgress[]): StageProgress | null {
  const main = progress.filter(
    (item) => !item.stage.optional && item.stage.number !== null && !item.comingSoon,
  );
  return (
    main.find((item) => item.status === 'active') ??
    main.find((item) => item.status !== 'done') ??
    null
  );
}

export function useResumeDoc(progress: StageProgress[]): IndexedDoc | null {
  const ordered = useDocsInOrder();
  const read = useReadDocs();

  return useMemo(() => {
    const main = ordered.filter((doc) => {
      const stage = STAGES.find((item) => item.id === doc.stage);
      return stage && !stage.optional;
    });
    let lastIndex = -1;
    let lastAt = 0;
    main.forEach((doc, index) => {
      const at = read[doc.permalink];
      if (at && at > lastAt) {
        lastAt = at;
        lastIndex = index;
      }
    });
    const after = main.slice(lastIndex + 1).find((doc) => !read[doc.permalink]);
    return after ?? currentStageOf(progress)?.nextUp ?? null;
  }, [ordered, read, progress]);
}
