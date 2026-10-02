import {LEFTOVER_STAGE_ID, PATH_ROUTE, STAGES} from '../data/learningPath';

type Item = {
  type: string;
  id?: string;
  items?: Item[];
  link?: {type: string; id?: string};
  customProps?: Record<string, unknown>;
  [key: string]: unknown;
};

const SPLIT_FOLDERS = new Set(['theory', 'mlops']);

function firstDocId(item: Item): string | undefined {
  if (item.type === 'doc' || item.type === 'ref') {
    return item.id;
  }
  if (item.type === 'category') {
    if (item.link?.type === 'doc' && item.link.id) {
      return item.link.id;
    }
    for (const child of item.items ?? []) {
      const id = firstDocId(child);
      if (id) {
        return id;
      }
    }
  }
  return undefined;
}

function unitKeyOf(item: Item, depth: number): string {
  const id = firstDocId(item);
  return id ? id.split('/').slice(0, depth).join('/') : '';
}

function tagUnit(item: Item, unit: string): Item {
  return {...item, customProps: {...item.customProps, unit}};
}

function stageCategory(
  id: string,
  label: string,
  number: number | null,
  optional: boolean,
  items: Item[],
): Item {
  return {
    type: 'category',
    label,
    collapsible: true,
    collapsed: true,
    className: ['sidebar-stage', optional && 'sidebar-stage--optional'].filter(Boolean).join(' '),
    customProps: {stage: id, number},
    items,
  };
}

export function groupIntoStages(items: Item[]): Item[] {
  const units = new Map<string, Item[]>();
  const seen: string[] = [];

  const add = (unit: string, item: Item) => {
    if (!units.has(unit)) {
      units.set(unit, []);
      seen.push(unit);
    }
    units.get(unit)!.push(tagUnit(item, unit));
  };

  for (const item of items) {
    const top = unitKeyOf(item, 1);
    if (item.type === 'category' && SPLIT_FOLDERS.has(top)) {
      (item.items ?? []).forEach((child) => add(unitKeyOf(child, 2), child));
    } else {
      add(top, item);
    }
  }

  const claimed = new Set<string>();
  const grouped: Item[] = [];

  for (const stage of STAGES) {
    const children = stage.units.flatMap((unit) => {
      claimed.add(unit);
      return units.get(unit) ?? [];
    });
    if (stage.id === 'start') {
      children.push({type: 'link', label: 'Interactive learning path', href: PATH_ROUTE});
    }
    if (children.length === 0) {
      continue;
    }
    const label = stage.number === null ? stage.sidebarLabel : `${stage.number} · ${stage.sidebarLabel}`;
    grouped.push(stageCategory(stage.id, label, stage.number, Boolean(stage.optional), children));
  }

  const leftovers = seen.filter((unit) => !claimed.has(unit)).flatMap((unit) => units.get(unit)!);
  if (leftovers.length > 0) {
    grouped.push(stageCategory(LEFTOVER_STAGE_ID, 'More', null, true, leftovers));
  }

  return grouped;
}
