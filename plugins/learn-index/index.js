/**
 * Builds a lightweight, client-consumable index of every doc on the site.
 *
 * It piggybacks on the docs plugin's own loaded content (rather than re-parsing
 * markdown), so titles, descriptions and permalinks always match what
 * Docusaurus actually routed — including front-matter `slug` overrides.
 */
function walkStages(versions, permalinkOf) {
  const placement = new Map();
  const stages = [];
  let order = 0;

  const collect = (item, into) => {
    if (item.type === "doc" || item.type === "ref") {
      into.push(item.id);
    } else if (item.type === "category") {
      if (item.link?.type === "doc") {
        into.push(item.link.id);
      }
      (item.items ?? []).forEach((child) => collect(child, into));
    }
  };

  const linkOf = (item) => {
    if (item.type === "doc" || item.type === "ref") {
      return permalinkOf.get(item.id) ?? null;
    }
    if (item.link?.type === "doc") {
      return permalinkOf.get(item.link.id) ?? null;
    }
    return item.link?.permalink ?? null;
  };

  for (const version of versions) {
    for (const sidebar of Object.values(version.sidebars ?? {})) {
      for (const stageItem of sidebar) {
        const stage = stageItem.customProps?.stage;
        if (stageItem.type !== "category" || !stage) {
          continue;
        }
        const units = [];
        for (const unitItem of stageItem.items ?? []) {
          const unit = unitItem.customProps?.unit;
          const ids = [];
          collect(unitItem, ids);
          ids.forEach((id) => {
            if (!placement.has(id)) {
              placement.set(id, { stage, unit: unit ?? null, order: order++ });
            }
          });
          if (unit) {
            units.push({
              key: unit,
              label: unitItem.label ?? null,
              permalink: linkOf(unitItem),
              docCount: ids.length,
            });
          }
        }
        stages.push({ id: stage, units });
      }
    }
  }

  return { placement, stages };
}

module.exports = function learnIndexPlugin() {
  return {
    name: "learn-index",

    async allContentLoaded({ allContent, actions }) {
      const docsPlugin = allContent["docusaurus-plugin-content-docs"] ?? {};
      const versions = Object.values(docsPlugin).flatMap(
        (content) => content?.loadedVersions ?? [],
      );

      const allDocs = versions.flatMap((version) => version.docs ?? []);
      const permalinkOf = new Map(allDocs.map((doc) => [doc.id, doc.permalink]));
      const { placement, stages } = walkStages(versions, permalinkOf);

      const docs = allDocs
        .map((doc) => ({
          id: doc.id,
          title: doc.title,
          description: (doc.description ?? "").slice(0, 180),
          permalink: doc.permalink,
          dir: doc.sourceDirName === "." ? "" : doc.sourceDirName,
          updatedAt: doc.lastUpdatedAt ?? null,
          tags: (doc.tags ?? [])
            .map((tag) => (typeof tag === "string" ? tag : tag.label))
            .slice(0, 6),
          stage: placement.get(doc.id)?.stage ?? null,
          unit: placement.get(doc.id)?.unit ?? null,
          order: placement.get(doc.id)?.order ?? null,
        }))
        .sort((a, b) => a.permalink.localeCompare(b.permalink));

      actions.setGlobalData({ docs, count: docs.length, stages });
    },

    /**
     * Applies saved reading preferences before first paint, so a reader who
     * chose a larger type size never sees the default size flash first.
     */
    injectHtmlTags() {
      return {
        preBodyTags: [
          {
            tagName: "script",
            innerHTML: `(function(){try{var p=JSON.parse(localStorage.getItem('learn-ai-ml:reading')||'{}');var r=document.documentElement;r.dataset.readingSize=p.size||'default';r.dataset.readingWidth=p.width||'default';}catch(e){}})();`,
          },
        ],
      };
    },
  };
};
