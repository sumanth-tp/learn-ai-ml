/**
 * Builds a lightweight, client-consumable index of every doc on the site.
 *
 * It piggybacks on the docs plugin's own loaded content (rather than re-parsing
 * markdown), so titles, descriptions and permalinks always match what
 * Docusaurus actually routed — including front-matter `slug` overrides.
 */
module.exports = function learnIndexPlugin() {
  return {
    name: "learn-index",

    async allContentLoaded({ allContent, actions }) {
      const docsPlugin = allContent["docusaurus-plugin-content-docs"] ?? {};
      const versions = Object.values(docsPlugin).flatMap(
        (content) => content?.loadedVersions ?? [],
      );

      const docs = versions
        .flatMap((version) => version.docs ?? [])
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
        }))
        .sort((a, b) => a.permalink.localeCompare(b.permalink));

      actions.setGlobalData({ docs, count: docs.length });
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
