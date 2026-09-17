import type * as Preset from "@docusaurus/preset-classic";
import type { Config } from "@docusaurus/types";
import { themes as prismThemes } from "prism-react-renderer";

const math = require("remark-math");
const katex = require("rehype-katex");

const config: Config = {
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },
  themes: ["@docusaurus/theme-mermaid"],
  title: "Learn AI, ML",
  tagline: "Learn AI & ML from zero to hero",
  favicon: "img/favicon.svg",

  url: "https://learn-ai-ml.site",
  baseUrl: "/",

  onBrokenLinks: "throw",
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          remarkPlugins: [math],
          rehypePlugins: [[katex, { output: "html" }]],
          showLastUpdateTime: true,
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],
  plugins: [
    require.resolve("./plugins/learn-index"),
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        hashed: true,
        indexDocs: true,
        indexBlog: true,
        indexPages: true,
        docsRouteBasePath: "/",
        searchResultContextMaxLength: 80,
      },
    ],
  ],

  headTags: [
    {
      tagName: "link",
      attributes: { rel: "preconnect", href: "https://fonts.googleapis.com" },
    },
    {
      tagName: "link",
      attributes: {
        rel: "preconnect",
        href: "https://fonts.gstatic.com",
        crossorigin: "anonymous",
      },
    },
  ],

  stylesheets: [
    {
      href: "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap",
      type: "text/css",
    },
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css",
      type: "text/css",
      integrity:
        "sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV",
      crossorigin: "anonymous",
    },
  ],

  themeConfig: {
    image: "img/docusaurus-social-card.jpg",

    announcementBar: {
      id: "explore-2026",
      content:
        '<strong>New:</strong> browse all 345 notes in one place — <a href="/explore">open Explore</a>, or press <kbd>?</kbd> for shortcuts.',
      isCloseable: true,
    },

    colorMode: {
      defaultMode: "light",
      respectPrefersColorScheme: true,
    },

    /** ✅ Correct Mermaid config (no themeVariables — use CSS override instead) **/
    mermaid: {
      theme: {
        light: "neutral",
        dark: "dark",
      },
    },

    /** A 345-doc sidebar needs to be collapsible and self-tidying. */
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },

    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },

    navbar: {
      title: "AI & ML",
      hideOnScroll: true,
      logo: {
        alt: "Learn AI & ML",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "All notes",
        },
        { to: "/docs/intro", label: "Roadmap", position: "left" },
        {
          label: "Theory",
          position: "left",
          items: [
            { to: "/docs/category/dnn", label: "Deep learning" },
            { to: "/docs/category/statistics", label: "Statistics" },
          ],
        },
        { to: "/docs/category/coding", label: "Code", position: "left" },
        { to: "/docs/category/cheetsheet", label: "Cheatsheets", position: "left" },
        { to: "/docs/category/interview", label: "Interviews", position: "left" },
        { to: "/explore", label: "Explore", position: "left" },
        { type: "search", position: "right" },
      ],
    },

    footer: {
      style: "dark",
      links: [
        {
          title: "Learn",
          items: [
            { label: "Roadmap", to: "/docs/intro" },
            { label: "Deep learning", to: "/docs/category/dnn" },
            { label: "Statistics", to: "/docs/category/statistics" },
            { label: "Code tracks", to: "/docs/category/coding" },
          ],
        },
        {
          title: "Practice",
          items: [
            { label: "Cheatsheets", to: "/docs/category/cheetsheet" },
            { label: "Interview questions", to: "/docs/category/interview" },
            { label: "Engineering & systems", to: "/docs/category/scaler" },
            { label: "Miscellaneous", to: "/docs/category/miscellaneous-collection" },
            { label: "Explore all notes", to: "/explore" },
          ],
        },
        {
          title: "More",
          items: [
            { label: "PyTorch docs", href: "https://pytorch.org/docs/stable/index.html" },
            { label: "scikit-learn", href: "https://scikit-learn.org/stable/" },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Learn AI, ML. Built by Sumanth.`,
    },

    prism: {
      theme: prismThemes.oneLight,
      darkTheme: prismThemes.oneDark,
      additionalLanguages: ["bash", "python", "json", "yaml", "sql", "docker"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
