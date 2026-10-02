# Brief for Codex: senior AI curriculum, your tracks

From Claude, 2026-10-01. Read `senior-ai-plan.md` first (5 minutes), then this file. Status goes in
`senior-ai-progress.md`. Rules for every chapter are in `.codex/AGENTS.md` (the same file as
`.claude/AGENTS.md`); the gates G1 to G10 in the plan section 6 are the definition of done.

## Standing instructions from the user (override anything older)

1. **Do not send readers to the bansal website.** No hyperlink to `learning.bansal-ai.in` anywhere. Bring the
   content across instead. Credit the lecture in plain text only:
   `Built from the course lecture "ir-s5-vsm" (Lecture Library series).` The generator already does this.
2. **Copy the content.** Keep the lecture text (the earlier imports kept about 84%; only interactive-panel
   chrome is dropped). Bring across the lecture's practice questions, and the subject's question banks and
   solved mid-semester paper as a final practice chapter (section "Practice chapters" below).
3. **Infographics and interactive visualisations for every topic.** See "Visuals" below. Infographics means
   board-style SVG images, not Mermaid. A lecture's own interactive widget becomes our own lab, so a reader
   never needs the original page.
4. No code comments, British spelling, no GitHub references, no em dashes as connectors.

## Your tracks, in order

| # | Track | Chapters | Folder | Source spine |
| --- | --- | ---: | --- | --- |
| 1 | B1 Information retrieval | 14 + 2 practice | `docs/theory/ir/` | bansal IR, 14 of 16 sessions (8 and 16 are reviews) |
| 2 | C1 Data management and data observability | 16 + 2 practice | `docs/mlops/data/` | bansal DM |
| 3 | H Computer vision (P1) | 15 + 2 practice | `docs/theory/cv/` | bansal CV |
| 4 | K1 Time series (P1) | 5 | `docs/theory/timeseries/` | open sources |
| 5 | K2 Recommender systems (P1) | 4 | `docs/theory/recsys/` | bansal IR session 14, then open sources |

**Scope confirmed: P0 first.** Do tracks 1 and 2 now. Tracks 3 to 5 start only after I write "P1 OPEN" in the
progress file. Within a track, work in order. After the first three chapters of a track, record results in the
progress file and carry on; fix whatever my cross-review raises afterwards.

## What "bansal spine" means

The source lectures are short: IR averages about 320 words, DM 340, DML 330, CV 415. They are outlines with
worked examples and practice questions, and no code. A finished chapter is 2,500 to 4,000 words (the SEML,
NLP and DRL notes are the yardstick). So you author most of the chapter. The agreed shape, in order:

1. `**In one line.**` the point of the chapter.
2. The idea in plain words.
3. Infographic board(s), then a Mermaid diagram only where the idea is a flow or state machine.
4. **How it works**: the converted lecture text, kept.
5. A real system that uses it, named, with what it does in production. Only name a system you can back with a
   ledger URL; otherwise describe the pattern without naming a company.
6. Runnable Python. Real imports, CPU-only, fast, deterministic. Reproduce the lecture's worked numbers in code
   and show that they agree.
7. Interactive lab where the idea moves (see Visuals).
8. Design guidance: what to choose and what goes wrong.
9. The 2026 industry view.
10. Practice questions (the lecture's, kept, with answers) plus any you add.
11. Go deeper: your ledger sources, as links. No link to the bansal site.
12. Closing checklist phrased "I can explain why...".

Anything from outside the lecture needs a ledger row (progress file). Where a whole section is an addition
beyond the lecture, open it with `:::note Beyond the lecture`.

Exemplars: `docs/theory/seml/01-systems-thinking/01-what-changes-with-ml.md` (frontmatter, shape, code style)
and its generator spec `.lecture-import/spec_seml_1.py`.

## The pipeline (already working)

- Python: `.lecture-import/venv/bin/python` (beautifulsoup4, lxml). For running chapter code use
  `.lecture-import/venv-ml/bin/python` (numpy, scipy, pandas, scikit-learn, matplotlib, shap). Need another
  library? `venv-ml/bin/pip install` it there, never system-wide, and note it in the progress file.
- Crawled pages: `.lecture-import/crawl/pages/<stem>_lecture.html`. IR, DM, DML, CV and ML are all there,
  including `*-question-bank_question-bank.html`, `*-comprehensive-question-bank_*` and `*-midsem-2026_*`.
- Convert: `cd .lecture-import && ./venv/bin/python convert.py <indir> <outdir>`. Use **your own dirs**: copy
  only your stems into `in-<track>/`, convert into `converted-<track>/`.
- Author: a `spec_<track>_N.py` per group with a `GROUPS` dict like `spec_seml_1.py`, then
  `generate.build(entry, subject_key, pathlib.Path('out-<track>')/group)`. See the loop in `regen.sh`.
  Entry fields: `src, file, pos, id, title, label, slug, desc, tags, oneline, plain, mermaid, example, code,
  design, industry, reading, viz, viz_import`.
- Clean maths for Mermaid: `./venv/bin/python demath.py out-<track>`.
- **Run the code with `.lecture-import/codetest/run_ml.py <root> <venv-ml python>`**, not `run_runnable.py`.
  The old runner only syntax-checks anything importing scikit-learn. Read the printed output; a zero exit code
  is not enough, because a block that shells out can swallow a failure.
- Copy `out-<track>/...` into `docs/`.

**Do not run `regen.sh`.** It does `rm -rf converted out` and would wipe other work. Do not touch
`spec_drl_*`, `spec_nlp_*`, `spec_seml_*`.

Hard-won rules:

- Never run a whitespace-collapsing regex over a whole markdown file; it destroys Python indentation.
- Authored one-off chapters go straight into `docs/`, never into the regenerable `out-*`.
- Strip decorative pictographs; keep typographic arrows.
- Mermaid cannot typeset maths: keep formulas in KaTeX in the body.
- Currency is `\$`. Internal links start `/docs/`. No bare `{...}` in prose.
- Never paste video frames or screenshots. Redraw as original boards.

## Visuals (required, gate G9 and G10)

**Infographics.** At least one board per chapter, two where the chapter has two distinct ideas. Original
board-style SVGs in the whiteboard style of the existing ones.

- Kit: `scripts/infographics/board.py` (`Board`, `group`, `card`, `cylinder`, `diamond`, `pill`, `text`,
  `person`, `bar`, `table`, `arrow`). Read its docstring, and `scripts/infographics/ai_security.py` for a full
  example of a definition file.
- One definition script per track: `scripts/infographics/<track>.py`, output to `static/img/<track>/<name>.svg`
  (underscores become dashes).
- Place with `import Infographic from '@site/src/components/Infographic';` and
  `<Infographic src="/img/<track>/<name>.svg" alt="one sentence saying what it shows" caption="..." />`.
  `scripts/infographics/place.py` can place into an existing doc, but for new chapters just write the tag.
- Check by eye: render each SVG with headless chromium (cached at
  `~/Library/Caches/ms-playwright/chromium_headless_shell-*/`, driven with `playwright-core` as in AGENTS
  section 10) and look at the PNG. No overlapping or clipped text; no text outside the viewBox.
- Board facts must match the chapter text and its code output exactly.

**Interactive labs.** One lab for every concept that moves: a parameter you can change and see the effect of.
Rule of thumb: about one lab per two chapters, and every widget the original lecture had.

- Files: new `src/components/viz/<Name>Lab.tsx`, built on `VizPanel` with `useDarkViz()`, colours from
  `palette.ts`, a `table` prop for the data view, deterministic seeds, no external dependencies.
  References: `RetrievalLab.tsx` and `PPOClipLab.tsx`. Keyboard operable; sensible `aria-label`s.
- You may add new files in `src/components/viz/`. Do not edit `VizPanel*`, `palette.ts`, `CourseLab*` or
  anyone else's lab.
- Embed in the chapter with `import XLab from '@site/src/components/viz/XLab';` then `<XLab />`.
- Each lab's defaults must reproduce a number the chapter's code prints (for example the lecture's worked
  example). State that number in the chapter next to the lab.
- Typecheck with `npm run typecheck`.
- Write each lab as a short spec first in `.codex/visuals/<track>.md`: name, controls and ranges, defaults,
  what is drawn, the expected numbers. Claude placed the spec for the earlier seven labs this way and it worked.

## Practice chapters

Each bansal subject ends with a `99-practice` group holding:

| Chapter | Sources |
| --- | --- |
| `01-question-bank` | `<subject>-question-bank_question-bank` and `<subject>-comprehensive-question-bank_question-bank`, merged and de-duplicated, grouped by lecture, answers in `<details>` |
| `02-midsem-solved` | `<subject>-midsem-2026_midsem-2026-solved` (IR, DM, DML, CV only; ML has none) |

Keep the questions and the worked answers. Each chapter still gets a board or lab where it helps (for
example a marks-allocation board or a worked-calculation lab); do not pad.

## Track layouts

Chapter titles are `<Subject> · <Session N> — <title>` trimmed to a topic name, as the SEML notes do;
`sidebar_label` is short; `slug` is `/theory/<subject>/<topic>`. Category labels must be unique site-wide
(Docusaurus builds a `/docs/category/<label>` page from each), so prefix generic ones ("IR foundations", not
"Foundations").

### 1. IR: `docs/theory/ir/`

| Group folder | Sources |
| --- | --- |
| `01-foundations` | ir-s1-intro, ir-s2-boolean, ir-s3-dictionary-tolerant, ir-s4-index-compression |
| `02-ranking` | ir-s5-vsm, ir-s6-classification-clustering, ir-s7-evaluation |
| `03-the-web` | ir-s9-web-search, ir-s10-web-crawling, ir-s11-link-analysis, ir-s12-cross-language |
| `04-modern-retrieval` | ir-s13-multimodal-clip, ir-s14-recommender, ir-s15-neural-ir |
| `99-practice` | question banks and mid-semester paper |

Sessions 8 and 16 are reviews: carry their content into the relevant chapters' takeaways and say so in the
folder's `_category_.json` description. Tie chapters to what the site teaches: BM25 and TF-IDF to
`genai/16-rag`, neural IR to `projects/enterprise-rag`. Sessions 5, 7 and 15 get a runnable BM25 vs dense vs
hybrid vs reranked comparison on a tiny corpus. Lab ideas: tf-idf weighting explorer, inverted-index builder,
edit-distance matrix, precision/recall/nDCG explorer with draggable ranking, PageRank iteration.

### 2. DM: `docs/mlops/data/`

| Group folder | Sources |
| --- | --- |
| `01-data-foundations` | dm-s1-intro, dm-s2-principles, dm-s3-architectures |
| `02-pipelines-and-infrastructure` | dm-s4-pipelines, dm-s5-infra-dataops, dm-s6-ml-lifecycle |
| `03-getting-data-ready` | dm-s7-collection-ingestion, dm-s8-profiling-validation, dm-l9-analytics-engineering, dm-l11-feature-preparation |
| `04-data-in-production` | dm-l10-orchestration, dm-l12-experimentation-metadata, dm-l13-distributed-processing |
| `05-modern-concerns` | dm-l14-llm-pipelines, dm-l15-privacy-governance, dm-l16-observability |
| `99-practice` | question banks and mid-semester paper |

Cover feature stores in dm-l11, drift in dm-l16 (link `llm-evals/22`), experiment tracking in dm-l12 (link
the MLflow cheatsheet). Code runs locally: pandas, DuckDB or SQLite, validation checks written by hand where
installing the tool is heavy. Lab ideas: DAG scheduler with failures and retries, drift detector (PSI and KS)
with a shifting distribution, data-quality rule explorer, train/serve skew demo.

### 3. CV (P1): `docs/theory/cv/`

| Group folder | Sources |
| --- | --- |
| `01-image-fundamentals` | cv-s1-intro, cv-s2-image-fundamentals, cv-s5-color |
| `02-features-and-geometry` | cv-s3-edges, cv-s4-canny-hough, cv-s6-harris-hog, cv-s7-sift, cv-s8-ransac |
| `03-recognition` | cv-s9-10-image-classification, cv-s15-visual-bag-of-words |
| `04-segmentation-detection-tracking` | cv-s11-segmentation, cv-s12-semantic-metrics, cv-s13-object-detection, cv-s14-object-tracking |
| `05-deployment` | cv-s16-edge-devices |
| `99-practice` | question banks and mid-semester paper |

Deep learning for vision already lives in `theory/dnn`: link, do not repeat. OpenCV and NumPy for classical
chapters; show IoU, mAP and Dice on synthetic boxes and masks. Verify current detection and segmentation
families live and record them in the ledger. Lab ideas: convolution kernel playground, Canny threshold
explorer, Hough accumulator, IoU and NMS explorer, RANSAC line fit.

### 4. K1 time series (P1): `docs/theory/timeseries/`, 5 chapters, open sources

1. What makes time series different: trend, seasonality, stationarity, leakage-safe splits.
2. Classical forecasting: naive baselines, exponential smoothing, ARIMA and SARIMAX.
3. Machine learning for forecasting: lag and window features, tree models, global models.
4. Deep and pretrained forecasting models (verify which are current before writing).
5. Evaluation and production: backtesting, MASE and others, prediction intervals, anomaly detection.

First source to check: Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.). Synthetic
series so every block runs offline.

### 5. K2 recommender systems (P1): `docs/theory/recsys/`, 4 chapters

1. Framing: explicit vs implicit feedback, cold start, what you are optimising.
2. Collaborative filtering and matrix factorisation (build on `ir-s14-recommender`).
3. Two-tower retrieval plus ranking: candidate generation, ANN search, re-ranking.
4. Evaluating and operating recommenders: offline metrics, A/B tests, feedback loops, diversity.

## Per-track set-up checklist

1. Add the folder's `_category_.json` files (`label`, `position`, generated-index `description`).
2. Ledger rows for every outside source before using it.
3. Write the first three chapters complete with boards and labs, run G1 to G10, record results.
4. Carry on. Update the progress file at least once per group.

## Build and report

- Build in isolation: `DOCUSAURUS_GENERATED_FILES_DIR_NAME=.docusaurus-codex npx docusaurus build --out-dir .lecture-import/codex-build`.
  It must pass with no warnings. Run it at the end of every group.
- Do not edit `src/**` except new lab files in `src/components/viz/`. Do not edit `sidebars.ts`,
  `docusaurus.config.ts`, `package.json`, `docs/interviews/**`, or anyone else's folders. A new stage entry or
  label: ask in the progress file and I will wire it.
- Do not commit or push. Leave the working tree for me to review.
- Report honestly: what is covered, what is not, and what you changed from this brief.

## When your tracks are done

Take the next unowned track in the plan (K3 first), claim it in the progress file, and carry on. Then do the
independent verification pass: build, route diff, every new page in a real browser at light, dark and 390 px,
code-block re-run, every lab operated, and a read-through of Claude's first three chapters per track.
