# Handoff: Claude → Codex. Project infographics, interactive labs, AI Security chapters

## ⚑ CURRENT STATE: Claude has handed over (user: "You reached 90%, handover now")

Codex now owns everything below. Claude's three drawing sub-agents were **stopped**, so nobody else is
editing these files.

| Project | Boards drawn | Placed in doc | What's left |
| --- | --- | --- | --- |
| AI Security (4 modules) | 51 SVGs, all reviewed by eye | **Done**: M1 11, M2 9, M3 17, M4 14 `<Infographic>` | Build and verify only. Mermaid left on purpose: M1 3 (bodyguard, whole stack, trace), M2 3 (goldens, phase 1, app wiring), M4 1 (feedback flow). |
| Secure EHR | 14 SVGs in `static/img/secure-ehr/`, **not reviewed, not placed** | 0; the doc still has 17 Mermaid | Render and check the 14; draw any missing (compare with the 17 `*Redrawn from Monal's whiteboard*` captions and the 9 PDF pages); place them. |
| Enterprise RAG S1 | 7 SVGs (`s1-security`, `s1-production-goals`, `s1-full-architecture`, `s1-prototype-to-cloud`, `s1-gateway-sketch`, `s1-agentic-architecture`, `s1-request-path`), **not reviewed, not placed** | 0; the doc has 57 Mermaid | Render and check the 7; draw the rest of the 38 in `scratchpad/videos/s1_catalogue.md`; place them. |
| Enterprise RAG S2 | 13 SVGs (`s2-recap-*` ×3, `s2-llm-security`, `s2-colang-define`, `s2-gateway`, `s2-gateway-keys`, `s2-api-vs-llm-engineering`, `s2-knowledge-distillation`, `s2-guard-approaches`, `s2-langgraph-graph`, `s2-gateway-explorer-streaming`, `s2-why-evaluate`), **not reviewed, not placed** | 0; the doc has 44 Mermaid | Render and check the 13; draw the rest of the 31 in `scratchpad/videos/s2_catalogue.md` (evaluation posters, AWS board, parsing boards…); place them. |
| Interactive labs | Codex's batch | none | Spec in `.codex/claude-coordination.md`; placement listed below. |

**How to place images** (the same method Claude used for AI Security):
`scripts/infographics/place.py` → `Doc(path)`, then `.replace_mermaid(caption_anchor, tag)`,
`.replace_mermaid_containing(needle_inside_mermaid, tag)`, `.insert_after(anchor, tag)` and `.save()`.
Anchors match with flexible whitespace and must be unique, otherwise nothing is written. `save()` adds
the `Infographic` import. Worked examples are the scripts Claude ran:
`scratchpad/place_m3.py` and `scratchpad/place_m4.py`, which ran from the repo root. Module 1 and 2
calls follow the same pattern.

**Sub-agent definition files** in `scripts/infographics/`: `secure_ehr.py`, `enterprise_rag_s1.py` and
`enterprise_rag_s2.py`. They may be partial or mid-edit, so run them and fix any that fail.

**Nothing has been built since the images were placed.** Run `npm run typecheck` and `npm run build`
first. `npx tsc --noEmit` passed after the component changes.

Claude keeps this file current at every milestone. If Claude stops (session limit), Codex continues from
here. Last updated 2026-09-30, during the AI Security doc integration.

## What the user asked for

1. **Infographics**: visual board-style images for every infographic in the course videos, for ALL
   projects. That means images like the whiteboards and slides, **not Mermaid**. Their words:
   "Infographics means visual images like this not mermaid diagrams", "you missed all of them",
   "for all the projects".
2. **Interactive visualisations** in the project pages.
3. Earlier, done: AI Security chapters 1–4 written (`docs/projects/ai-security/0*.md`); Runnables code fix
   (`docs/genai/10-runnables-part-1.md`); Enterprise RAG and Secure EHR chapters.

Rule that still applies (`.claude/AGENTS.md` §6): never paste video frames. Redraw every board as an
original SVG with the same content and layout.

## The kit (done)

- `scripts/infographics/board.py` holds the SVG primitives: `Board`, `group`, `card`, `cylinder`, `diamond`,
  `pill`, `text`, `person`, `bar`, `table` and `arrow`. Boxes expose `.top() .bottom() .left() .right()`.
- `scripts/infographics/ai_security.py` defines 51 boards. Run it with
  `python3 scripts/infographics/ai_security.py [name ...]`; it writes `static/img/ai-security/<name>.svg`,
  with underscores turned into dashes.
- `src/components/Infographic/index.tsx` is the `<Infographic src alt caption />` figure. It has an
  Expand button that opens the Mermaid lightbox, the image can't be dragged, and focus moves into the
  dialog and back.
- `src/theme/Mermaid/index.tsx`: `Lightbox` is now a named export and has focus management.
- Renderer for checking by eye:
  `node /private/tmp/claude-502/-Users-sumanth-tp-Resources-ai-ml-learn-ai-ml/e767fe90-f3f3-410c-83b6-d1e9e20865a4/scratchpad/verify/render_svg.mjs <outdir> <svg...>`,
  which writes PNGs. It uses the cached chromium headless shell `chromium_headless_shell-1228`.

## Status by project

### AI Security (Claude)

- **Boards: 51 SVGs, generated and reviewed by eye.** They're in `static/img/ai-security/`.
  - m1: why-guardrails, where-guardrail-sits, gateways-vs-guardrails, frameworks, dialog-rails,
    pii-urgency, colang, intent-matching, observability, three-rails, keys.
  - m2: hiring, two-things, goldens-judge, faithfulness, answer-relevancy, rag-triad,
    context-precision, context-recall, answer-correctness.
  - m3: mosaic, lineage, survey, buffer, sliding-window, summary, summary-buffer, token-buffer,
    vector-store, entity, hot-background, episodic, semantic, procedural, self-reflection, routing,
    forgetting.
  - m4: prototype-production, six-pillars, system-overview, phase1-infra, langfuse-trace,
    bedrock-guardrails, dag, hybrid-search, rag-cache, langgraph, mcp, eks, cicd, load-test.
- **Doc integration: IN PROGRESS** in `docs/projects/ai-security/01..04-*.md`. Method:
  - Add `import Infographic from '@site/src/components/Infographic';` after the frontmatter.
  - Replace each Mermaid block that redraws a video board (preceded by an italic `*Redrawn from …*`
    line) with `<Infographic src="/img/ai-security/<file>.svg" alt="…" caption="Redrawn from … H:MM to H:MM." />`.
    Fold the italic line into `caption`.
  - Keep explanatory Mermaid that isn't a video board. Where a board has no Mermaid, insert the image at
    the matching point in the flow.
  - Placement map (board → chapter section):
    - Module 1:
      - why-guardrails → "Why LLM security is a topic of its own", replacing the 0:03–0:06 Mermaid.
      - where-guardrail-sits → "The two kinds of LLM application", replacing 0:06–0:07. Also drop the
        0:11–0:12 Mermaid, which duplicates it.
      - gateways-vs-guardrails → "Four properties…", replacing 0:13–0:16.
      - frameworks → "Four frameworks for guardrails", inserted after the table.
      - dialog-rails → "Experiment 5".
      - pii-urgency → "Experiment 6".
      - colang → "Colang: the language of rails", replacing both 0:38–0:41 and 0:40–0:44.
      - intent-matching → "How NeMo matches an intent", replacing 0:47–0:48.
      - three-rails → "Three kinds of rails", replacing 0:55–0:56.
      - observability → "The Pydantic ecosystem", replacing 0:51–0:54.
      - keys → "Set up your keys".
      - Keep the bodyguard-sketch Mermaid, "The whole stack" and "Trace, span, waterfall".
    - Module 2: each `m2-*` board replaces the Mermaid under the section of the same name. The mapping
      is 1:1 with the captions `*Redrawn from … (1:23…2:37)*`. Keep "How the app is wired", which
      comes from the repo.
    - Module 3: `m3-mosaic` replaces all three MOSAIC Mermaid blocks. `m3-lineage` goes after the
      13-technique table. `m3-survey` replaces the survey Mermaid. Each technique section `## N.` gets
      its board, replacing its video-board Mermaid. `m3-hot-background` goes under "Hot path and
      background".
    - Module 4:
      - prototype-production → "From MLOps to AgentOps".
      - system-overview → "The project".
      - phase1-infra → "Phase 1".
      - langfuse-trace → "Tracing with Langfuse". It replaces the trace Mermaid; keep the feedback
        Mermaid, or replace it too, since the board includes feedback.
      - bedrock-guardrails → "Bedrock Guardrails".
      - dag → "Airflow, OpenSearch and Neon on screen".
      - hybrid-search → "Keyword, dense and hybrid search". It replaces both the chunking and the
        hybrid Mermaid.
      - rag-cache → "RAG with a Redis cache".
      - langgraph → "Phase 7".
      - mcp → "An MCP server…".
      - eks → "Deploy to Amazon EKS".
      - cicd → "CI/CD".
      - six-pillars → "What is AgentOps?".
      - load-test → "Load testing with Locust". It replaces the HPA-loop Mermaid; the table stays.

### Secure EHR (Claude sub-agent, running)

- Files: `scripts/infographics/secure_ehr.py` → `static/img/secure-ehr/`, placed in
  `docs/projects/secure-ehr-insight/01-live-implementation.md`.
- Sources, in the scratchpad (`/private/tmp/claude-502/-Users-sumanth-tp-Resources-ai-ml-learn-ai-ml/e767fe90-f3f3-410c-83b6-d1e9e20865a4/scratchpad/`):
  - `ehr/instructor_notes/monal-handwritten-notes.pdf`: the whiteboard pages.
  - `videos/ehr_frames/m_NNNN.jpg` (one per minute), `videos/ehr_sheets/`, `videos/ehr_full/`.
- If the sub-agent didn't finish, check which `*Redrawn from Monal's whiteboard*` Mermaid blocks still
  exist in the doc and draw those.

### Enterprise RAG, Session 1 (Claude sub-agent, running)

- Files: `scripts/infographics/enterprise_rag_s1.py` → `static/img/enterprise-rag/s1-*.svg`, placed in
  `docs/projects/enterprise-rag/01-session-1.md`. Very large file with base64 lines: use `awk`, skip
  lines over 600 characters.
- Checklist: `scratchpad/videos/s1_catalogue.md`, 38 boards with transcriptions and timestamps.
  Frames: `scratchpad/videos/s1_frames`, `s1_sheets`, `s1_full`.

### Enterprise RAG, Session 2 (Claude sub-agent, running)

- Files: `scripts/infographics/enterprise_rag_s2.py` → `static/img/enterprise-rag/s2-*.svg`, placed in
  `docs/projects/enterprise-rag/02-session-2.md`.
- Checklist: `scratchpad/videos/s2_catalogue.md`, 31 boards. Frames: `scratchpad/videos/s2_frames`,
  `s2_sheets`, `s2_full`.

The shared brief the sub-agents follow is
`scratchpad/infographic_brief.md`, with the placement rules and the MDX rules.

### Interactive labs (allocated to Codex)

Seven labs; the full spec is in `.codex/claude-coordination.md` → "Allocation update":
FaithfulnessLab, ContextPrecisionLab, ContextRecallLab, AnswerCorrectnessLab, MemoryWindowLab,
ForgettingCurveLab and HPALab. They go in `src/components/viz/`, built on `VizPanel`.

Placement once they're built:
- The four metric labs go in `02-llm-evaluations.md`, under their metric sections, after the board.
- MemoryWindowLab goes in `03-agentic-memory.md`, after "Choosing a short-term memory".
- ForgettingCurveLab goes in `03-agentic-memory.md`, "13. Forgetting and decay", after the formula.
- HPALab goes in `04-agentops-and-production.md`, "Load testing with Locust", after the runs table.
- Each doc needs `import XLab from '@site/src/components/viz/XLab';` after the frontmatter.
- The existing `RetrievalLab` can also go in Module 4, "Keyword, dense and hybrid search".

## Finishing checklist

1. Every board image placed. `grep -c "<Infographic"` per doc, and no leftover `*Redrawn from*` caption
   line directly followed by a Mermaid block that the board replaced.
2. `npm run typecheck`, then `npm run build` (must pass with 0 warnings; `onBrokenLinks: throw`). The
   build uses `DOCUSAURUS_GENERATED_FILES_DIR_NAME=.docusaurus-build`. The user's dev server runs on port
   3000 (PID 806); don't kill it.
3. Serve the build with `npx docusaurus serve --port 3111 --no-open`, then check in a browser:
   - every `<Infographic>` img loads (`naturalWidth > 0`);
   - Expand opens the lightbox, zoom and pan work, and Esc closes it;
   - the remaining Mermaid renders. `scratchpad/verify/mermaid_check.mjs <url>` prints the counts.
4. Update memory: the user's memory directory is
   `/Users/sumanth.tp/.claude/projects/-Users-sumanth-tp-Resources-ai-ml-learn-ai-ml/memory/`.
   Add to `ai-security-course-import.md` and `infographics-means-images.md`.
5. Report to the user: counts per project, anything skipped and why. Don't commit unless asked.

## Known facts and corrections to keep

- AI Security Module 1, 0:12: the enterprise example is 50 GB of data with about **500 MB relevant**, not
  "500 ms". The board is already corrected.
- Module 4's presenter is Sourangshu Pal ("Paul"); the repo is `sourangshupal/Agentic-RAG-project`, branch
  `agentops`.
- The Module 3 notebooks were never published; the chapter's code was written for the notes and tested
  offline (`scratchpad/memcode/`).
