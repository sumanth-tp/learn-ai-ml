# Secure EHR infographic completion

Completed 2026-09-30. Owner: Codex Secure EHR agent.

## Result

- **14 original SVG boards**, regenerated with `python3 scripts/infographics/secure_ehr.py`.
- **14 `<Infographic>` placements** in `docs/projects/secure-ehr-insight/01-live-implementation.md`.
- **15 source-board Mermaid blocks replaced**. The request-flow and API blocks are combined into one board.
- **2 explanatory Mermaid diagrams retained**: the redaction/guardrail responsibilities in Phase 5, and the deployed system in Phase 8.
- All **9 instructor PDF pages** reviewed and reconciled. Bappy's deployment sketch checked against minute frames 224 and 225.
- No missing source-board topics identified. No source screenshots embedded. Existing prose, headings, examples, callouts and source order preserved. All **39 non-Mermaid code blocks** checked against HEAD and preserved byte for byte within their fences.

## Placement map

All paths below are under `static/img/secure-ehr/`.

| SVG | Source | Chapter heading |
| --- | --- | --- |
| `ehr-problem.svg` | Monal page 1, 0:13–0:25 | Problem statement |
| `ehr-ai-sdlc.svg` | Page 2, 0:29–0:37 | AI SDLC: how an FDE builds this, versus classic SDLC |
| `ehr-overview.svg` | Page 3, 0:39–0:59 | Solution architecture |
| `ehr-phase0.svg` | Page 4 top, 1:10–1:34 | Phase 0 — Give the client their data |
| `ehr-phase1.svg` | Page 4 bottom, 1:54–2:02 | Phase 1 — Turning Postgres into a vector store |
| `ehr-phase2.svg` | Page 5 top, 2:09–2:14 | Phase 2 — Domain-specific clinical embeddings |
| `ehr-recap.svg` | Page 5 bottom, 2:21–2:27 | Phase 2, after the prepared-instance explanation |
| `ehr-presidio.svg` | Page 6 top, 2:37–2:47 | Phase 4 — Zero-trust PII redaction |
| `ehr-scores.svg` | Page 6 bottom and page 7 cutoff, 2:46 | Phase 4, worked entity-score example |
| `ehr-guardrails.svg` | Page 7 top, about 2:50 | Getting and paying for the DeepSeek API key |
| `ehr-request-apis.svg` | Page 7 bottom, 3:05–3:08 | Phase 6 — The FastAPI backend |
| `ehr-ui-prompt.svg` | Page 8 top, 3:08–3:26 | Phase 7 — The Streamlit frontend |
| `ehr-memory.svg` | Page 8 bottom and page 9, 3:08–3:26 | Phase 7 — The Streamlit frontend |
| `ehr-docker-plan.svg` | Bappy Excalidraw, 3:41–3:45 | Phase 8 — Productionising with Docker on AWS EC2 |

## Verification

- Rendered and visually inspected all 14 boards with the shared Chromium renderer. Re-rendered and re-inspected the score board after increasing its height and separating the threshold label from the group heading.
- Browser SVG bounding-box audit: **440 text elements**, **zero canvas overflows**, **zero text/text overlaps**.
- Source PDF rendered locally with `pdftoppm` and each of its 9 pages visually inspected.
- Existing development server page at `http://localhost:3000/docs/projects/secure-ehr-insight/live-implementation`: **14/14 images loaded**, **2 Mermaid SVGs rendered**, **zero page errors**. Expand opened a dialog; keyboard zoom executed; Escape closed it.
- All image references resolve to local SVGs; one component import; no remaining source-board caption preceding a Mermaid block.
- Review artefacts and scripts: `/tmp/ehr-review/` (PNG renders, source page renders, `bounds.json`, `verify.mjs`, `page.mjs`, placement script).
- Full build and typecheck intentionally left to the root agent to avoid concurrent builds.

## Notes

The instructor's 0.45 redaction cutoff remains visible in the score board. The adjacent chapter explanation retains the distinction from the committed code's 0.4 cutoff. Existing compliance qualifications remain untouched.

`place.py` collapses repeated blank lines globally, including code blocks. This changed whitespace in one Python example during insertion; those original blank lines were restored before the preservation check.
