# Enterprise RAG project progress

## User requirements

Two self-contained Markdown chapters under Projects, one per supplied YouTube session. Keep the transcript's teaching sequence and actual student/host doubts. Inline corrections as `NOT from session`; use full continuous code, links, setup/expected results, topic summaries and source-grounded infographics. All learner-facing code and diagrams stay in the two Markdown files. Do not claim cloud/model execution that did not occur.

## Deliverables now present

- `docs/projects/enterprise-rag/01-session-1.md`: problem and requirements; repo structure; security/agent architecture; heterogeneous loading; embedding, chunking, indexing, retrieval/reranking; LangGraph state and nodes; FastAPI; Logfire; Streamlit; guardrails and gateway demos; doubts and summaries. All 18 labelled stage-3 source files match commit `1fae886310a4e122f9349b3ee32bf1e90a089b03` byte-for-byte apart from trailing whitespace. 8 inline figures, including 6 source-whiteboard PNGs and 2 redrawn video boards.
- `docs/projects/enterprise-rag/02-session-2.md`: guardrails, Portkey, evaluation dataset/metrics/full application, deployment fork runtime files, Docker/Compose, AWS network/IAM/ECR/ECS/ALB/TLS/CI/CD instructions, multimodal retrieval, Nemotron/Mistral/UnlimitedOCR, PP-DocLayout/GLM-OCR. 59 labelled complete source files checked against pinned repos: 53 exact; 6 intentional, inline-labelled corrections (Jina embedding, Jina reranking, Compose authentication, two ECS task definitions and CD workflow). 9 inline figures, including 6 source-whiteboard PNGs and 3 redrawn video boards. The unshared ColQwen notebook is explicitly identified; a complete direct-engine reproduction is marked `NOT from session`.
- Projects categories and navbar in `docusaurus.config.ts`. Two Markdown learner deliverables only; source figures embedded as data URIs.

## Sources and limitations

Full YouTube transcripts were extracted through the browser: Session 1 3,688 timed segments, Session 2 4,032. Public comments, supplied Google Doc, tldraw, Notion commands and pinned teaching/deployment/OCR/multimodal repositories were reviewed. Videos were sampled in 10-second storyboard frames, not literally every individual video frame; static source posters and whiteboard images were inspected and used for inline figures. The five hand-redrawn boards preserve source structure and labels but are readable redraws, not pixel-identical frame crops.

Source commits: teaching `52b771cbdea2e2215c823cc1ae522183b77a85b7`, stage-3 `1fae886310a4e122f9349b3ee32bf1e90a089b03`, deployment `f97dc63318b11db3e2d806db4d6528fef7baebf8`, OCR `c8d91fe0f5d474aa331dcdcde03da930ce212a9b`, multimodal `2e004a1abdb60ff4b6b38ccf850ed032d7abfe8f`, ColPali engine `3a562fc0d78acec847f067c832ad875fcdf51d32`.

No paid provider, GPU inference or AWS deployment was run. The user's own credentials, GPU and AWS account are required for those stages.

## Validation completed and remaining

- All Python/JSON/TOML fences parse (39 S1, 114 S2 checked by `/tmp/audit-rag-fences.py`). All 7 S1 and 47 S2 Bash blocks pass `bash -n`; all 4 S2 YAML blocks parse via `js-yaml`.
- 17 inline image payloads decode; SVG XML parses; first complete Docusaurus build passed. Browser verified both pages, all 15 S2 numbered headings, zero page errors and all inline figures loaded. Five redrawn SVGs were visually inspected; security and AWS layout overlaps were repaired.
- Final `npm run build` passed after all edits. Browser verified both pages: 8/8 and 9/9 inline figures loaded, all numbered headings present, no page errors or Mermaid syntax errors; diagrams rendered with Expand controls. Corrected Jina files pass Ruff lint/format in the deployment configuration and temporary mocked smoke tests for outage refusal, vector count and reranker index mapping. The source AWS code was not live-tested; retain this limit in final response.

## Other worktree notes

The separate prior user request added a ten-row chain table to `docs/genai/09-chains.md`; that task was completed and verified earlier. Many unrelated changes exist under `static/examples/projects/` from other work in this shared workspace; do not touch, stage or revert them. No subagents were authorised.

## Follow-up: code explanations and right contents pane

The user asked for explanatory comments in project code and a collapsible right table of contents whose closure expands the article. Added 39 `# Reader note:` comments in Session 1 and 113 in Session 2, including notes inside all commentable labelled file blocks and several long standalone snippets. Three JSON files have adjacent field explanations because JSON comments would invalidate copyable AWS/evaluation files. A nearby `NOT from session` marker identifies the reader comments as editorial additions. The 18 Session 1 source files still match their pinned originals after removing only those comment lines; the 59 Session 2 files retain only the six documented functional differences.

Swizzled `src/theme/DocItem/Layout/` to add an accessible desktop contents toggle. The state persists in localStorage across docs; the mobile TOC remains unchanged. When collapsed, the right rail shrinks to 3.25rem and the article's reading measure expands to fill the released width. On a 1680px browser, the article content increased from 794px to 1276px. Browser checks passed for hide/show, persistence across both RAG chapters, mobile, and a non-RAG docs page. `npm run build` and `npm run typecheck` passed after the changes. Python/JSON/TOML and YAML code fences parse; all Bash fences pass `bash -n`.

The first toggle was hard to spot. It now has a visible divider handle that reveals the labelled “Hide contents” control when the right pane is hovered or the button receives keyboard focus. The collapsed rail always shows a vertical “Show contents” button. On desktop widths through 1600px, the toggle uses a reserved row at the top of the pane so it does not cover article text; on wider screens it sits in the gap beside the pane. Production build, typecheck, hover/show/hide, persisted state, mobile behavior and article expansion were rechecked in the browser.

Latest follow-up: the user's screenshot showed no control because the long-running Docusaurus dev server on port 3000 had not picked up the newly swizzled theme file (port 3000 had zero matching buttons, while the production preview on 3111 had one). Replaced the hover handle with an always-visible “On this page” header and “Collapse” button; the collapsed rail has a vertical “Show contents” button. Restarted the dev server on the same port and verified the actual dark-mode page at 1918px: one visible toggle, article width 794px open → 1316px closed → 794px reopened, no page errors. Production build and typecheck passed; retested port 3000 after the build.
