# Guides for each job

`.codex/AGENTS.md` is the rulebook: the standard every chapter meets. These guides give the method for each kind of job, step by step, with the tools that were tested on this machine. Where a guide and the rulebook disagree, the rulebook wins. Report the conflict in the progress file.

Both agents read the same files: `.claude/AGENTS.md` and `.claude/guides` are links to `.codex/AGENTS.md` and `.codex/guides`, and `AGENTS.md` at the repo root links to the rulebook so Codex loads it automatically.

## Which guide

| The job | Read, in order |
| --- | --- |
| A YouTube video or playlist into chapters | `youtube-source-pack.md`, `youtube-chapter.md`, `notes-voice.md`, `fidelity-review.md` |
| A live class or lecture recording | The same as a video, and `notes-voice.md` is critical: classes are full of audience questions and session talk |
| A website, docs section, paper or PDF into a chapter | `web-sources.md`, then `youtube-chapter.md` sections 3 to 10 if the page is the source to follow, or `authored-chapters.md` if it is a reference |
| A new chapter with no source | `authored-chapters.md` |
| Projects for a topic | `projects.md` |
| A board (infographic) | `boards.md` |
| A lab (interactive visualisation) | `labs.md` |
| Reviewing a chapter someone else wrote, or fixing one that narrates | `fidelity-review.md`, `notes-voice.md` |
| Prompts to hand Codex for any of these | `prompts.md` |

Templates: `templates/chapter.md`, `templates/project.md`, `templates/source-report.md`.

## The tools

| Tool | What it does | Run with |
| --- | --- | --- |
| `scripts/source-import/yt_pack.py` | Video metadata, description chapters and links, transcript with fallback, 40 s blocks, coverage ledger, 360p video, de-duplicated frames, contact sheets, exact-moment grabs, notebook dump | `/usr/bin/python3` |
| `scripts/source-import/web_extract.py` | Web page or PDF to a private Markdown file, with fetch date, licence text, images and links; respects `robots.txt` | `/usr/bin/python3` |
| `scripts/source-import/render_svg.mjs` | Renders a board to PNG and fails on text outside the viewBox | `node` |
| `scripts/source-import/project_inventory.py` | Topics, their projects, and chapters no project uses | `/usr/bin/python3` |
| `.lecture-import/track-c/coverage_check.py` | Transcript blocks the chapter skipped or moved | `python3` |
| `.lecture-import/track-c/quality_gate.py` | Shape and voice gate for chapters and projects | `python3` |
| `.lecture-import/track-c/link_check.py` | Internal links and static assets | `python3` |
| `.lecture-import/codetest/run_all.py` | Runs every block under the runnable headings | the chapter's venv |
| `scripts/infographics/board.py` | Board kit | `python3` |

`.lecture-import/` is git-ignored: the gate, coverage check and runners live only on this machine. The `scripts/source-import/` tools are tracked.

## The pipeline, whatever the source

1. **Collect** the source into a private pack (video, page, paper).
2. **Ledger**: number the source's blocks and list every claim in order.
3. **Run** the code and the experiment before writing about them.
4. **Write** in the chapter shape, block by block, as notes, not narration.
5. **Draw** every picture as a board. **Build** a lab for what moves.
6. **Connect** the chapter to its topic's project.
7. **Verify**: coverage, gate, code runner, links, `tsc`, boards seen, browser.
8. **Report** from the template, including what was not verified, and update `.codex/senior-ai-progress.md`.

## Stale briefs

Older briefs under `.lecture-import/` (`proj3/BRIEF.md`, `agentic/BRIEF.md`, `agentic/PROMPT-for-other-ai.md`) allowed timestamps in headings and captions such as "Redrawn from the instructor's slide at 1:54:00". They are marked superseded. Do not copy their conventions.
