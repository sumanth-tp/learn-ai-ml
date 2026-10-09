---
name: youtube-chapter
description: Turn a YouTube video, playlist or class recording into site chapters that teach everything the video teaches, in its order, as proper notes (not narration), with boards, a lab and runnable code. Use when the user gives a YouTube URL or playlist to import.
---

# YouTube video into chapters

The method lives in shared files that Codex also reads. Read them fully before starting, in this order:

1. `.codex/AGENTS.md` (the rulebook; sections 2, 3 and 5 matter most here)
2. `.codex/guides/youtube-source-pack.md`: build the pack with `/usr/bin/python3 scripts/source-import/yt_pack.py all <url> .lecture-import/<job>/<NN>-<id> --frames`
3. `.codex/guides/youtube-chapter.md`: ledger, spine, block-by-block writing, code, boards, additions
4. `.codex/guides/notes-voice.md`: no "he says", no "the class", no "this session", no pointing words
5. `.codex/guides/fidelity-review.md`: review your own chapter before reporting

For a playlist, do one video at a time, sequentially, and record progress after each one in `.codex/senior-ai-progress.md`. When delegating to sub-agents or Codex, use the prompts in `.codex/guides/prompts.md`.

Done means: coverage_check, quality_gate, run_all and link_check are clean; every board has been rendered and looked at; the report from `.codex/guides/templates/source-report.md` is written, including what was not verified.
