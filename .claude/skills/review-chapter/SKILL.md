---
name: review-chapter
description: Review a chapter (often Codex output) against its source video, transcript or page for missing or reordered content, lost frames and code, unreproduced numbers and narrated voice ("he said", "the class", "this session"); or rewrite a narrating chapter into proper notes. Use when the user asks to check, review, accept or fix a chapter.
---

# Review a chapter against its source

Read `.codex/guides/fidelity-review.md` and `.codex/guides/notes-voice.md` fully.

For a review: do not edit the chapter. Run the seven checks and the five-block spot check, then write the report in the guide's format with a verdict (accept, accept after fixes, redo) and numbered findings, most important first.

For a voice rewrite: follow "The rewrite pass" in `notes-voice.md`. Keep every claim in place and in order; run `coverage_check.py` before and after, and the counts must not get worse.

The gate (`python3 .lecture-import/track-c/quality_gate.py <chapter>`) fails narration of the speaker, the room and the session, and warns on pointing words. It checks shape, not truth: the manual walk is still required.
