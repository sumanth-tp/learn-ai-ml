# How to write chapters for this site

This is the single rulebook for every agent that writes or edits chapters under `docs/`, Codex and Claude alike. It replaces the older briefs, prompts and playbooks. The rules exist because each one was learned by getting it wrong first.

The gate is `python3 .lecture-import/track-c/quality_gate.py <folder or file>`. Passing it is the minimum. It checks shape, not truth and not quality. The standard is in sections 1 to 4.

---

## 1. Think before you write

The difference between a chapter that teaches and a summary of a lecture is decided before the first sentence. Do these in order.

1. **Name the reader and the outcome.** A reader who finished the previous stage of the learning path. After the chapter they can do one concrete thing they could not do before. Write that sentence first. Everything in the chapter serves it.
2. **Find the hardest idea.** Every chapter has one idea that beginners get wrong. Build the chapter around making that idea easy: a person in a situation, a small example with numbers, then the rule.
3. **Run something real before you write about it.** Write the experiment (50 to 150 lines, the real library, seeded, deterministic, under a minute) and run it. Read the output carefully. What it prints decides what the prose says, including when it contradicts what you expected or what the source claims. Never write a number you did not print or read from a source you opened that day.
4. **Look for the surprise.** The most valuable sentence in a chapter is usually "this did not work, and here is why". A method that loses, a metric that disagrees, a setting that does not matter. Keep it and explain it. Say what you did not test.
5. **Plan the spine, then write.** For each concept: situation, small worked example by hand, the general statement with "In words: ...", the code that reproduces the hand numbers exactly, the printed result, how it breaks.
6. **Review as three readers before you report.** As a beginner: is every term defined before use, can I follow the example with a pencil? As a sceptic: does every number come from a run, does every claim have a source, where would this fail? As an editor: delete every sentence that does not teach.
7. **When something cannot be done, say so.** Not obtainable, not runnable, not verified: a `:::warning` on the page and a line in the report. Never quietly drop it, never invent a substitute and present it as real.

If you are blocked or unsure, write `BLOCKER:` and the question in the progress file and continue with what you can do. Do not guess facts, versions, prices or benchmarks.

---

## 2. The voice of the notes

Notes teach the **subject**. They are not a record of a video, a review of a speaker or a log of what was on screen. A reader should finish knowing the topic, with no idea who presented it or what the screen looked like.

**Never write:**

- Video timestamps: not in headings ("## 04:57 · ..."), not in sentences ("At 06:23 the instructor ..."), not in link text, not in `&t=` links, not in a timestamp table.
- Narration of the video or speaker: "the instructor says", "in the video", "the lecture shows", "he opens", "he types", "on the screen", "in the frame", "the source's final answer", "the notebook's output", "the recorded version".
- Commentary on what the speaker got wrong. State the correct idea and explain it. If the source's code or claim is wrong, ship the correct version and put the correction in a `:::note` without saying who said what.
- Claims about how you checked the source ("checked against frames", "captions were translated"). That is your process. Put it in the progress file.

Keep exactly one pointer to the source: the source line at the top of the chapter, one line, no timestamps.

| Narrates the source (reject) | Teaches the idea (accept) |
| --- | --- |
| "At 15:17, Ajay explains that Ollama is like WhatsApp for models." | "Think of Ollama as WhatsApp for open models. You send and receive; it handles storage, loading and the interface behind the scenes." |
| "He types `ollama rm llama3.2:1b` and lists the models again." | "`ollama rm` deletes a model. List the models afterwards to confirm it is gone." |
| "The video passes explicit schemas, but its claim that functions cannot be passed directly is too broad." | "Pass an explicit JSON schema, as below. The Python SDK can also derive a schema from a typed function." |
| "The notebook's moon answer confuses moonlight with eclipses." | "A one-billion-parameter model can get this wrong: it may confuse moonlight with eclipses. Treat its answer as a demonstration of the call, not as astronomy." |

Clock times inside worked examples ("the job runs at 09:00") are fine.

---

## 3. The chapter shape

Every chapter, in this order. Existing chapters keep all existing text and are enriched around it.

1. `**In one line.**` The chapter's point in plain words.
2. `:::tip Before you start`: what you should already know (2 to 4 bullets, each linking to the chapter that teaches it), reading time, and "after this chapter you can ..." (2 to 3 outcomes).
3. `## In 30 seconds`: 3 to 5 sentences, no jargon, one everyday example and one analogy.
4. `## Words you will meet`: a table (Term, Plain meaning, Tiny example), 5 to 10 terms. Every other new term is glossed in brackets at first use.
5. A **big-picture board** near the top.
6. `## The idea in plain words`, then `## Worked example, step by step`: numbered steps with the arithmetic shown, small numbers, before any code. A step-by-step board goes here. The code later reproduces these exact numbers.
7. The source's content in its own order (section 5), then `## Code you can run`: blocks of about 40 lines (self-contained, because the runner executes each block separately; a longer self-contained experiment is acceptable, with a one-sentence introduction). Before each block, one sentence: what we are about to do and why. After each block: **Reading the output** (what each printed number means, what a wrong value would look like) and **Line by line** (bullets for the non-obvious lines only). Code comments are forbidden, so all explanation lives in the prose around the code.
8. Each lab is embedded with **What each control does** (one line per control) and **Try it yourself**: 3 numbered guided experiments, each "set X to Y, watch Z change, here is why", with results you checked. The lab's defaults reproduce a number the chapter prints.
9. `## Common mistakes`: 3 to 5 items, each "the mistake, why it feels right, what to do instead".
10. `## Practice questions`: block-form `<details>` labelled Easy, Medium or Stretch; answers explain the reasoning.
11. `## Go deeper` (sources, each opened and dated), `## Check yourself` ("I can ..." items, capabilities not topics), `## Where to go next` (the next chapter and one related chapter).

Measured targets (the gate warns): average sentence 22 words or fewer, paragraph about 90 words or fewer, Flesch 50 or higher, at least 80 lines of Python with a real third-party library, at least 2 boards and 1 lab.

Practice-question format, always block form:

```markdown
<details>
<summary><strong>Q1 (Easy).</strong> The question.</summary>

The answer, with the reasoning.

</details>
```

A one-line `<details>` breaks hydration.

---

## 4. The craft

### Style, with before and after

| Thin | Taught |
| --- | --- |
| "ARIMA models autocorrelation with an integrated moving-average structure." | "Imagine forecasting tomorrow's sales. Today's sales tell you a lot, and last week's tell you a bit. ARIMA writes that as a weighted sum of the last p values and the last q forecast errors. With p = 1 and a weight of 0.8, tomorrow = 0.8 x today. In words: each value leans on the one before it." |
| "Use cross-validation to evaluate." | "Random splits leak the future into the past on a time series. We split by time instead, and on this data the random split scored 0.97 R squared against 0.71 for the honest split. Same model, same data." |
| "Quantisation reduces model size." | "Weights stored as 32-bit floats take 4 bytes each. As 8-bit integers they take 1. A 7B model goes from 28 GB to 7 GB. We measured speed too: on this CPU int8 was 1.3x slower, because the matrix routine was not optimised for it." |
| Seven fragment bullets. | One idea per paragraph, at most four sentences. Bullets for steps and parameters only. |

### Exemplars to copy

**Open with a person, not a definition.** "Imagine a new colleague who has read your company handbook. You ask a question. A good colleague finds the right page, tells you the answer and shows you the page. A bad colleague guesses and sounds sure."

**Do the arithmetic by hand before the code.** "Chunk A is 1st in the meaning list and 3rd in the word list; B is 2nd and 1st. With weights 0.6 and 0.4 and k = 60, A scores 0.6/61 + 0.4/63 = 0.016185 and B scores 0.6/62 + 0.4/61 = 0.016235. B wins by a hair. In words: each list gives a chunk a small score that shrinks with rank, and the scores add up."

**Read the output, including a disappointing result.** "Hybrid retrieval did not beat keyword search here (recall 0.88 against 0.92). On 17 chunks the top 5 is almost a third of everything, so every method finds the right file. This is a useful result, not a failed one. Measure it on your documents before you adopt a technique."

**Name the failure with its cause.** "The system refused question 21 although the right file was retrieved. That is a generation failure, not a retrieval failure: the report says 'down' and the question says 'unavailable'. Good recall with a bad answer points at the prompt or the model, not the retriever."

**Line by line, only for non-obvious lines.** "`ids=[chunk.metadata["citation"] ...]` is the key detail. Giving Chroma an id means adding the same chunk twice replaces it. Without ids, every ingestion duplicates the library."

**A lab with exact experiments.** "Set the chunk size to 200 and the overlap to 0. You now get 10 chunks of about 136 characters. Small chunks are precise, but one this small often cuts an answer in half."

**A warning that says what was not verified.** "All 75 tests were run. The real-model path was written against the documentation but not run against a live provider. Run it with a real key before you trust it."

### Rules that follow

1. Every number is printed or computed, or read from a source you opened that day. Record library versions and dates.
2. The chapter's code is the tested code. Run it from a clean copy before saying it works.
3. No code comments, anywhere, including in shipped ZIP projects.
4. Short sentences, active voice, short words where they are as exact. Define a symbol at first use. Every equation is followed by "In words: ...". Spell out an abbreviation at first use.
5. Prefer a table to a long comparison, a diagram to a table of flows, an example to a definition.
6. Every technique gets a failure case. Say when it does not help and why, and show it when you can.
7. Keep the source's examples and analogies. Add your own only where the source has none.
8. British spelling. No em dashes as connectors. No GitHub references (github.com or github.io). `<details>` in block form.

---

## 5. Chapters built from a source (lecture, video, paper, book)

### Contract first

| Question | Default |
| --- | --- |
| One file per source unit, or reorganised by topic? | One file per unit. Do not merge two videos into one page unless told to. |
| Whose titles? | The source's own, verbatim in `title`, with a short numbered `sidebar_label`. |
| Follow the source's flow, or restructure? | Its flow (below). |
| Anything added beyond the source? | Allowed, but labelled as an addition (section 8). |

"Exactly the same as the video" means the same topics, order, examples, analogies and code, nothing skipped, nothing reordered. It does not mean transcribing.

### Follow the flow, as teaching content

Mirror the lecture's shape with plain headings that name the idea (never a time, never "the instructor"): why the thing is needed, what it is, the plan, each demo named by what it teaches, what comes next. Keep their examples, analogies (the load-bearing teaching device), warnings and order. Their code is kept, made correct and runnable; if it has a bug, ship the working code and explain the correct behaviour in a `:::note`. Do not skip a section because it seems minor, reorder for tidiness, or swap their example for one you prefer.

### Complete and in order (the most common failure)

The owner has found, more than once, that conversions from a transcript skip passages and do not follow the lecture's own order and sentences. The rule is: **every statement the lecturer makes appears in the notes, in the order and in the sentence structure the lecturer used.** Statement 1 of the lecture is statement 1 of the notes. The notes do not merge, drop, reorder, summarise away or "improve" statements.

What must survive, each in its original place:

- every definition, example, analogy, rule of thumb, number, parameter, command, name and warning;
- every aside and every correction the lecturer makes, as teaching content (a correction goes in a `:::note Correction`, neutral, section 2);
- every demo step and every line of demo code, in the order shown, with the same variable names;
- the lecturer's own phrasing for definitions, analogies and memorable lines, as a short quotation or a very close rendering, because that wording is part of what is being taught.

What may be dropped, and only these: greetings and sign-offs, pure filler ("right?", "okay so"), requests to like or subscribe, sponsor and course promotions, repetition of a sentence already written, and talk about the recording itself. List what you dropped, by block, in your report.

The method:

1. Build the English working transcript first (private file under `.lecture-import/`), with its blocks numbered. For another language, translate every block; do not summarise while translating.
2. Make a coverage ledger: one line per statement, example, number, command and warning in the block, in order. Write the notes block by block against the ledger.
3. Write each statement in clear English, one note sentence for one lecture sentence where you can, adding explanation after the statement and never in place of it. Explanation, worked examples, code output and diagrams are added around the lecture's content, not instead of it.
4. Run the coverage check and fix every line it reports:

```bash
python3 .lecture-import/track-c/coverage_check.py <english_transcript.txt> <chapter.md>
```

   `SKIPPED?` means no passage of the chapter matches that transcript block. `MOVED?` means the passage appears earlier in the chapter than the blocks before it. Each is a lead, not a verdict: either fix the chapter, or explain in the report why the block is filler. A clean run is required before you report the chapter done.
5. Finish with a manual walk: read the transcript block by block next to the chapter and tick each ledger line. Report the block count, how many are covered, and the dropped filler blocks.

### Own words, without losing the lecture

Notes are written in your own words and must still teach. That does not allow skipping or re-ordering. Never paste whole transcript passages and never reproduce a long stretch of the lecturer's speech; render each statement in clear English, close to the lecturer's phrasing, and keep short quotations of defining sentences. The test: could someone read the chapter instead of watching and meet every point the lecturer made, in the same order, and learn the same thing? Could they tell who the speaker was or how the screen looked? That fails (section 2).

### Source line and frontmatter

```markdown
> **Video 5 of 21** (playlist video 3) · [Watch on YouTube](https://www.youtube.com/watch?v=...)
```

```yaml
---
id: langchain-models
title: "The source's own title, verbatim"
sidebar_label: "5 · Models"
sidebar_position: 5
slug: /genai/models
description: "One sentence on what the chapter delivers, written for search results."
tags: [langchain, models]
---
```

Numeric filename prefixes match `sidebar_position`. A folder or file prefixed `_` is ignored by Docusaurus; stash superseded drafts in `_old/` instead of deleting the user's work. Ids and slugs are unique site-wide, and so are category labels.

### Getting transcripts

```bash
yt-dlp --extractor-args "youtube:player_client=android" --skip-download --write-auto-subs --sub-langs "en" --sub-format json3 -o "subs/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=VIDEO_ID"
```

- The default and `tv` clients fail. Use `android`; rotate `ios`, `mweb`, `web_embedded`, `android_vr` across rounds.
- Never fetch in parallel. Loop sequentially with 6 to 8 second sleeps.
- When yt-dlp is rate-limited (HTTP 429), use the `youtube-transcript-api` package (`YouTubeTranscriptApi().list(id)`, then `find_transcript([...])`, `translate("en")` if `is_translatable`, else fetch the original and translate while writing). On this machine it installs under `/usr/bin/python3`. Fetching a caption `baseUrl` directly returns an empty body; do not try.
- Merge caption fragments into blocks of 30 to 45 seconds before reading.
- Private working files (transcripts, frames, review notes) stay in a git-ignored folder under `.lecture-import/`, never in `docs/` or `static/`.

---

## 6. Diagrams and labs

**Boards.** Original board-style SVGs, never Mermaid for flagship diagrams and never extracted video frames or screenshots. Use the kit `scripts/infographics/board.py` (`Board`, `group`, `card`, `cylinder`, `diamond`, `pill`, `text`, `table`, `arrow`) with one script per track in `scripts/infographics/<track>.py`, output `static/img/<track>/<name>.svg`. Place with `<Infographic src="/img/<track>/<name>.svg" alt="what it shows" caption="where to look first" />`. Render every board with headless Chromium, look at it, and fix clipped text, text outside the viewBox and arrows that cross cards. Every number on a board equals a printed number.

**Mermaid** is acceptable for small flows inside prose; every diagram gets the Expand lightbox automatically. Use `<br/>` for line breaks and `<b>` for emphasis, label conditional edges, one idea per diagram.

**Labs.** One lab for every concept that moves. Files `src/components/viz/<Name>Lab.tsx` built on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop for the data view, deterministic seeds, keyboard operable, sensible `aria-label`s. Put the maths in a `.ts` module and cross-check it against the Python numbers. Create the lab file before any chapter imports it. Do not edit `VizPanel*`, `palette.ts`, `CourseLab*` or another agent's lab. Typecheck with `npx tsc --noEmit`.

---

## 7. Docusaurus mechanics that bite

- **MDX reads `{...}` in prose as JSX.** Put placeholders in code fences or inline code, or write `\{x\}`. A bare `<` followed by a letter in prose is read as a tag; use backticks. Currency is `\$`.
- **Internal links need the `/docs` prefix**: slug `/genai/models` is served at `/docs/genai/models`. `onBrokenLinks` is `throw`, so one bad link fails the whole build. Check with `python3 .lecture-import/track-c/link_check.py`.
- **Callouts** for things that change what the reader does: `:::tip` a better default, `:::note` a clarification or a correction, `:::warning` a trap, `:::danger` money, data loss or security.
- Never run a whitespace-collapsing regex over a markdown file; it destroys code indentation.
- Never save a chapter that imports a lab or image that does not exist yet.

---

## 8. Honesty on the page

- A source you could not obtain, a command you could not run, a claim you could not verify: say so in a `:::warning` and in the report.
- Anything not from the source gets a marker: `:::note Added for this site`.
- Corrections to the source: `:::note Correction`, neutral, no attribution.
- Name licences for data and models, read from the card or README. Do not redistribute data whose licence forbids it; download at run time into a temp folder.
- Never invent numbers, versions, benchmarks, prices or "company X does Y".

---

## 9. Environments, ownership and build

- Chapter code runs in `.lecture-import/venv-llm/bin/python` (torch CPU, transformers, scikit-learn, scipy, pandas, duckdb, opencv, statsmodels, networkx and more). `venv-ml` is the lighter numeric one. Install another library with that venv's `pip` only, and list it in your report. Hugging Face downloads that stall: rerun with `HF_HUB_OFFLINE=1` once cached and say so.
- Run chapter code with `python .lecture-import/codetest/run_all.py <folder> <python>`. It executes every block under the runnable headings. Read the printed output; a zero exit code is not enough.
- Do not run `.lecture-import/regen.sh` (it wipes other work). Work in your own directories.
- Edit only the folders you were assigned. Never edit `src/**` other than new lab files, `sidebars.ts`, `docusaurus.config.ts`, `package.json`, `learningPath.ts` or a `_category_.json` you did not create. Ask in the progress file for anything that needs wiring.
- Build in isolation: `DOCUSAURUS_GENERATED_FILES_DIR_NAME=.docusaurus-<yourname> npx docusaurus build --out-dir .lecture-import/build-<yourname>`. If it fails only on other people's unfinished pages, say so and verify a copy of the tree without them.
- Do not commit or push unless the user asks.

---

## 10. Verify and report

Before you say a chapter is done:

1. Every Python block ran (`run_all.py`) and every number in the prose equals a printed number.
2. `python3 .lecture-import/track-c/quality_gate.py <your folder>` passes. Fix every FAIL. Treat warnings as work to do.
3. `python3 .lecture-import/track-c/link_check.py` reports no broken links or images.
4. `npx tsc --noEmit` is clean if you added a lab.
5. You looked at every board. Where a browser is available, you opened the page at desktop and 390 px width, operated each lab, and saw no console errors. If you skipped this, say so.

Report in the progress file and in your final message: what you added per chapter, the key printed numbers, the surprising result, sources with dates and versions, the gate line, and exactly what you did not verify. Report problems as plainly as successes.

After a large import or batch, update `.codex/senior-ai-progress.md` with what was done, the tooling workarounds that worked, and what is outstanding. The next session will not have this context.
