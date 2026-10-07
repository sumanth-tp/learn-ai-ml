# How to write chapters at the depth of the best ones (mandatory for Codex)

The user compared Codex's chapters with Claude's and found Codex's thinner. The measured gap (2026-10-07, `.lecture-import/track-c/readability_baseline_2026-10-07.txt`): the Codex-written lecture chapters average 14 to 26 lines of code, one or two blocks, almost no real library, and no experiment whose output is quoted. The best chapters run 100 to 300 lines of code, use real libraries, quote what they printed, and say what surprised the author.

This file is the playbook. `.codex/beginner-friendly-standard.md` says what shape a chapter has. This file says how to fill the shape so it teaches. The gate at the end is automatic: a chapter that fails it is not done.

## The four habits that make the difference

### 1. Run something real, and write what happened

For every chapter, before writing prose, write and run one experiment of 50 to 150 lines with the real library the topic belongs to (scikit-learn, statsmodels, torch, opencv, transformers, langchain, and so on). Not a toy loop that imitates the library. Then:

- Quote the printed numbers in the chapter, under "Reading the output".
- Include one result that is honest about what did not work or surprised you: "hybrid retrieval did not beat keyword search on 17 chunks (0.88 against 0.92 recall)", "int8 was slower than float32 on this CPU".
- Never write a number you did not print. If you cannot run something, say so in a `:::warning` on the page.

A chapter with no run experiment is a summary of a lecture, not a chapter.

### 2. Explain with a small example before the idea

Order every concept the same way:

1. An everyday situation in two or three sentences.
2. The smallest example with numbers a reader can follow by hand, with the arithmetic shown.
3. The general statement, followed by "In words: ...".
4. The code that reproduces step 2's numbers exactly.

Worked example from the capstone chapter: before any code, one question is followed through seven steps, and the reciprocal-rank-fusion scores are added up by hand (0.6 / 61 + 0.4 / 63 = 0.016185), so when the code prints 0.016185 the reader is not surprised.

### 3. Teach the code, not just show it

After every code block, write both of these, in prose (never as code comments):

- **Reading the output**: what each printed value means and what a wrong value would look like.
- **Line by line**: bullets for the non-obvious lines only, saying why that line exists.

Keep blocks to about 40 lines and split longer code, with one sentence before each block saying what it is about to do and why. A reader who skips the code must still learn the lesson from the prose.

### 4. Name the failure

For every technique, say how it breaks. Not "reranking improves quality", but "a cross-encoder reranker lifts precision at 5 but cannot recover a passage the first stage never retrieved, so recall of the first stage caps everything". Every chapter ends its main text with 3 to 5 common mistakes in the form: the mistake, why it feels right, what to do instead.

## Style, with before and after

| Before (thin) | After (taught) |
| --- | --- |
| "ARIMA models autocorrelation with an integrated moving-average structure." | "Imagine forecasting tomorrow's sales. Today's sales tell you a lot, and last week's sales tell you a bit. ARIMA writes that as a weighted sum of the last p values and the last q forecast errors. With p = 1 and a weight of 0.8, tomorrow = 0.8 x today. In words: each value leans on the one before it." |
| "Use cross-validation to evaluate." | "Random splits leak the future into the past on a time series. We split by time instead, and on this data the random split scored 0.97 R squared against 0.71 for the honest split. Same model, same data." |
| "Quantisation reduces model size." | "Weights stored as 32-bit floats take 4 bytes each. Stored as 8-bit integers they take 1. A 7B model goes from 28 GB to 7 GB. We measured speed too: on this CPU int8 was 1.3x slower, because the matrix routine was not optimised for it." |
| A list of seven bullet points, each a fragment. | One idea per paragraph, at most four sentences, with the bullet list reserved for steps and parameters. |

Rules that follow from those examples:

- Short sentences, active voice, average under 22 words. Say the plain word where it is as exact.
- Define a symbol the first time it appears. Spell out an abbreviation at first use.
- Prefer a table to a long comparison, a diagram to a table of flows, an example to a definition.
- Analogies from the source lecture are kept. Add your own only when the lecture has none.
- Never criticise a speaker. State the correct idea and explain it. Where the source is wrong, put the correct information in a `:::note` without naming anyone's mistake.
- No em dashes as connectors, British spelling, no code comments, no GitHub links, no invented companies or figures.

## Chapters to imitate (read two before writing)

| Chapter | What to copy |
| --- | --- |
| `docs/genai/23-capstone.md` | Worked example before code, file-by-file walkthrough, honest evaluation with named failures, labs with "Try it yourself" that quote exact numbers |
| `docs/llm-engineering/02-inference-and-serving/02-kv-cache-and-paged-attention.md` (predates the section standard, so the gate fails its section names; copy its depth, not its headings) | Formula checked against real model configs, every figure printed by a block, practice questions with reasoning |
| `docs/llm-engineering/01-adapting-models/03-supervised-fine-tuning-with-lora.md` | A real training run on CPU, memory arithmetic, surprises recorded |
| `docs/theory/ml/03-ensembles-and-unsupervised-learning/02-gradient-boosting-in-practice.md` | Measured comparison across libraries, a leakage trap shown and fixed |

## Diagrams and labs

- At least 2 boards per chapter, drawn as original SVG with the kit in `scripts/infographics/board.py`. Render each one, look at it, and fix overflow, clipped text and arrows that cross cards. A big-picture board near the top; a step-by-step board for the worked example.
- Every board has alt text that says what it shows and a caption that says what to look at first.
- A lab for each idea that moves. The lab's default settings reproduce a number the chapter prints, the chapter lists what each control does, and gives three guided experiments ("set X to Y, watch Z change, here is why") whose results you actually checked in the browser.
- Put the lab's maths in a `.ts` module and cross-check it against the Python numbers.

## Process (do it in this order)

1. Read the source (lecture, paper, docs). Note its examples, analogies, code and warnings; they stay.
2. Write the experiment, run it, keep the output.
3. Write the worked example by hand and check it against the output.
4. Write the chapter in the required order, quoting the output.
5. Draw and inspect the boards. Build and check the lab.
6. Run the gate below. Fix every failure. Only then build the site and report.

## The gate

```bash
python3 .lecture-import/track-c/quality_gate.py docs/path/to/your/folder
```

It fails a chapter that has: fewer than 80 lines of Python, no real-library import, fewer than 2 boards, no lab, a missing required section (before you start, in 30 seconds, words you will meet, a worked example, common mistakes, practice questions, check yourself), code comments, an inline one-line `<details>`, a GitHub link, or an em dash used as a connector. It prints warnings, which you should also fix, for long sentences and low readability. Paste the gate's final line into your report in `.codex/senior-ai-progress.md`.

The gate checks shape, not truth. You are still responsible for running every claim you make.

## Exemplars to copy (taken from the capstone chapter)

**Opening a concept with a person, not a definition.**
"Imagine a new colleague who has read your company handbook. You ask a question. A good colleague finds the right page, tells you the answer and shows you the page. A bad colleague guesses and sounds sure."

**Doing the arithmetic by hand before the code.**
"Chunk A is 1st in the meaning list and 3rd in the word list. Chunk B is 2nd and 1st. With weights 0.6 and 0.4 and k = 60: A scores 0.6 / 61 + 0.4 / 63 = 0.009836 + 0.006349 = 0.016185. B scores 0.6 / 62 + 0.4 / 61 = 0.016235. B wins by a hair. In words: each list gives a chunk a small score that shrinks with rank, and the scores add up."

**Reading the output, including a result that disappoints.**
"Hybrid retrieval did not beat keyword search here (recall 0.88 against 0.92). On a library of 17 chunks the top 5 is almost a third of everything, so every method finds the right file. This is a useful result, not a failed one. Measure it on your documents before you adopt a technique."

**Line by line, only for the non-obvious lines.**
"`ids=[chunk.metadata["citation"] ...]` is the key detail. Giving Chroma an id means that adding the same chunk twice replaces it. Without ids, every ingestion duplicates the library and the same passage fills your top 5."

**Naming the failure, with the cause.**
"The stand-in refused q21 although the right file was retrieved. That is a generation failure, not a retrieval failure: the report says 'down' and the question says 'unavailable'. Good recall with a bad answer points at the prompt or the model, not at the retriever."

**A lab with exact guided experiments.**
"Set the chunk size to 200 and the overlap to 0. You now get 10 chunks of about 136 characters. Small chunks are precise, but a chunk this small often cuts an answer in half, and you pay for 10 embeddings instead of 2."

**A warning that says what was not verified.**
"All 75 tests were run. The real-model path was written against the documentation but not run against a live provider. Run it with a real key before you trust it."

## Constraints I work under (copy these)

1. **Every number is printed or computed.** No figure, version, price, benchmark or model name that I did not run or read from a source I opened that day. Versions are recorded with the date.
2. **The chapter's code is the tested code.** Generate the chapter from the real files where you can, and run the code from an unzipped copy before saying it works.
3. **No code comments, anywhere.** All explanation lives in the prose around the code, in "Line by line".
4. **Short blocks.** About 40 lines of code per block, one sentence before each saying what it does and why.
5. **Short paragraphs.** At most four sentences and about 90 words, one idea each. Average sentence under 22 words.
6. **Concrete before abstract.** Everyday situation, then a small example with numbers, then the general rule, then "In words: ...".
7. **Every technique has a failure case.** Say when it does not help and why, and show it when you can.
8. **Be honest on the page.** If something could not be obtained, verified or run, a `:::warning` says so. Never silently drop a section or invent a replacement.
9. **Respect the source.** Keep the lecturer's order, examples, analogies and code. Explain in your own words, never transcribe. State corrections neutrally without naming anyone's mistake, and label anything you added as an addition.
10. **Diagrams are original SVG boards that I render and look at.** I fix clipped text and crossed arrows before saving. Alt text says what the picture shows; the caption says where to look first.
11. **Labs reproduce the chapter's numbers.** I check them in a real browser, not just by building.
12. **House style.** British spelling, no em dashes as connectors, no GitHub links, `<details>` in block form, internal links with `/docs` prefix, no `{...}` in prose.
13. **Verify before reporting.** Typecheck, build, run every Python block, run the quality gate, check the page in a browser at desktop and phone width. Report what was not verified.
14. **No video timestamps in the notes.** Not in headings ("## 04:57 · ..."), not in sentences ("At 06:23 the instructor ..."), not in link text, not in `&t=` links, not in a timestamp table. The notes teach the subject; they are not a navigation index of the video. The one allowed pointer to a video is the single source line at the top of the chapter. Do not narrate the video either (no "on screen", "in the frame", "he now opens"): write the explanation.
15. **Small steps, one owner per file.** Stay inside the folders you own and append to the progress file instead of rewriting it.
