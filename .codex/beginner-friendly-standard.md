# Beginner-friendly standard (added 2026-10-05 at the user's request)

"Make content more readable and beginner friendly with detailed explanation and working labs and diagrams."

This applies to every chapter in `docs/`: new chapters you write, and existing ones you edit. It adds to the other gates (G1 to G10); it never lowers them. A reader who has finished only the
previous stage of the learning path must be able to follow the chapter without opening another tab. Depth stays: the senior-level material is still there, but it is reached by a ramp.

## Required shape (in this order)

1. `**In one line.**` (keep).
2. `:::tip Before you start` with: what you should already know (2 to 4 bullets, each linking to the chapter that teaches it), reading time, and "after this chapter you can ..." (2 to 3 outcomes).
3. `## In 30 seconds`: 3 to 5 sentences, no jargon, one concrete everyday example and one analogy.
4. `## Words you will meet`: a table (Term, Plain meaning, Tiny example) for 5 to 10 terms the chapter uses. Every other new term is glossed in brackets at its first use.
5. `## The idea in plain words`: start from an everyday situation, then the smallest example with numbers you can follow by hand, then generalise. Why before how.
6. A **big-picture board** near the top, and a **step-by-step board** for the worked example. Every board: alt text that says what it shows, and a caption that says what to look at first.
7. `## Worked example, step by step`: numbered steps with the arithmetic shown, small numbers, before any code. The code later reproduces these exact numbers.
8. `## How it works` (existing content stays), with sub-headings that are questions or plain statements, not jargon labels.
9. `## Code you can run`: blocks of at most about 40 lines (split longer ones). Before each block, one sentence: what we are about to do and why. After each block: **Reading the output** (what each printed number means) and **Line by line** (bullets for the non-obvious lines). Code comments are still forbidden, so all explanation lives in the prose around the code.
10. Each lab is embedded with **What each control does** (one line per control) and **Try it yourself**: 3 numbered guided experiments, each "set X to Y, watch Z change, here is why". The lab's defaults reproduce a number the chapter prints.
11. `## Designing with it`, `## Where this stands in 2026` (keep).
12. `## Common mistakes`: 3 to 5 items, each "the mistake, why it feels right, what to do instead".
13. `## Practice questions`: block-form `<details>`, labelled Easy, Medium or Stretch, answers explain the reasoning.
14. `## Go deeper` (sources), `## Check yourself` ("I can ..."), and `## Where to go next` (links to the next chapter and one related chapter).

## Writing rules

- Paragraphs of at most 4 sentences and about 90 words. One idea per paragraph. Average sentence under 22 words. Active voice. Short words where a short word is as exact.
- Define a symbol the first time it appears. Every equation is followed by "In words: ...".
- No unexplained abbreviation. First use is spelled out with the abbreviation in brackets.
- Prefer a table to a long list of comparisons, a diagram to a table of flows, an example to a definition.
- Keep British spelling, no em dashes as connectors, no code comments, no GitHub references.

## Measured targets

| Measure | Target |
| --- | --- |
| Flesch reading ease of prose (code, tables and maths removed) | 50 or higher |
| Average sentence length | 22 words or fewer |
| Average paragraph length | 90 words or fewer |
| Has the required sections 2, 3, 4, 7, 12 | all five |
| Every embedded lab followed by "Try it yourself" | yes |
| Boards per chapter | 2 or more |

The audit script that measures these is still to be written (`.lecture-import/track-c/readability_audit.py`).

## Depth bar (measured 2026-10-05; applies to every new chapter)

Chapters written so far differ sharply in depth. Claude's authors' chapters average 145 to 205 lines of runnable code per chapter (3 to 6 blocks, 0.8 to 4.9 real libraries) and about 4,200 words.
The Codex-written lecture chapters (IR, data management, computer vision, time series, recommenders) average 14 to 26 lines (2 blocks, almost no real library) and about 2,600 words. New chapters
meet the higher bar: at least one experiment of 50 to 120 lines using the real library, run, with its printed numbers quoted and one result that is honest about what did not work, and about 3,500 to
4,500 words including code.

## Before and after (the pattern, not a template to copy)

Before: "Exponential smoothing updates a state with a convex combination of the latest observation and the previous level."
After: "Imagine you are guessing tomorrow's temperature. You trust today's reading a little and your old guess a lot. Exponential smoothing does exactly that: new guess = 0.5 x today + 0.5 x old guess. With readings 10, 12, 11, 13 and a first guess of 10, the guesses are 10, 11, 11, 12. The weight 0.5 is called alpha. In words: a bigger alpha trusts today more."

The simplicity comes from the example, not from removing the idea. Keep the precise statement too, one paragraph later, once the reader has the picture.

## How to fill this shape well

This file is the shape. `.codex/write-like-claude.md` is the craft: run a real experiment, explain with a small example before the idea, teach every code block, name the failure. The measured gate is `.lecture-import/track-c/quality_gate.py`.
