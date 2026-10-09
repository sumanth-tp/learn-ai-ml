# Writing a chapter from a YouTube video

Start with a finished source pack (`youtube-source-pack.md`). This guide turns the pack into a chapter that teaches everything the video teaches, in its order, as proper notes, and then adds what the video lacks: worked examples with numbers, experiments that were actually run, boards, a lab and a project.

Read these first: `.codex/AGENTS.md` sections 1 to 5, and `notes-voice.md`.

## 1. Agree the contract

| Question | Default |
| --- | --- |
| How many pages? | One per video. A playlist gives one page per video, in `playlist.tsv` order. |
| A very long single video (2 hours or more)? | One page per description chapter group of about 30 to 60 minutes, split at the description's own chapter boundaries. Each page gets "Part N of M" in its source line. |
| Titles? | The video's own title, verbatim, in `title`; a short numbered `sidebar_label`. |
| Order? | The video's own order. Statement 1 of the video is statement 1 of the notes. |
| Additions? | Allowed and expected (worked example, experiment, lab, boards, project), each one marked `:::note Added for this site` when it is more than a sentence. |

If the owner said otherwise for this job, their instruction wins. Write it at the top of your progress notes.

## 2. Know which source wins

The video, the frames, the repo and the description can disagree. Settle each conflict like this:

| Conflict | What to do |
| --- | --- |
| The video and the repo differ (the speaker edits the code on camera, renames a variable, changes the model) | Follow the video, because that is what is taught. Mention the repo version only if it matters. |
| The video's code has a bug or uses a removed API | Ship code that works today, and explain the correct behaviour in a `:::note Correction`, without attribution. Pin the versions you ran. |
| A claim in the video is wrong or out of date | State the correct idea, with a source you opened today, in a `:::note Correction`. |
| The transcript and the frame disagree about a name or a number | The frame wins for what was on screen. Your run wins for any output. |
| The description's chapter titles do not match what is actually covered | The content wins. Use the titles only as a skeleton. |
| The video says something you cannot verify | Keep it as the video's teaching, hedged, and add a `:::warning` saying what you could not verify. |

## 3. Build the ledger before writing

Open `ledger-en.md` next to `blocks-en.txt` and the contact sheets. For each block, write in a private file the claims it makes, one line each. A claim is a definition, an example, an analogy, a rule of thumb, a number, a parameter, a command, a name, a warning, a question raised, a demo step, or a correction. Mark the frames that go with it.

```
B014 [00:09:20] f_00057-f_00060
  - claim: an LLM call is stateless; history must be resent every turn
  - analogy: a waiter with no memory who reads the whole order back each time
  - demo: messages list with system, user, assistant; second call includes the first answer
  - code: frames f_00058 (cell 7), notebook cell 12
  - question raised: does resending history cost tokens? answer: yes, every turn
```

The ledger is where the coverage rule is won or lost. Every later step checks against it.

## 4. Plan the spine

1. The outcome sentence: after this chapter the reader can do one concrete thing.
2. The hardest idea in the video. Plan a small worked example with numbers for it, done by hand, which the code later reproduces.
3. Headings from the description's chapters, renamed to name ideas, filled in ledger order.
4. Where each board goes (every slide or whiteboard in the sheets, plus at most one explanatory board for a mechanism explained only in words).
5. Which concept moves enough for a lab (`labs.md`).
6. The experiment you will run to produce the printed numbers. It is often the video's own demo made runnable and measured.
7. The topic project this chapter feeds (`projects.md`).

## 5. Write block by block

Work through the ledger in order. For each block:

- Write each claim as a statement about the subject, in the video's order and close to the speaker's phrasing for definitions and analogies (`notes-voice.md`). Keep the speaker's example and analogy. Add your own only where the video has none.
- Resolve every "this", "here" and "like this" from the frames.
- Add explanation after the claim, never instead of it: why it is true, a small example, what goes wrong.
- When the block has code, put the code in a runnable block under `## Code you can run` (or inline if it is a one-line command), and follow it with **Reading the output** and **Line by line**. Keep the video's variable names.
- When the block shows a picture, place the board there.
- Tick the ledger line.

Use the chapter shape in `.codex/AGENTS.md` section 3 and `templates/chapter.md`. The video's content sits in its own order in the middle of that shape. The opening sections (In one line, Before you start, In 30 seconds, Words you will meet, big-picture board, worked example) are written last, once you know what the chapter covers.

## 6. Make the code real

1. Copy every code cell from the ledger into a scratch project under the pack, in order.
2. Make it run in `.lecture-import/venv-llm/bin/python` (or a throwaway `uv` venv for libraries you must not add to the shared venv). Fix what is broken, using the current API.
3. Replace calls to paid APIs with a local or offline path where the lesson allows it (Ollama, a small Hugging Face model, a fake model with fixed answers). Keep the real call on the page as well, and say in a `:::warning` that the real path was not run, if it was not.
4. Seed everything. Run it twice and check the numbers repeat.
5. Paste the code into the chapter with no comments, then run `python .lecture-import/codetest/run_all.py <folder> <python>` and read every printed line.
6. Every number in the prose must equal a printed number. If your run disagrees with the video, your run is what the page says, and the difference is worth a sentence.

## 7. Boards from frames

For each slide, whiteboard or diagram in the sheets: read the full frame, then redraw it as an original board with the kit (`boards.md`). Keep its structure, labels and colour grouping where they teach something, and fix its mistakes. Never paste the frame. Caption the board with where to look first, not where it came from.

## 8. Additions that make it teach

The video is the floor, not the ceiling. Add these, marked as additions:

- The worked example with hand arithmetic, before any code.
- An experiment that measures something the video only asserts. Look for the surprise: a setting that does not matter, a method that loses.
- A lab for the concept that moves.
- Common mistakes, practice questions and Check yourself (the gate requires them).
- A short "Production notes" section when the video stops at a demo: what breaks at scale, cost, security.

Do not add a different example in place of the video's. Add yours next to it.

## 9. Playlists and series

- One page per video, `sidebar_position` equal to the playlist index, filename prefix matching.
- A video that opens with a recap of the previous one: keep the recap as two or three sentences that link the earlier chapter. Do not teach it again.
- A video that promises "next video we will ...": nothing in the body; **Where to go next** links the next chapter.
- Repeated setup across videos (the same `.env` and installs): teach it fully in the first chapter, then link back to it.
- Write `playlist.tsv` progress into the progress file after each video, so a stopped session can resume.

## 10. Finish

Run, in order, and fix until clean:

```bash
python3 .lecture-import/track-c/coverage_check.py $P/blocks-en.txt docs/<folder>/<chapter>.md
python3 .lecture-import/track-c/quality_gate.py docs/<folder>/<chapter>.md
python .lecture-import/codetest/run_all.py docs/<folder> .lecture-import/venv-llm/bin/python
python3 .lecture-import/track-c/link_check.py
```

Then do the review in `fidelity-review.md` on your own chapter, as if someone else wrote it. Write the report from `templates/source-report.md`.
