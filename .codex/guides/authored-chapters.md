# Writing a new chapter with no source

Some chapters have no video or lecture behind them: gaps in the curriculum (`.codex/senior-ai-plan.md`), additions the owner asks for, and topic projects. With no source to follow, the risk changes. A video-based chapter tends to narrate. An authored chapter tends to come out as a confident summary of things the writer half-remembers. The remedy is to research, run and measure before writing.

The rulebook (`.codex/AGENTS.md` sections 1, 3 and 4) is the standard. This guide is the order of work.

## The order of work

### 1. Place it

- Find the chapter's place in the learning path: what the reader finished just before, and what comes next. Link both in **Before you start** and **Where to go next**.
- Write the outcome sentence: "After this chapter the reader can ...". It has to be a capability, not a topic.
- Read the two neighbouring chapters, so you do not repeat them or contradict their terms.

### 2. Research, and keep a log

Open primary sources today: official docs, the paper, the model card, the specification. Save each with `web_extract.py` (`web-sources.md`). Keep a research log in the job folder:

```
2026-10-09  scikit-learn 1.9.1 docs, TimeSeriesSplit   gap parameter exists; default 0
2026-10-09  Hyndman & Athanasopoulos, FPP3, ch. 5.10    time-series CV: rolling origin
2026-10-09  NOT FOUND: a source for "random splits overstate R squared by 0.2 to 0.3"; will measure instead
```

Anything you cannot source, you either measure or leave out. Versions, prices, benchmark scores and "company X does Y" need a source opened that day, or they do not appear.

### 3. Run the experiment first

Write the experiment before the prose: 50 to 150 lines, a real third-party library, seeded, under a minute (`.codex/AGENTS.md` section 1, step 3). Run it. Read the output. If it contradicts what you expected, that is your best paragraph.

Good experiments here compare things: two methods, a setting swept over a range, an honest split against a leaky one. A single happy-path run teaches least.

### 4. Find the hardest idea and the surprise

Write down the one idea beginners get wrong in this topic, and the most surprising line your experiment printed. The chapter is built around the first, and the second is what makes it memorable.

### 5. Write in the shape

Copy `templates/chapter.md`. Fill the middle first: the idea in plain words, the worked example by hand, then the code that reproduces the hand numbers, then the experiment. Write the opening sections (In one line, In 30 seconds, Words you will meet) last.

### 6. Pictures and a lab

At least two boards (`boards.md`): a big-picture one near the top, and a step-by-step one by the worked example. One lab (`labs.md`) whose defaults reproduce a printed number.

### 7. Feed a project

Every chapter is exercised by at least one of its topic's projects (`projects.md`). If none uses it yet, add a task to a project or note the gap in the progress file.

### 8. Verify and report

`.codex/AGENTS.md` section 11, unchanged. The report names the surprise, the key printed numbers, sources with dates and versions, and what was not verified.

## A test for an authored chapter

Pick three sentences at random. For each, can you point to the run that printed it, the source that states it (opened and dated), or the arithmetic on the page that derives it? A sentence with none of the three is either general explanation, which is fine, or an unsupported claim, which goes.
