# Turning speech into notes

This guide fixes the most common failure in chapters built from a video or a live class. The page reports what the speaker said and did ("he says", "the class guessed", "like this", "that was the session") instead of teaching the subject. `.codex/AGENTS.md` section 2 states the rule. This guide shows how to follow it, sentence by sentence.

The gate (`quality_gate.py`) now fails both kinds of leak: the speaker ("the instructor", "he says") and the room ("a question from the class", "this session shows", "if remembered correctly", "shared the next day"). It warns on pointing words ("as you can see", "this thing", "over here"). Passing the gate does not prove the voice is right. The test at the end of this guide does.

## Why it happens

A transcript is a record of speech acts: someone explains, asks, types, scrolls, jokes, corrects themselves. Writing from it sentence by sentence copies the speech act along with the content. "He explains that overlap keeps sentences whole" is a faithful summary of the transcript and a bad note. The reader wanted the second half of that sentence only.

The fix is one question per statement: **what does this tell the reader about the subject?** Write that. The speech act around it is packaging, and it goes.

This does not conflict with the coverage rule ("every statement, in order"). The ledger tracks claims, not packaging. "He explains that overlap keeps sentences whole" contains one claim: overlap keeps sentences whole. That claim must appear in the chapter, at that point. "He explains" must not.

## The eight leaks and their fixes

### 1. The speaker as subject

| Leak | Note |
| --- | --- |
| He sets the chunk size to 1000 and explains that overlap is needed because a sentence may be cut. | Split the text into chunks of 1,000 characters with a 200-character overlap. A cut can land mid-sentence; the overlap repeats the last 200 characters at the start of the next chunk, so the sentence is whole in one of them. |
| He copies that activation line into the terminal. | Activate the environment in a terminal: (code block). |
| He restarts the kernel and re-runs the cells. | If an import fails right after `pip install`, restart the kernel and run the cells again. The kernel loaded the old environment when it started, so it cannot see the new package. |
| He hovers over `Annotated` in VS Code and reads the tooltip. | `Annotated` (from `typing`) attaches extra metadata to a type. LangGraph reads that metadata to learn how to merge a field. |

The pattern: the verb the speaker performed becomes an instruction to the reader (imperative), or a statement about the thing (present tense). The reason the speaker gave becomes the "because".

### 2. The room: class, audience, chat

| Leak | Note |
| --- | --- |
| A question from the class: without column descriptions, how does the LLM understand what the tables hold? | A heading or a callout that asks the question itself: `### How does the model understand a table with no descriptions?` Then the answer as content. |
| The class threw out numbers (30 lakh, 2 lakh, 5 lakh) for a monthly budget. The business manager says stay under ₹3 lakh. | The monthly budget is a business decision, not a technical one. Estimates for a feature like this range from 2 lakh to over 30 lakh a month. This example fixes it at ₹3 lakh; every model choice below has to fit under it. |
| The class suggested 5 seconds, which feels too long. Two to three seconds is good. | Aim for a response in 2 to 3 seconds. At 5 seconds a user starts to wonder whether something has broken. |
| While waiting, some questions from the class: | Delete the frame. Each question becomes a short `###` section or an item in Common mistakes, at the point where it was raised. |

Audience questions are often the best teaching in a live class, because they voice the reader's own confusion. Keep every one. Rewrite each as the question a reader would ask, followed by the answer.

### 3. The session, the recording and lecture time

| Leak | Note |
| --- | --- |
| This session shows the whole selection process on one case study. | **In one line.** Choosing a model for an application is a measured process: requirements, a shortlist, a custom evaluation and a decision. |
| Until a few days before this session, every model scored in single digits. | Until November 2025, every model scored in single digits. (Take the absolute date from the upload date in `description.md`, or from the source you checked. If you cannot date it, say "at the time of writing (2026-10)".) |
| That was the session, and its goal was achieved. The code will be shared the next day. | Delete. The code is on the page. |
| As we saw in the last lecture, ... | As the chapter on [embeddings](/docs/genai/embeddings) showed, ... (a link to the chapter that taught it) |
| In the next video we will build the agent. | **Where to go next** links the next chapter. Delete the sentence from the body. |

### 4. The speaker's own voice: hedges, memories, opinions, anecdotes

| Leak | Note |
| --- | --- |
| It is called, if remembered correctly, AskCricinfo. | Check the name in a source you open today, then state it. If you cannot confirm it, describe the feature without the name. |
| Honestly, this feature has never been very popular. | Delete, unless it teaches something. If it does, state the lesson: "A feature buried inside a match page gets a small fraction of home-page traffic, so size the load from the page it lives on." |
| I personally prefer Chroma. | Chroma is a good default for local work: it runs in-process, needs no server and persists to a folder. The reason is the teaching; the preference is not. |
| In my last company we had two lakh users and the cache saved us. | In one deployment with about two lakh users, a semantic cache absorbed most repeated questions. (Keep the anecdote as an example; drop the "I".) |
| No sorry, it is not 4096, it is 8192. | State 8192 only, and check it. A self-correction is not a statement to preserve; its corrected content is. |

### 5. Pointing at a screen the reader cannot see

The transcript says "this goes here, and that comes out like this". The words point at the screen. Open the frame at that moment (`yt_pack.py grab`) and name what is pointed at.

| Transcript | Leak | Note, after reading the frame |
| --- | --- | --- |
| "see, this thing goes into this, and we get this" | The output of this goes into that. | The retriever returns four chunks. They are joined and passed into the prompt as `{context}`; the model answers from them. |
| "as you can see, accuracy came 0.91" | As you can see, accuracy is 0.91. | Accuracy on the test split is 0.91 (printed below). Re-run it: the number must come from your run. |
| "so like this you have to write it" | Write it like this: | Write the schema as a Pydantic model with one field per output key: (code block). |

The gate warns on "as you can see", "this thing", "over here", "shown here" and similar. "Like this" and "this one" are too common to flag automatically, so search for them yourself.

### 6. Narrated process: "first X is opened, then Y is typed"

| Leak | Note |
| --- | --- |
| First a new notebook is opened, then the imports are typed, then the model is loaded. | Steps as an ordered list or as code: "1. Create a notebook. 2. Import the client. 3. Load the model." Better, one runnable block followed by **Line by line**. |
| The demo then switches to the browser and opens the docs. | If the docs teach something, state it and link the page. If not, delete. |

### 7. Logistics, promotion and housekeeping

Delete, and list it in the report as dropped filler: links in the description, "like and subscribe", course or sponsor promotion, "we will take a break", "the code will be shared", attendance, microphone checks, "can you see my screen". A recommendation to practise along can become one line of advice if it adds something ("type each cell yourself rather than reading it").

### 8. Captions and labels that narrate

Board captions say where to look, not where the board came from. "Redrawn from the mentor's whiteboard" becomes "Read left to right: three memory types, then where each is stored." The source line at the top of the chapter already credits the source.

`:::note Not from the session` becomes `:::note Added for this site`.

## A full paragraph, before and after

The transcript block (Hinglish auto-captions, translated):

> so guys see here, what I did, I took chunk size 1000 and overlap 200, why overlap? because suppose a sentence is cut here, the meaning goes, so these 200 characters come again in the next chunk, okay? clear? let's run it, see, 7 chunks came. One student asked can overlap be zero, yes it can be, but then you lose this, okay.

Narrated (rejected):

> He takes a chunk size of 1000 and an overlap of 200. He explains that overlap is needed because a sentence can be cut. When he runs it, 7 chunks are produced. A student asks whether overlap can be zero, and he says yes, but you lose this.

Notes (accepted):

> Split the document into chunks of 1,000 characters with an overlap of 200. A cut can fall in the middle of a sentence, which splits its meaning across two chunks. The overlap repeats the last 200 characters of each chunk at the start of the next one, so a sentence cut at the boundary is still whole in one of them.
>
> (code block, then: "On this document the splitter returns 7 chunks.")
>
> **Can the overlap be zero?** Yes. The chunks are then smaller in total and no text is stored twice, but a sentence that falls on a boundary is split across two chunks and neither chunk holds all of it. Set the overlap to 0 in the lab below and watch the boundary sentences break.

Every claim survived in order: the sizes, the reason for overlap, the chunk count, the question and its answer. The speaker, the student and the screen did not. "You lose this" was resolved from the frame into what is actually lost. The 7 must come from your own run of the code, not from the video.

## The rewrite pass

Run this on every chapter built from speech, after drafting and before the gate.

1. Search the chapter for the leak words and fix each hit:

   ```bash
   grep -nEi "\b(he|she|they) (says|said|asks|explains|shows|opens|types|copies|pastes|runs|scrolls|mentions|tells|wants|tries)\b|\b(instructor|lecturer|speaker|presenter|mentor|narrator|sir)\b|\b(the|this|today's|next|last) (video|session|lecture|class|recording|livestream)\b|question from the|the class \w+ed\b|as you can see|like this|like that|this thing|over here|this one|if (i )?remember|honestly|personally|in my (company|experience)|will be shared|description" docs/<your-folder>/<chapter>.md
   ```

2. Read only the first sentence of every section. Each should state something about the subject. If it names a person, a session or a screen, rewrite it.
3. Apply the textbook test to each paragraph: could it sit unchanged in a good textbook written by someone who never saw the video? If not, find the claim and rewrite around it.
4. Check that every number is yours, either printed by your run or read from a source you opened, and not quoted from the speaker.
5. Run `python3 .lecture-import/track-c/quality_gate.py <chapter>` and `coverage_check.py`. The gate checks the voice; the coverage check confirms that rewriting did not drop a claim.

## What strong chapters on this site do, so you can do it too

Read one of these before you write, and match its depth rather than its topic: `docs/genai/23-capstone.md`, `docs/mlops/data/02-pipelines-and-infrastructure/02-dataops-and-reliability.md`, `docs/agentic-ai/31-project-3-autonomous-data-analyst.md`.

- They open with a situation a reader recognises, not a definition.
- They do arithmetic by hand with small numbers, then reproduce the same numbers in code.
- They print results and read them aloud: what each number means and what a wrong value would look like.
- They keep a result that disappoints and explain why it happened.
- They name a failure with its cause, and say what was not tested.
- Each paragraph holds one idea in four sentences or fewer. Bullets are for steps and parameters only.
- They never mention who taught it, when, or what the screen looked like.

The difference between a thin chapter and a strong one is rarely the facts. It is whether each fact arrives with its reason, an example and a check.
