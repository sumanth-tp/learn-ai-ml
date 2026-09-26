# Task: turn three Hindi YouTube playlists into English course notes

You will produce study notes for three CampusX YouTube playlists (speaker:
Nitish, speaking Hinglish). Work in two steps for every video: **first convert
the Hindi transcript to English, then write the notes from that English
transcript.**

My requirements, which override anything else:

- **Don't hallucinate. Don't add extra things. Stick to the exact transcript.**
- **Reading the notes should feel like watching the video.**
- **The order must be the same as the video.**
- If YouTube rate-limits you, use another method (listed below) or a browser to
  get the transcript. Never write notes for a video whose transcript you could
  not get. Tell me instead.

## The playlists

| #   | Course                               | Playlist                                                                 | Videos |
| --- | ------------------------------------ | ------------------------------------------------------------------------ | ------ |
| 1   | Agentic AI using LangGraph           | https://www.youtube.com/playlist?list=PLKnIA16_RmvYsvB8qkUQuJmJNuiCUJFPL | 28     |
| 2   | Model Context Protocol (MCP Trilogy) | https://www.youtube.com/playlist?list=PLKnIA16_Rmva_oZ9F4ayUu9qcWgF7Fyc0 | 8      |
| 3   | LLM Evaluation                       | https://www.youtube.com/playlist?list=PLEneLIDJFpcA                      | 19     |

Each playlist starts with a short "New Playlist" announcement video. **Include
it** as page 1. The notes are 1:1 with the playlist: one page per video, in
playlist order, nothing skipped, nothing merged.

## Step 0: get the transcripts

1. List each playlist:

   ```bash
   yt-dlp --extractor-args "youtube:player_client=android" --flat-playlist \
     --print "%(playlist_index)s|%(id)s|%(title)s" "<playlist URL>"
   ```

2. Download the auto-captions, **one video at a time**:

   ```bash
   yt-dlp --extractor-args "youtube:player_client=android" --skip-download \
     --write-auto-subs --write-subs --sub-langs "hi,en" --sub-format json3 \
     -o "subs/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=<VIDEO_ID>"
   ```

   - The default and `tv` clients fail with "The page needs to be reloaded". Use
     `android`.
   - **Never fetch in parallel.** Sleep 6–8 s between videos. If a video fails,
     rotate the client (`android`, `ios`, `mweb`, `web_embedded`, `android_vr`)
     and retry in a later round.

3. If yt-dlp keeps returning HTTP 429, use the `youtube-transcript-api` Python
   package:

   ```python
   from youtube_transcript_api import YouTubeTranscriptApi
   api = YouTubeTranscriptApi()
   tr = api.list(video_id).find_transcript(["hi", "en"])
   segments = tr.fetch()   # each has .start, .duration, .text
   ```

   Do not fetch the caption `baseUrl` directly. It returns HTTP 200 with an
   empty body because it needs a player token. If both methods fail, open the
   video in a browser and copy the transcript from "Show transcript".

4. Merge the caption fragments into **~40-second blocks**, each prefixed with
   its start time, `[HH:MM:SS]`. Raw fragments are unreadable.

## Step 1: English transcript (for every video, before any notes)

The captions are Hinglish written in Devanagari. English technical words appear
transliterated: "लैंग ग्राफ" = LangGraph, "स्टेट" = state, "नोड" = node, "इनवोक"
= invoke. Produce one English transcript file per video:

- Keep **every** block, in order, with its timestamp.
- Translate faithfully. Keep the speaker's meaning, examples, analogies, asides,
  corrections and warnings. Remove only pure filler ("ठीक है?", "right?").
- Spell code identifiers, library names and values correctly (`StateGraph`,
  `add_edge`, `TypedDict`, `gpt-4o-mini`).
- Fix obvious speech-recognition errors from context, such as "Lindin" →
  LinkedIn. If you are not sure, write `[unclear]` or `word [?]` rather than
  guessing.
- Do not summarise at this step.

## Step 2: notes page, written from the English transcript

One Markdown file per video, named `NN-<short-slug>.md`, where NN is the
two-digit playlist position. Folders: `agentic-ai/`, `mcp/`, `llm-evals/`.

### Page header

```yaml
---
id: <folder>-<slug>
title: "<the video's YouTube title, verbatim; escape inner double quotes>"
sidebar_label: "<N> · <short topic>"
sidebar_position: <N>
slug: /<folder>/<slug>
description: "<one sentence on what this video covers, in its own terms>"
tags: [<3–6 lowercase tags>]
---

> **Video <N> of <TOTAL>** · [Watch on YouTube](<url>) · Translated from the
> Hindi transcript. Notes follow the video section by section, in its order.
```

N is the playlist position; TOTAL is 28, 8 or 19. Keep the title verbatim even
when it says a different "Video N". Then add **one sentence** stating the
video's point, built only from the speaker's own framing.

### What the notes must be

- **Same order as the video.** Headings follow the video's own segments as they
  happen: the recap, the "why", the "what", each example, each demo, and "what
  comes next". Never reorder, merge, or impose your own topic structure.
- **Nothing added.** No facts, tips, best practices, extra examples, further
  reading, checklists, interview questions, projects or explanations the speaker
  does not give. There is one exception. If he states something factually wrong
  (a wrong API name, date or library behaviour), keep what he said and add a
  short note giving the correction. Leave simplifications and opinions alone.
- **Nothing skipped.** Every topic, example, analogy, number, diagram
  description, code step, bug and its fix, warning, aside and homework must
  appear, at the point where he gives it. Only "like and subscribe" and sponsor
  plugs may be dropped.
- **Feels like watching.** Write flowing teaching prose that walks through the
  video as it unfolds. Keep his running examples exactly, with the same names,
  numbers and scenarios. Use an impersonal voice. **Never** write "He says…",
  "Nitish explains…" or "In this video he…".
- **Your own words, not a transcript dump.** Explain each point clearly and
  completely, but do not paste the translation. Short quotes of a definition are
  fine.
- **Slides.** When he reads out slide content, include that content as prose or
  a list.
- **Code:**
  - Reproduce the code he writes or reads out, with his variable names, in the
    order he builds it.
  - Show the mistakes he makes, and the fix at the point where he fixes it.
  - If he builds a file step by step, show each step, then the complete file
    once at the end of that demo.
  - Use real imports for the libraries he names.
  - If a line is necessary but never spoken, add it with the trailing comment
    `# (implied, not shown in narration)`.
  - Never add functionality.
  - Include only outputs he reads out, as short `text` blocks. Never invent
    outputs.
- **Diagrams.** Only when he draws or walks through a flow (a graph of nodes, an
  architecture, a pipeline, a loop), recreate it as a Mermaid diagram at that
  point. No decorative diagrams. Never embed screenshots or video frames.
- **Tables.** Use them only for comparisons he makes (X vs Y, or a checklist he
  ticks off repeatedly). Use lists for everything else.
- **No links** to other pages or sites, except the video link in the header.
- End with "What comes next" only if he says what the next video covers.
  Otherwise end with his closing point.
- Use British spelling in prose. Keep code and product names as they are.

### If the notes go into a Docusaurus/MDX site

- `{...}` in prose is parsed as JSX and breaks the build. Put prompt
  placeholders like `{topic}` in backticks or code blocks, or escape them as
  `\{topic\}`.
- `<` followed by a letter in prose is parsed as a tag. Use backticks.
- Callouts use `:::note` / `:::warning`, on their own lines with blank lines
  around them.

## Coverage self-check (every page)

After writing a page, walk its English transcript block by block. Confirm that
every block's substance appears in the notes, in the same order. Fix any gaps
before moving on.

## Working method

Process one video at a time: fetch → English transcript → notes → coverage check
→ next video. Long videos (1–2 hours) should be translated in chunks of about 30
blocks, appended to the transcript file. If you are interrupted, resume. Never
re-translate a finished transcript, never overwrite a finished page, and
continue a partial transcript from its first missing block.

## Deliverables

1. `english/<course>/<NN>-<videoId>.md`: the full English transcript for each of
   the 55 videos.
2. `<folder>/NN-<slug>.md`: one notes page per video.
3. A final report listing, per video: the page path, the number of transcript
   blocks, the headings in order, any `[unclear]` passages, any correction
   notes, and any code lines marked implied. Also list any video you could not
   fetch or cover, and why. **Do not claim a video is covered if you did not
   process its full transcript.**
