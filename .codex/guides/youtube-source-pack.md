# Building a source pack from a YouTube video

Before writing a word, collect everything the video offers into one private folder: metadata, the description, the transcript, numbered blocks, a coverage ledger, frames, contact sheets, and the repos and notebooks the description links to. Writing from a complete pack is what makes the notes match the video. Writing from the transcript alone is how code, slides and diagrams go missing.

Everything in the pack is private working material. It lives under `.lecture-import/<job>/`, which is git-ignored. Nothing from it is published as is: no frames, no transcript text, no description text.

## The tool

`scripts/source-import/yt_pack.py`, run with `/usr/bin/python3` (that interpreter has yt-dlp, youtube-transcript-api, Pillow, numpy, bs4 and requests on this machine). Tested on 2026-10-09 with yt-dlp 2025.10.14.

```bash
P=.lecture-import/<job>/<NN>-<video-id>
/usr/bin/python3 scripts/source-import/yt_pack.py all "https://www.youtube.com/watch?v=<id>" $P --frames
```

`all` runs these steps in order. Each also runs on its own:

| Step | Command | Writes |
| --- | --- | --- |
| Metadata and description | `meta URL OUT` | `meta.json`, `description.md` (title, channel, upload date, duration, the description's chapter list, every link classified as repo, notebook, drive, slides, paper, model-or-data, social or course-promo, and the description verbatim) |
| Playlist | `playlist URL OUT` | `playlist.tsv`: index, id, duration, title. One chapter per line, in this order. |
| Transcript | `transcript URL OUT [--langs en,hi] [--translate]` | `subs/`, `transcript-<lang>.txt` (`[hh:mm:ss] text` per caption) |
| Blocks and ledger | `blocks OUT [--seconds 40]` | `blocks-<lang>.txt` (the input for `coverage_check.py`), `ledger-<lang>.md` (one row per block) |
| Video | `video URL OUT` | `video.mp4`, 360p |
| Frames and sheets | `frames OUT [--every 10] [--diff 6]` | `frames/f_NNNNN.jpg`, `frames/index.tsv` (time, kept or duplicate, change score), `sheets/sNNN.jpg` (3x3 contact sheets of distinct frames, each stamped with its time) |
| Exact moments | `grab OUT 1:16:05 4570` | `grabs/g_011605.png` at full resolution |
| Notebook | `notebook path.ipynb [--outputs]` | prints every cell in order, with outputs if asked |

Global options go before the step name: `--pause 7` (seconds between YouTube requests) and `--rounds 2` (passes over the player clients).

## What happened on the test run, so you know what to expect

On a 17-minute video: metadata came through the `android` client first time. Captions from yt-dlp failed with HTTP 429 on every client for two rounds, and the tool then fell back to youtube-transcript-api, which returned 403 English segments (24 blocks of 40 s). The 360p video was 19.5 MB. Frames every 10 s gave 102 frames, of which 10 were distinct. That is typical for a talking-head video with a few slides.

## Rules that keep YouTube working

- **Never fetch in parallel.** One video at a time, one request at a time. The tool sleeps 7 s between attempts and rotates `android`, `ios`, `mweb`, `web_embedded`, `android_vr`. The default and `tv` clients fail on this machine.
- **HTTP 429 means stop, not retry harder.** The tool falls back to youtube-transcript-api automatically. If both fail, wait 15 to 30 minutes, then rerun only the missing step. For a playlist, loop over `playlist.tsv` in a shell loop with a `sleep 8` between videos.
- **360p is the reliable video format** (format 18). The 720p and 1080p URLs often return 403. At 360p, notebook code at full frame size is readable. Small terminal fonts sometimes are not; see "When a frame is unreadable" below.
- **Fetching a caption `baseUrl` yourself returns an empty body.** Do not try. The library handles the player token.

## Languages

`--langs en,hi` asks for both. A Hindi video usually gives `hi` (the original auto captions) and `en` (YouTube's auto-translation). Keep both:

- Write from the English file, but check technical terms, numbers and code names against the original. Auto-translation mangles names: LangGraph becomes "Langraph", Groq becomes "Gro", Qwen becomes "Quen", Pydantic becomes "Pantic". The frames settle the spelling.
- If only the original track exists and it is not translatable, translate block by block into `blocks-en.txt` yourself, keeping the `[hh:mm:ss]` prefix on each block. Translate every block in full. Do not summarise while translating, because a summary loses statements that the coverage check will then not find.
- Hinglish filler ("matlab", "basically", "theek hai", "samjhe?") is filler. Technical words said in English stay as said.

If a video has no captions at all, write `BLOCKER: no captions for <id>` in the progress file and ask. Local speech-to-text (for example faster-whisper in `venv-llm`) is possible, but it has not been set up or tested on this machine.

## Reading the description

`description.md` is the most underused file in the pack. Read it first.

- **Chapters.** The description's chapter list is the video's own outline. Use it as the skeleton of the chapter's headings, renamed to name the idea ("Why Ollama exists", not "15:17 Why Ollama Exists"). Never copy the times into the page.
- **Repos and notebooks.** Clone each linked repo into the pack, not into the site:

  ```bash
  git clone --depth 1 <repo-url> $P/repos/<name>
  /usr/bin/python3 scripts/source-import/yt_pack.py notebook $P/repos/<name>/<notebook>.ipynb --outputs
  ```

  The notebook is the closest thing to the exact code. The video is still the ground truth for what is taught, and the order it is taught in. Record the commit hash you cloned in the report.
- **Drive, Colab and Kaggle links** often need a login or a click-through. Try once. If you cannot get the file, read the code from frames and put a `:::warning` on the page saying the code was reconstructed from the video and where it might differ.
- **Slides and PDFs.** Save them with `web_extract.py` (see `web-sources.md`). Slide text is exact where frames are blurry.
- **Papers and model cards.** Open them and use them as sources in **Go deeper**, with the date you opened them.
- **Social and course-promo links.** Ignore them. They never appear on the page.

The page never links to GitHub (house rule). A repo informs the chapter. It is not linked from it, except the source line, which names only the video.

## Frames: what to extract and how

Scan every contact sheet in order. For each distinct frame, decide what it holds, then open the full frame (`frames/f_NNNNN.jpg`) or grab a sharper moment (`grab`) for anything you will reproduce.

| On the frame | Becomes |
| --- | --- |
| Code in an editor or notebook cell | Code on the page. Read the final state of the cell; when the code is typed live, check the last frame before it runs. |
| Terminal or cell output | A printed result, but only once you have reproduced it with your own run. |
| A slide, whiteboard, Excalidraw page or drawn diagram | An original SVG board (`boards.md`). A slide that builds up over several frames: use the last, complete frame. On the test video a "Curriculum" slide went from empty to three boxes to five over two minutes. |
| A table, or a comparison stated on a slide | A Markdown table, and a board too if the table is the centrepiece. |
| A file tree, `pip install` line, version print or `.env` key names | Setup steps and pinned versions on the page. Never copy a key value. |
| A web page or docs page being read | A source to open yourself and cite, dated. |
| Talking head only | Nothing. The transcript carries it. |

Default `--every 10` suits most videos. Use `--every 5` for fast live coding. Raise `--diff` (for example to 10) when a moving face makes too many frames count as distinct; lower it (to 3) when small code edits are being missed. Check `frames/index.tsv`: its `diff` column shows the change score for every frame.

### When a frame is unreadable

At 360p, small fonts blur. In order: grab the exact moment at full resolution; look at the frames a few seconds either side; compare with the notebook from the repo; read the transcript, where the speaker often reads code aloud. If it is still uncertain, write the most likely code, run it to prove it works, and put a `:::note` on the page saying it was reconstructed. Never invent output to match a blurry screen.

### Secrets on screen

API keys, tokens and passwords sometimes appear in frames. Never copy them into notes, reports or code. Use `os.environ["OPENAI_API_KEY"]` or `...`.

## The pack, finished

```
.lecture-import/<job>/<NN>-<id>/
  meta.json  description.md
  transcript-en.txt  transcript-hi.txt
  blocks-en.txt  ledger-en.md
  video.mp4  frames/  sheets/  grabs/
  repos/<name>/  (cloned, commit recorded)
  extra/  (slides, docs pages saved with web_extract.py)
```

The pack is ready when you can answer three questions from it: what the outline is (description chapters plus blocks), what code is taught (frames plus notebooks), and what pictures are shown (sheets). Then go to `youtube-chapter.md`.

A 2-hour video's pack with frames is about 300 MB. Delete `video.mp4` and `frames/` when the chapter is accepted. Keep `blocks-*.txt`, `ledger-*.md` and `description.md`, which are small and let the next session re-check the chapter.
