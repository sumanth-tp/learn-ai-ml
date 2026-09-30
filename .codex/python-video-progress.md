# Python video enrichment

## Request and source

- Preserve the existing Python content while improving it using the transcript
  and video frames from https://www.youtube.com/watch?v=ygXn5nV5qFc.
- Follow-up request: resolve build warnings.
- Source: Dave Ebbelaar, Python for AI - Full Beginner Course, 5:15:31.
- Read the English automatic transcript across the whole video and inspected
  84 sampled frames across its published chapters. Several chapter labels are
  out of sync with the actual topics; the guide uses transcript-checked starts.
- Default yt-dlp client failed; `youtube:player_client=android` worked for
  JSON3 subtitles, metadata and format 18 video. Working files are ignored under
  `.lecture-import/python-ai-ygXn5nV5qFc/`, including the frame manifest.

## Changes

- Six new lesson pages: video guide, setup/interactive Python, first program,
  weather API/report lab, modular sales analysis lab, Git/secrets/Ruff/uv workflow.
- Added introductory material or clarifications to ten existing Python pages.
  A comparison against HEAD verified that every original line remains in order.
- Runnable original examples and practice data are in
  `static/examples/python-beginner/`. Weather defaults to synthetic offline data;
  `--live` uses Open-Meteo. Sales exports CSV, JSON and XLSX.
- Corrected existing broken AI Security links to unpublished Memory and AgentOps
  chapters. They now link to the source video and explicitly mark notes pending.
  No unfinished module was invented to satisfy the link checker.

## Verification

- `npm run build`: passed with no warnings. Docusaurus prints a 3.10.2 update
  availability notice; dependencies were not changed for this content task.
- `npm run typecheck`: passed.
- Ruff check and format check: passed for all four new Python files.
- Python 3.12: offline and live weather runs passed, as did sales exports.
- Checked actual CSV/JSON/XLSX values, PNG dimensions, execution from another
  working directory, invalid sales inputs, weather date/value validation and the
  HTTP request contract. Parsed 39 new documentation Python snippets.
- Browser: nine Python pages, Mermaid expand controls, guide anchor targets,
  example downloads and mobile overflow passed without page errors.
- Verification scripts, screenshots, logs and generated reports are in the
  ignored working directory. No video frames or transcript were published.

No remaining work identified. Changes have not been committed or published.
