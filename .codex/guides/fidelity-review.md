# Checking a chapter against its source

Use this to prove a chapter carries everything the video, lecture or page teaches, in its order, as proper notes. Run it on your own chapter before you report, and on any chapter another agent hands back before you accept it. Codex output in particular gets this review every time.

A reviewer does not rewrite the chapter. A reviewer produces a list of findings, each with a location and a fix, and a verdict.

## The seven checks

### 1. Coverage: nothing skipped

```bash
python3 .lecture-import/track-c/coverage_check.py <pack>/blocks-en.txt <chapter>.md
```

`SKIPPED?` means no passage of the chapter matches that block. Open the block. If it holds a claim, it is a finding: name the claim and where it belongs. If it is filler (greeting, promotion, housekeeping), list it as dropped. The script's default thresholds (`--min 0.18`, `--window 3`) were tuned on synthetic data, so treat each line as a lead.

Then the manual walk, which the script cannot replace. Open the ledger and the chapter side by side and tick every claim. Report: blocks, claims, claims found, claims missing.

### 2. Order: nothing moved

`MOVED?` in the same output means a passage appears earlier in the chapter than the blocks before it. The source's order is the chapter's order. Moving a topic for tidiness is a finding. Opening sections written for the site (In one line, In 30 seconds, Words you will meet, the worked example) are expected before the source's content and are not moves.

### 3. Frames: no picture, code or table lost

Walk every contact sheet. For each distinct frame that is not a talking head, record where it went:

```
s004 f_00121  slide "Three types of memory"     -> board 03-memory-types.svg, section "Memory types"
s004 f_00126  notebook cell 9 (summary buffer)   -> Code you can run, block 4
s005 f_00140  terminal: pip install output       -> setup step, versions pinned
s005 f_00143  talking head                       -> nothing
```

A content frame with no destination is a finding. A board whose labels differ from the slide without a reason is a finding.

### 4. Code: same teaching, and it runs

- Every cell shown in the video is on the page, in order, with the video's names.
- Differences from the video are deliberate and explained (`:::note Correction` for fixes).
- `run_all.py` ran it, and the printed output was read, not just the exit code.
- No comments in code. No secrets.
- Versions are pinned or printed.

Diff the chapter's code against the repo notebook programmatically when there is one, and list every difference.

### 5. Numbers: every number has a home

List every number in the prose. Each must match a printed output from the chapter's code, or a source opened and dated, or arithmetic shown on the page. A number quoted from the speaker and not reproduced is a finding, unless it is clearly the speaker's example input (a chunk size of 1,000 is an input; "accuracy came out at 0.91" is an output and must be reproduced).

### 6. Voice: notes, not narration

```bash
python3 .lecture-import/track-c/quality_gate.py <chapter>.md
```

The gate fails narration of the speaker, the room and the session, and warns on pointing words. Then run the grep in `notes-voice.md` ("The rewrite pass") and read the first sentence of every section. Any sentence a textbook would not contain is a finding. Quote it and give the rewrite.

### 7. Additions are marked, and the craft is there

- Content not from the source sits under `:::note Added for this site`, unless it is a sentence of explanation woven into a source claim.
- The chapter has a worked example by hand whose numbers the code reproduces, a board for each picture, a lab, common mistakes, practice questions in block form, Check yourself, and Where to go next.
- Reading the output and Line by line follow every code block.

## Spot check by listening

Pick five blocks at random (`shuf -n 5` on the ledger). For each, read the block in the original language and then the chapter at that point. Could a reader of the chapter answer any question the block answers? This catches paraphrases that kept the words but lost the point, which no script detects.

## The review report

```markdown
## Review: <chapter path>  (<date>, against <pack path>)

Verdict: accept | accept after fixes | redo

| Check | Result |
| --- | --- |
| Coverage | 41 blocks, 63 claims, 60 found, 3 missing (B012, B027, B033) |
| Order | 1 moved: "Token costs" sits before "Message types" (B019 before B015) |
| Frames | 14 content frames, 13 placed, 1 missing (s006 f_00171 comparison table) |
| Code | 9 cells, all present; 2 corrections noted; run_all clean |
| Numbers | 12 numbers, 11 printed, 1 quoted only ("0.91 accuracy") |
| Voice | gate PASS; 2 pointing lines (L144, L201) |
| Additions and craft | lab present; worked example numbers match code |

### Findings, most important first

1. B027 missing: the claim that ... belongs after "...". Fix: add ...
2. L201 "as you can see the loss drops": name what drops and from what to what (frame f_00190 shows ...).
```

**Redo** when more than about a tenth of the claims are missing, the order is reorganised, or the voice is narration throughout. Fixing a narrated chapter line by line takes longer than rewriting it from the ledger.
