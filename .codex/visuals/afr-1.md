# Agent frontier, chapters 01 to 03: lab and board specs

Labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, native controls, no randomness at render time apart from seeded generators. Numbers that Python produced are embedded; the chapter code prints the same numbers. Working files: `.lecture-import/afr-1/` (data generator `ctx_data.py`, message capture `capture_flows.py`).

Boards: `scripts/infographics/afr_1.py` writes `static/img/afr/`. Chapter 01: `context-window-anatomy`, `context-window-worked-example`, `context-window-results`. Chapter 02: `mcp-and-a2a-big-picture`, `mcp-call-worked-example`, `a2a-task-lifecycle`. Chapter 03: `computer-use-observe-act-loop`, `computer-use-click-arithmetic`, `computer-use-injection-channels`. Every number on a board is printed by the chapter code.

## ContextWindowLab (chapter 01)

- Data: per-request token counts and cache-hit tokens for the 24-step synthetic agent run of chapter code block 3, for six strategies (keep everything, keep everything with a clock at the top, sliding window at windows 6,000, 10,000, 16,000, mask every step, mask in batches, summarise in batches; the last three for keep 2, 4, 6, 8), plus facts surviving (of 6). Tokeniser tiktoken 0.14.0 `o200k_base`.
- Controls: strategy select; recent steps kept select (2, 4, 6, 8; only for mask, batch, summary); window select (6,000, 10,000, 16,000); cache read price slider 0.05 to 0.50 (default 0.10); cache write price slider 1.00 to 2.00 (default 1.25).
- Drawn: 24 stacked bars, dark part cached, light part written; dashed window line; readout line (peak, billed, requests over window, cache hit share, cost units, facts kept).
- Defaults (batch, keep 4, window 10,000, 0.10, 1.25) reproduce: peak 7,900, billed 127,273, 0 over, 68.0% hits, cost 59,552, 4 of 6 facts.
- Other checks: keep everything peak 24,383, billed 289,768, 13 over, 57,017 units; clock 362,376 units, 0.1%; read price 0.50 makes keep everything cost 163,171, batch 94,174, mask every step 112,022.
- Table: step, tokens sent, from cache, written, vs window. Keyboard: native select and range inputs.

## ProtocolFlowLab (chapter 02)

- Data: four captured conversations (real messages and byte counts): MCP 2025-11-25 cold call (5 messages), MCP 2026-07-28 cold call (2), MCP multi round-trip refund call (4), A2A two-turn task (6, including the Agent Card fetch). The long `requestState` string is abbreviated in the display only. Servers: mcp 2.3.0, a2a-sdk 1.2.2.
- Controls: flow select; step range (reveals messages); calls-in-one-session range 1 to 100.
- Drawn: two-lane sequence diagram, each arrow labelled with message and bytes; the selected message shown as indented JSON below.
- Byte model from chapter code block 2: handshake style 447 setup plus 261 per call; stateless style 482 per call. Default (flow 1, step 5, 1 call) reproduces 708 against 482; at 2 calls 969 against 964; at 3 calls 1,230 against 1,446; at 100 calls 26,547 against 48,200.
- Table: step, direction, message, bytes. Keyboard: native controls.

## ActionSpaceLab (chapter 03)

- Data: toy form with five widgets (name 300x32, amount 300x32, category 200x32, submit 120x40, close 16x16) and the six error sizes of chapter code block 1; simulated task success (20,000 runs, seed 42 plus error) with and without a 30% chance of a 48 px layout shift per step; 150 seeded Gaussian samples drawn in the browser.
- Controls: grounding error select (0, 4, 8, 12, 16, 24 px); action space select (pixel coordinates or element references); layout select (still or banner shift); target widget select.
- Drawn: widgets, the aim point, 150 click dots (green lands, red misses), dashed outline of the old position when the page shifts.
- Defaults (12 px, pixels, still, submit) reproduce: analytic task success 0.494, simulated 0.497; P(hit submit) 0.904; with shift 0.120; element references 1.000.
- Table: widget, size, P(click lands). Keyboard: native selects.
