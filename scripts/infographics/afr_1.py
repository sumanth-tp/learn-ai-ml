"""Infographics for docs/agentic-frontier, chapters 01 to 03.

Run from the repo root:

    python3 scripts/infographics/afr_1.py            # all boards
    python3 scripts/infographics/afr_1.py worked     # boards whose name contains the word
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "afr"
BOARDS = {}
NAMES = {}


def board(name):
    def deco(fn):
        BOARDS[fn.__name__] = fn
        NAMES[fn.__name__] = name
        return fn
    return deco


def raw_text(b, x, y, text, size=12, fill=INK, anchor="middle", weight="400"):
    b.parts.append(
        f'<text xml:space="preserve" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MONO}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(text)}</text>'
    )


def rect(b, x, y, w, h, fill, stroke, width=1.6, rx=4):
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def hbar(b, x, y, w, h, frac, color, label, value, label_w=0):
    rect(b, x, y, w, h, "#f1f3f5", "#ced4da", 1, 3)
    if frac > 0:
        rect(b, x, y, max(2, w * frac), h, PALETTE[color]["stroke"], PALETTE[color]["stroke"], 1, 3)
    raw_text(b, x - 10, y + h / 2 + 4, label, 12, INK, "end")
    raw_text(b, x + w + 10, y + h / 2 + 4, value, 12, PALETTE[color]["text"], "start", "700")


@board("context-window-anatomy")
def context_anatomy():
    b = Board(1240, 720, "A context window is a budget", "Tokens counted with the o200k_base tokenizer, chapter code blocks 2 to 4")
    b.group(20, 90, 1200, 230, "What one request carries: 20 tools and an almost empty conversation", "blue")
    parts = [("system", 71, "blue"), ("tool definitions", 1060, "red"), ("memory", 20, "teal"), ("history", 51, "purple"), ("message", 6, "green")]
    total = sum(p[1] for p in parts)
    x = 60.0
    width = 1120.0
    for name, tokens, color in parts:
        w = width * tokens / total
        rect(b, x, 150, w, 56, PALETTE[color]["fill"], PALETTE[color]["stroke"], 2, 4)
        if w > 120:
            raw_text(b, x + w / 2, 175, f"{name}", 14, PALETTE[color]["text"], "middle", "700")
            raw_text(b, x + w / 2, 195, f"{tokens:,} tokens, {tokens / total:.1%}", 12, INK)
        x += w
    sys_w = width * 71 / total
    b.arrow((60 + sys_w / 2, 208), (60 + sys_w / 2, 236), color="blue", width=1.2)
    raw_text(b, 60, 258, "system prompt  71 tokens", 12, PALETTE["blue"]["text"], "start", "700")
    for px, (name, tokens, color) in zip((560, 760, 960), parts[2:]):
        pill = b.pill(px, 232, f"{name} {tokens} tokens, {tokens / total:.1%}", color, size=12, solid=True)
    raw_text(b, 1180, 222, "the three thin slices at the right end of the bar", 11, FAINT, "end")
    raw_text(b, 620, 302, "4 tools: 362 tokens in all.  10 tools: 680.  20 tools: 1,208.  About 53 tokens per tool, paid on every request.", 13, INK, "middle", "700")

    b.group(20, 340, 1200, 360, "Six levers, each with the number the chapter measured", "green")
    cards = [
        ("1  Count", ["measure tokens per section", "before you send, with the", "tokenizer your model uses"], "blue"),
        ("2  Trim", ["keep the head and tail of a", "long tool output: one result", "went from 1,233 to 508 tokens"], "teal"),
        ("3  Mask", ["swap old tool results for a", "one-line stub: peak context", "24,383 down to 5,804 tokens"], "orange"),
        ("4  Compact", ["summarise old steps in", "batches: 123,500 tokens", "billed, 4 of 6 facts kept"], "purple"),
        ("5  Remember", ["write findings into the", "message stream or a notes file:", "sliding window kept 1 of 6"], "pink"),
        ("6  Layout", ["stable text first: 91.6%", "cache hits, against 0.1%", "with a clock at the top"], "yellow"),
    ]
    for i, (title, lines, color) in enumerate(cards):
        cx = 40 + (i % 3) * 395
        cy = 385 + (i // 3) * 150
        b.card(cx, cy, 370, 125, title, lines, color, size=13)
    return b


@board("context-window-worked-example")
def context_worked():
    b = Board(1240, 740, "Eight requests by hand", "Fixed part 200 tokens. A full step adds 200, a masked step only 30. Window 1,000. Mask keeps the last 2 results.")
    b.group(20, 95, 580, 620, "Tokens sent in request n", "blue")
    rows = [["n", "keep all", "mask old", "slide"]]
    full = [400, 600, 800, 1000, 1200, 1400, 1600, 1800]
    mask = [400, 600, 630, 660, 690, 720, 750, 780]
    slide = [400, 600, 800, 1000, 1000, 1000, 1000, 1000]
    for i in range(8):
        rows.append([str(i + 1), f"{full[i]:,}", f"{mask[i]:,}", f"{slide[i]:,}"])
    rows.append(["total", "8,800", "5,230", "6,800"])
    rows.append(["peak", "1,800", "780", "1,000"])
    b.table(45, 145, [90, 150, 150, 150], rows, "blue", size=14, row_h=40)
    raw_text(b, 310, 600, "keep all passes the 1,000 window at request 5 (1,200)", 13, PALETTE["red"]["text"], "middle", "700")
    raw_text(b, 310, 630, "mask: 200 + 6 x 30 + 2 x 200 = 780 at request 8", 13, INK)
    raw_text(b, 310, 656, "slide: drops the oldest whole step when over 1,000", 13, INK)
    raw_text(b, 310, 690, "the printed code reproduces every number", 12, FAINT)

    b.group(620, 95, 600, 620, "Now switch the prefix cache on (read 0.10, write 1.25)", "orange")
    b.card(640, 145, 560, 105, "keep all, request 2", ["400 tokens already cached, 200 new", "400 x 0.10 + 200 x 1.25 = 40 + 250 = 290 units", "request 1 costs 400 x 1.25 = 500"], "orange", size=13)
    b.card(640, 270, 560, 125, "mask old, request 3", ["step 1 is rewritten as a stub, so only the", "fixed 200 tokens still match the cache", "200 x 0.10 + 430 x 1.25 = 20 + 537.5 = 557.5 units"], "red", size=13)
    rows2 = [["strategy", "cache cost", "fits window"], ["keep all", "2,950", "no, from request 5"], ["mask old", "4,180", "yes"], ["slide", "5,510", "yes"]]
    b.table(640, 420, [190, 150, 220], rows2, "orange", size=14, row_h=40)
    b.card(640, 600, 560, 95, "the lesson", ["fewer tokens is not the same as a lower bill:", "an edit near the front of the prompt throws the cache away"], "yellow", size=13)
    return b


@board("context-window-results")
def context_results():
    b = Board(1240, 640, "24 agent steps, six ways to manage the window", "Chapter code block 3. Window 10,000, trigger 8,000, keep 4, cache read 0.10 and write 1.25 (placeholder ratios)")
    rows = [
        ["strategy", "peak tokens", "billed", "over window", "facts kept", "cache hits", "cost units"],
        ["keep all", "24,383", "289,768", "13 requests", "6 of 6", "91.6%", "57,017"],
        ["keep all + clock", "24,400", "290,176", "13 requests", "6 of 6", "0.1%", "362,376"],
        ["sliding window", "9,992", "184,801", "0", "1 of 6", "32.3%", "162,305"],
        ["mask every step", "5,804", "96,328", "0", "4 of 6", "11.6%", "107,548"],
        ["mask in batches", "7,900", "127,273", "0", "4 of 6", "68.0%", "59,552"],
        ["summary in batches", "7,769", "123,500", "0", "4 of 6", "67.3%", "58,721"],
    ]
    b.table(30, 100, [230, 160, 150, 170, 150, 150, 160], rows, "blue", size=14, row_h=44)
    raw_text(b, 620, 440, "Cost units, lower is better (keep all overflows the window, so it is shown for reference)", 13, INK, "middle", "700")
    values = [("keep all", 57017, "grey"), ("keep all + clock", 362376, "red"), ("sliding window", 162305, "orange"),
              ("mask every step", 107548, "orange"), ("mask in batches", 59552, "green"), ("summary in batches", 58721, "green")]
    for i, (label, value, color) in enumerate(values):
        hbar(b, 260, 462 + i * 26, 640, 18, value / 362376, color, label, f"{value:,}")
    return b


@board("mcp-and-a2a-big-picture")
def interop_big_picture():
    b = Board(1240, 740, "Two protocols, two directions", "MCP revision 2026-07-28 and A2A 1.0.0, read from their specifications on 7 October 2026")
    b.group(20, 90, 590, 330, "MCP: an agent reaches tools and data", "blue")
    h = b.card(40, 140, 160, 110, "host app", ["the LLM and its", "own agent loop"], "grey", size=12)
    c = b.card(235, 140, 130, 110, "MCP client", ["one per", "server"], "blue", size=12)
    s = b.card(425, 140, 165, 110, "MCP server", ["tools, resources,", "prompts"], "blue", size=12)
    b.arrow(h.right(), c.left())
    b.arrow(c.right(), s.left(), label="JSON-RPC\ntools/call", label_dy=-6)
    b.card(40, 285, 550, 110, "what a call carries", ["every request names its protocol version and", "capabilities in _meta: no handshake, no session", "the server answers server/discover on demand"], "blue", size=12)

    b.group(630, 90, 590, 330, "A2A: an agent hands work to another agent", "purple")
    a = b.card(650, 140, 160, 110, "client agent", ["has a goal it", "cannot meet alone"], "grey", size=12)
    d = b.card(1045, 140, 155, 110, "remote agent", ["opaque: you see", "its card, not", "its internals"], "purple", size=12)
    b.arrow(a.right(), d.left(), label="SendMessage", label_dy=-8)
    b.card(650, 285, 550, 110, "what a task carries", ["a Task with an id and a state, a context id", "for the conversation, artifacts for results,", "messages when it needs more input"], "purple", size=12)

    rows = [
        ["", "MCP 2026-07-28", "A2A 1.0.0"],
        ["unit of work", "a call: tools/call, resources/read", "a Task that can run, pause and resume"],
        ["discovery", "server/discover (and tools/list)", "Agent Card at /.well-known/agent-card.json"],
        ["state", "none in the protocol; use handles", "task id and context id"],
        ["transports", "stdio, Streamable HTTP", "JSON-RPC, gRPC, HTTP+JSON"],
        ["authorisation", "OAuth 2.1 style, resource indicators", "schemes declared in the Agent Card"],
    ]
    b.table(30, 445, [200, 490, 490], rows, "teal", size=13, row_h=40)
    return b


@board("mcp-call-worked-example")
def mcp_worked():
    b = Board(1240, 760, "One tool call, four outcomes", "Chapter code block 3 against a real MCP server (Python SDK 2.3.0). Tool: refund(order_id), which asks the user to confirm.")
    xs = [40, 345, 650, 955]
    steps = [
        ("1  no elicitation", ["client declares no", "capabilities"], ["error -32021", "MissingRequired-", "ClientCapability", "needs elicitation.form"], "red"),
        ("2  with elicitation", ["client declares", "elicitation.form"], ["resultType:", "input_required", "inputRequests: confirm", "requestState: an opaque", "string over 300 chars"], "orange"),
        ("3  retry with answer", ["new request id, same", "arguments, plus", "inputResponses and", "the state, unchanged"], ["resultType: complete", "text: refunded A-17"], "green"),
        ("4  tampered state", ["same retry, but the", "last four characters", "of requestState changed"], ["error -32602", "Invalid or expired", "requestState"], "red"),
    ]
    for x, (title, sent, got, color) in zip(xs, steps):
        b.group(x - 10, 95, 285, 400, title, color)
        b.card(x, 145, 265, 120, "client sends", sent, "blue", size=12)
        b.card(x, 300, 265, 150, "server answers", got, color, size=12)
        b.arrow((x + 132, 267), (x + 132, 298), color=color)
    b.group(20, 520, 1200, 220, "The cost of being stateless, measured on the same server", "teal")
    rows = [["", "messages", "bytes, 1 cold call", "bytes, 20 calls", "bytes, 100 calls"],
            ["MCP 2025-11-25 (handshake)", "3", "708", "5,667", "26,547"],
            ["MCP 2026-07-28 (per-request _meta)", "1", "482", "9,640", "48,200"]]
    b.table(40, 565, [380, 140, 220, 220, 220], rows, "teal", size=13, row_h=40)
    raw_text(b, 620, 705, "break-even near 2 calls: the handshake is cheaper per call, but needs a session to be kept alive and routed", 12, FAINT)
    return b


@board("a2a-task-lifecycle")
def a2a_lifecycle():
    b = Board(1240, 800, "An A2A task is a small state machine", "States from the A2A 1.0.0 specification; the two-turn example is chapter code block 5")
    sub = b.card(40, 180, 170, 70, "SUBMITTED", ["acknowledged"], "grey", size=12)
    wrk = b.card(290, 180, 170, 70, "WORKING", ["agent is busy"], "blue", size=12)
    inp = b.card(600, 130, 220, 70, "INPUT_REQUIRED", ["interrupted: asks the user"], "orange", size=12)
    aut = b.card(600, 230, 220, 70, "AUTH_REQUIRED", ["interrupted: wants auth"], "orange", size=12)
    b.arrow(sub.right(), wrk.left())
    b.arrow(wrk.right(0.3), inp.left(0.5), label="needs input", label_dy=-14, label_dx=-6)
    b.arrow(wrk.right(0.7), aut.left(0.5), label="needs credentials", label_dy=14, label_dx=-6)
    b.arrow(inp.top(0.5), wrk.top(0.5), via=[(710, 105), (375, 105)], dashed=True, label="client replies: SendMessage with the taskId", color="orange")
    b.group(290, 340, 930, 130, "Terminal states: the task is over and cannot restart", "red")
    for x, name, note, color in ((310, "COMPLETED", ["finished,", "artifacts ready"], "green"), (550, "FAILED", ["ended with", "an error"], "red"),
                                 (790, "CANCELED", ["stopped before", "finishing"], "red"), (1030, "REJECTED", ["the agent", "declined it"], "red")):
        b.card(x, 385, 170, 70, name, note, color, size=11)
    b.arrow(wrk.bottom(0.5), (375, 338), label="ends in one of four", label_dx=82)

    b.group(20, 500, 1200, 280, "The two-turn example, as the wire shows it", "purple")
    b.card(40, 545, 560, 110, "turn 1: SendMessage", ["text: please approve my expense", "reply: a Task, new id and contextId", "state INPUT_REQUIRED: What is the amount?"], "orange", size=13)
    b.card(640, 545, 560, 110, "turn 2: SendMessage with taskId", ["text: amount 120, same taskId and contextId", "reply: the same Task, history of 3 messages", "state COMPLETED, artifact: expense of 120: approved"], "green", size=13)
    b.arrow((320, 657), (320, 685), color="purple")
    b.arrow((920, 657), (920, 685), color="purple")
    b.card(40, 685, 1160, 78, "what makes it a task and not a chat", ["the server stores the Task between calls, so the client sends only the new message and the ids", "GetTask returns the final state later; a streaming client would receive status and artifact updates instead"], "purple", size=12)
    return b


@board("computer-use-observe-act-loop")
def cu_loop():
    b = Board(1240, 720, "A computer-use agent is a loop with a gate", "Token counts measured in chapter code block 2: a 40-product shop page, 1280 x 800 viewport")
    env = b.card(40, 120, 230, 120, "screen or page", ["a desktop, a browser tab", "or a phone"], "grey", size=13)
    obs = b.card(330, 100, 300, 160, "observe", ["screenshot: 1,334 visual tokens", "ARIA snapshot: 2,492 tokens", "visible text: 1,015 tokens", "raw HTML: 2,923 tokens"], "blue", size=13)
    mod = b.card(690, 120, 200, 120, "model", ["reads the observation,", "proposes one action"], "purple", size=13)
    gate = b.card(950, 100, 250, 160, "gate (plain code)", ["origin allowlist", "ask a person for risky verbs", "step and time budget"], "red", size=13)
    b.arrow(env.right(), obs.left())
    b.arrow(obs.right(), mod.left())
    b.arrow(mod.right(), gate.left())
    b.arrow(gate.bottom(0.5), env.bottom(0.5), via=[(1075, 320), (155, 320)], label="act, then observe again", label_dy=-4)

    b.group(20, 360, 590, 330, "Observation channels", "blue")
    rows = [["channel", "tokens", "what it shows"],
            ["pixels", "1,334", "only the first screen: 16 of 40 buttons"],
            ["ARIA tree", "2,492", "the whole page, with roles and names"],
            ["visible text", "1,015", "words only, no controls"],
            ["raw HTML", "2,923", "everything, including hidden text"]]
    b.table(40, 410, [130, 100, 320], rows, "blue", size=13, row_h=44)
    b.group(630, 360, 590, 330, "Action spaces", "orange")
    rows = [["action", "needs", "breaks when"],
            ["click(x, y)", "pixel grounding", "the layout moves"],
            ["click(role, name)", "an ARIA tree", "names repeat or are missing"],
            ["key and type", "a focused field", "focus is somewhere else"],
            ["navigate(url)", "a URL policy", "the model is steered off-site"]]
    b.table(650, 410, [170, 170, 200], rows, "orange", size=13, row_h=44)
    return b


@board("computer-use-click-arithmetic")
def cu_arithmetic():
    b = Board(1240, 720, "Why small targets and long tasks fail", "Grounding error 12 px (one standard deviation), the chapter's default")
    b.card(40, 100, 560, 130, "one click on the submit button, 120 x 40 px", ["half-sizes 60 and 20, error 12", "erf(60 / (12 x 1.414)) = erf(3.54) = 1.000 across", "erf(20 / (12 x 1.414)) = erf(1.18) = 0.904 down"], "blue", size=13)
    b.card(640, 100, 560, 130, "so P(hit submit) = 1.000 x 0.904", ["= 0.904", "the 300 x 32 fields: erf(16 / 16.97) = erf(0.943) = 0.818", "the 16 x 16 close button: 0.245"], "blue", size=13)
    b.arrow((320, 232), (320, 270), color="blue")
    b.arrow((920, 232), (920, 270), color="blue")
    b.card(40, 270, 1160, 120, "four clicks in a row: name, amount, category, submit", ["0.818 x 0.818 x 0.818 x 0.904 = 0.494", "20,000 simulated runs gave 0.497", "if a banner may push the page down before each click, even perfect aim gives", "0.7 x 0.7 x 0.7 x 0.7 = 0.240 (the runs gave 0.239); with error 12 as well, 0.120"], "green", size=13)
    b.group(20, 415, 1200, 290, "Measured over six error sizes (error in px, task success)", "orange")
    rows = [["error (px)", "0", "4", "8", "12", "16", "24"],
            ["P(hit submit)", "1.000", "1.000", "0.988", "0.904", "0.789", "0.588"],
            ["pixel clicks, still page", "1.000", "1.000", "0.857", "0.497", "0.252", "0.072"],
            ["pixel clicks, 30% shift", "0.239", "0.239", "0.207", "0.120", "0.063", "0.017"],
            ["element references", "1.000", "1.000", "1.000", "1.000", "1.000", "1.000"]]
    b.table(40, 465, [300, 140, 140, 140, 140, 140, 140], rows, "orange", size=13, row_h=42)
    raw_text(b, 620, 692, "element references remove pixel error by construction; their own failures are measured in the real-browser block", 12, FAINT)
    return b


@board("computer-use-injection-channels")
def cu_injection():
    b = Board(1240, 740, "Where hidden text reaches the model, and what stops it", "Real headless Chromium, Playwright 1.63.0: the same sentence placed nine ways (chapter code blocks 3 and 4)")
    rows = [["placement", "HTML", "text", "ARIA", "pixels"],
            ["no injection (control)", "-", "-", "-", "-"],
            ["display:none block", "yes", "-", "-", "-"],
            ["white text on white", "yes", "yes", "yes", "-"],
            ["off-screen position", "yes", "yes", "yes", "-"],
            ["1 pixel font", "yes", "yes", "yes", "renders"],
            ["visible paragraph", "yes", "yes", "yes", "yes"],
            ["button aria-label", "yes", "-", "yes", "-"],
            ["image alt text", "yes", "-", "yes", "-"],
            ["HTML comment", "yes", "-", "-", "-"]]
    b.table(30, 100, [260, 90, 90, 90, 110], rows, "red", size=13, row_h=40)
    b.group(700, 95, 520, 185, "Reading the table", "red")
    b.text(960, 150, "An ARIA-tree agent reads text that a human\nand a screenshot-only agent never see:\nwhite-on-white, off-screen, aria-label, alt.\nA screenshot-only agent is not safe either:\na visible paragraph is plain to read.", 13, INK)
    b.group(700, 300, 520, 420, "The gate sees only the action", "green")
    rows2 = [["proposal", "verdict"],
             ["navigate to evil.test", "block"],
             ["navigate to shop.example.evil.test", "block"],
             ["click Send email to everyone", "ask a person"],
             ["type a password into Notes", "ask a person"],
             ["click Pay now (a real task step)", "ask a person"],
             ["click Add trail shoe 7 (injected)", "allow"]]
    b.table(720, 350, [350, 130], rows2, "green", size=13, row_h=42)
    raw_text(b, 960, 695, "the last line is the residue: allowed actions can still be wrong", 12, FAINT)
    b.card(30, 540, 640, 150, "so the defence is layered", ["1  treat every observation as untrusted data", "2  narrow the action space and the origins", "3  put risky verbs behind a person", "4  verify the outcome, do not trust the agent's report"], "yellow", size=13, align="left")
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        keys = [k for k, v in NAMES.items() if k == name or v == name or name in v]
        for key in keys:
            path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
            print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
