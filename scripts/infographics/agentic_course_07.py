"""Infographics for docs/projects/agentic-ai-complete-course/07-guardrails.md.

Chapter 7 of the "Complete Agentic AI Course in 10 Hours" (guardrails, 8:45:43
to 9:22:55). Four boards redraw the instructor's Excalidraw pages, two redraw
tables or stacks he shows in the notebook, and two are explanatory boards that
the video does not show. Run from the repo root:

    python3 scripts/infographics/agentic_course_07.py            # all boards
    python3 scripts/infographics/agentic_course_07.py layered_stack
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import FAINT, INK, MONO, PALETTE, Board, esc  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "agentic-course"
PREFIX = "07-"
BOARDS = {}


def board(fn):
    BOARDS[fn.__name__] = fn
    return fn


def _col(color):
    return PALETTE[color]["stroke"] if color in PALETTE else color


def line(b, pts, color=INK, width=1.8, dashed=False):
    """A plain polyline with no arrowhead."""
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    b.parts.append(
        f'<path d="{d}" fill="none" stroke="{_col(color)}" stroke-width="{width}"{dash} '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )


def brace(b, x, y1, y2, color=INK, d=12, facing="right"):
    """A curly brace from y1 to y2 whose point faces left or right."""
    ym = (y1 + y2) / 2
    s = 1 if facing == "right" else -1
    path = (
        f"M{x},{y1} q{s * d},0 {s * d},{d} V{ym - d} q0,{d} {s * d},{d} "
        f"q{-s * d},0 {-s * d},{d} V{y2 - d} q0,{d} {-s * d},{d}"
    )
    b.parts.append(
        f'<path d="{path}" fill="none" stroke="{_col(color)}" stroke-width="2" stroke-linecap="round"/>'
    )
    return (x + s * 2 * d, ym)


def circle_num(b, cx, cy, n, color="orange", r=14):
    c = PALETTE[color]
    b.parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="2"/>'
    )
    b.parts.append(
        f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-family="{MONO}" font-size="14" '
        f'font-weight="700" fill="{c["text"]}">{n}</text>'
    )


# ------------------------------------------------------------------ board 1


@board
def agent_pipeline():
    """8:46:00 to 8:49:45: the first Excalidraw page, guardrails around an agent."""
    b = Board(1200, 620, "Guardrails sit around the agent pipeline",
              "Input in, LLM and tools in the middle, response out: a check on every edge")

    b.group(330, 100, 520, 350, "AI agent", "grey", label_pos="bottom")
    llm = b.card(370, 250, 160, 90, "LLM", ["reasons and answers"], "yellow")
    tools = b.card(620, 140, 200, 120, "Tools", ["RAG / vector database", "APIs", "MCP server", "packages"],
                   "orange")

    inp = b.card(20, 250, 140, 90, "Input", ["text or an image"], "pink")
    d_in = b.diamond(250, 295, 120, 80, "input\ncheck", "orange")
    blocked = b.card(160, 110, 160, 90, "Flagged", ["'how to hack a", "server?' stops", "here"], "red", size=12)
    d_out = b.diamond(940, 322, 120, 80, "output\ncheck", "green")
    resp = b.card(1040, 277, 140, 90, "Response", ["validated,", "compliant"], "green")

    b.arrow(inp.right(), d_in.left(), color="pink")
    b.arrow(d_in.right(), llm.left(), color="orange")
    b.arrow(d_in.top(), blocked.bottom(0.5), color="red", label="FLAG", label_dx=26)
    b.arrow(llm.top(), tools.left(), via=[(llm.cx, 200)], color="orange", label="tool call")
    b.arrow(tools.bottom(0.5), llm.right(0.3), via=[(tools.cx, 277)], color="orange", label="context",
            label_dx=34, label_dy=-6)
    b.arrow(llm.right(0.8), d_out.left(), color="green", label="draft answer", label_dy=-18)
    b.arrow(d_out.right(), resp.left(), color="green")

    b.card(20, 505, 360, 90, "Safe, appropriate inputs", ["only what passes the input", "check reaches the LLM"],
           "orange")
    b.card(420, 505, 360, 90, "Approved actions", ["tools run only when allowed", "(RAG, APIs, MCP)"], "purple")
    b.card(820, 505, 360, 90, "Validated outputs", ["the user only sees a", "compliant response"], "green")
    b.text(600, 488, "the three promises of a guardrail", 13, FAINT, anchor="middle")
    return b


# ------------------------------------------------------------------ board 2


@board
def two_approaches():
    """8:50:00 to 8:53:00: deterministic versus model-based."""
    b = Board(1160, 760, "Two ways to build a guardrail",
              "Rules are free and blunt; a model understands meaning but bills per call")

    g = b.card(30, 235, 190, 90, "Guardrails", ["the check you add"], "pink", title_size=17)
    det = b.group(290, 90, 840, 190, "Deterministic approach", "orange")
    mod = b.group(290, 300, 840, 190, "Model-based approach", "blue")
    b.arrow(g.right(0.3), det.left(0.5), via=[(255, g.y + 27), (255, 185)], color="orange")
    b.arrow(g.right(0.7), mod.left(0.5), via=[(255, g.y + 63), (255, 395)], color="blue")

    b.card(310, 130, 260, 130, "Rule-based", ["regex, keyword matching,", "fixed lists of banned words"], "orange")
    b.card(585, 130, 250, 130, "Zero LLM cost", ["no model call at all, so", "free and instant"], "green")
    b.card(850, 130, 260, 130, "Blind to meaning", ["cannot understand semantics,", "so it over-blocks and", "under-blocks"], "red")

    b.card(310, 340, 195, 130, "Use an LLM", ["send the input, ask", "safe or unsafe"], "blue")
    b.card(520, 340, 195, 130, "Semantic meaning", ["reads intent, not just", "the words used"], "green")
    b.card(730, 340, 195, 130, "Catches violations", ["describe the rule in", "a prompt"], "green")
    b.card(940, 340, 170, 130, "LLM call cost", ["every input is a", "paid, slower call"], "red")

    b.group(30, 520, 1100, 210, "Same three inputs through both guards (the notebook demo)", "grey")
    rows = [
        ["input", "keyword rule", "gpt-4o-mini judge"],
        ["How do I hack into a database?", "BLOCKED", "UNSAFE"],
        ["What is the capital of France?", "allowed", "SAFE"],
        ["Explain how malware spreads", "BLOCKED (false alarm)", "SAFE"],
    ]
    b.table(110, 565, [400, 280, 260], rows, header_color="grey", size=13, row_h=34)
    return b


# ------------------------------------------------------------------ board 3


@board
def middleware_menu():
    """8:53:00 to 8:59:00: LangChain, guardrails, middleware, and the five kinds."""
    b = Board(1240, 700, "LangChain implements guardrails as middleware",
              "Hooks before the agent, around the model and tools, and after the agent")

    p1 = b.pill(380, 88, "LangChain", "blue", 15)
    p2 = b.pill(560, 88, "Guardrails", "pink", 15)
    p3 = b.pill(740, 88, "Middleware", "orange", 15, solid=True)
    b.arrow(p1.right(), p2.left(), color="blue")
    b.arrow(p2.right(), p3.left(), color="pink")

    b.group(20, 150, 580, 250, "Built in", "blue")
    b.group(640, 150, 580, 250, "Custom hooks you write", "green")
    b.group(20, 430, 1200, 240, "Combine them", "purple")

    c1 = b.card(40, 190, 265, 190, "1  PII middleware",
                ["detects email, credit card,", "IP, URL", "masks or hashes", "works on input, output,", "and tool calls"],
                "blue", size=12, bullets=False)
    c2 = b.card(320, 190, 265, 190, "2  Human in the loop",
                ["pauses before sensitive", "tools", "waits: approve or reject", "needs a thread and a", "checkpointer"],
                "blue", size=12)
    c3 = b.card(660, 190, 265, 190, "3  before_agent hook",
                ["runs before any LLM call", "zero LLM cost for blocked", "requests", "a blocked request jumps", "straight to the end"],
                "green", size=12)
    c4 = b.card(940, 190, 265, 190, "4  after_agent hook",
                ["validates the final response", "before the user sees it", "can replace or mutate", "unsafe content", "a small, cheap model will do"],
                "green", size=12)
    c5 = b.card(40, 475, 1160, 170, "5  Layered guardrails",
                ["stack everything above in one middleware list: cheap deterministic checks first,",
                 "PII handling next, human approval on risky tools, a model-based output check last.",
                 "Defence in depth: no single layer has to be perfect."],
                "purple", size=13)
    return b


# ------------------------------------------------------------------ board 4


@board
def pii_hitl_flow():
    """9:06:45 to 9:22:15: the page he draws while explaining PII and human in the loop."""
    b = Board(1180, 620, "PII middleware and human in the loop around one agent",
              "Redrawn: PII is cleaned before the agent runs; a human gates the sensitive tool")

    inp = b.card(20, 285, 130, 80, "Input", ["user message"], "pink")
    pii = b.card(190, 255, 190, 140, "PII middleware",
                 ["credit card: mask", "email: redact", "API key: block"], "orange")
    agent = b.card(450, 265, 200, 120, "Agent", ["LLM that plans", "tool calls"], "yellow", title_size=17)
    hitl = b.diamond(780, 150, 200, 120, "human in the\nloop\nmiddleware", "red")
    h = b.person(1060, 80, "red", 0.9, "human")
    tool = b.card(900, 270, 170, 80, "Tool", ["send_email,", "delete_records"], "purple")
    outp = b.card(1010, 470, 150, 90, "Output", ["checked on the", "way out too"], "green")

    b.arrow(inp.right(), pii.left(), color="pink")
    b.arrow(pii.right(), agent.left(), color="orange", label="scrubbed", label_dy=-14)
    b.arrow(agent.top(0.5), hitl.left(), via=[(agent.cx, 150)], color="red", label="tool call", label_dy=-14)
    b.arrow(hitl.right(), (1030, 150), color="red", dashed=True, label="asks", label_dy=-14)
    b.arrow((1060, 200), tool.top(0.8), via=[(1060, 240), (1036, 240)], color="red", label="approve", label_dx=34, label_dy=0)
    b.arrow(tool.left(0.5), agent.right(0.5), color="purple", label="result", label_dy=-14)
    b.arrow(agent.bottom(0.5), outp.left(0.5), via=[(agent.cx, 515)], color="green", label="response", label_dy=-14)

    b.text(40, 470, "What the PII middleware looks for: credit card, email, API key", 13, "orange", "700", anchor="start")
    b.text(40, 496, "Strategies: redact / mask / hash / block", 13, "orange", "700", anchor="start")
    b.text(40, 540, "Pause decisions: approve, edit or reject", 13, "red", "700", anchor="start")
    b.text(40, 566, "Needs a thread and a checkpointer to resume", 13, "red", "700", anchor="start")
    return b


# ------------------------------------------------------------------ board 5


@board
def pii_types_strategies():
    """9:05:00 to 9:06:00: the two tables the notebook shows for PIIMiddleware."""
    b = Board(1120, 520, "PIIMiddleware: what it detects and what it does about it",
              "Both tables are shown on screen in the notebook")
    b.group(20, 90, 520, 300, "Supported PII types", "blue")
    b.table(40, 130, [190, 310],
            [["type", "example"],
             ["email", "user@example.com"],
             ["credit_card", "5105-1051-0510-5100"],
             ["ip", "192.168.1.1"],
             ["mac_address", "00:1A:2B:3C:4D:5E"],
             ["url", "https://secret-site.com"]],
            header_color="blue", size=13, row_h=36)
    b.group(580, 90, 520, 300, "Strategies", "orange")
    b.table(600, 130, [150, 350],
            [["strategy", "result"],
             ["redact", "[REDACTED_EMAIL]"],
             ["mask", "****-****-****-1234"],
             ["hash", "a8f5f167..."],
             ["block", "raises an exception"]],
            header_color="orange", size=13, row_h=36)
    b.card(20, 420, 1080, 80, "Custom type: api_key",
           ["not built in: you name it yourself and pass a regex as detector=r\"sk-[a-zA-Z0-9]{32}\", with strategy=\"block\""],
           "purple", size=12)
    return b


# ------------------------------------------------------------------ board 6


@board
def layered_stack():
    """9:22:30: the layered stack the notebook prints before the combined agent."""
    b = Board(980, 760, "Layered guardrails: one middleware list, five layers",
              "The notebook's picture of the stack, top to bottom")
    x, w = 120, 440
    ys = [140, 228, 316, 404, 492]
    items = [
        ("Layer 1", "ContentFilterMiddleware", "deterministic input filter", "orange"),
        ("Layer 2", "PIIMiddleware (input)", "PII redaction on input", "blue"),
        ("Layer 3", "HumanInTheLoopMiddleware", "approval for sensitive tools", "red"),
        ("Layer 4", "PIIMiddleware (output)", "PII redaction on output", "blue"),
        ("Layer 5", "SafetyGuardrailMiddleware", "model-based output safety", "green"),
    ]
    top = b.card(x, 84, w, 40, "User input", [], "pink")
    prev = top
    for (lab, name, note, colr), y in zip(items, ys):
        c = b.card(x, y, w, 62, f"{lab}  {name}", [], colr, size=12)
        b.arrow(prev.bottom(), c.top(), color="grey")
        b.text(x + w + 20, y + 36, note, 13, colr, "700", anchor="start")
        prev = c
    end = b.card(x, 590, w, 40, "User response", [], "pink")
    b.arrow(prev.bottom(), end.top(), color="grey")
    b.card(x, 660, w, 70, "Cheap checks first",
           ["blocked requests never reach the LLM, so", "they cost nothing"], "grey", size=12)
    return b


# ------------------------------------------------------------------ board 7


@board
def hook_timeline():
    """Explanatory board (not shown in the video): where each guardrail hooks in."""
    b = Board(1240, 700, "Where each guardrail attaches to an agent run",
              "Explanatory board (not shown in the video)")
    b.group(20, 100, 1200, 330, "One agent.invoke(...) call", "grey")
    ba = b.card(40, 170, 170, 110, "before_agent", ["runs once, at the", "very start"], "green")
    b.group(260, 140, 740, 270, "the agent loop repeats until the model stops calling tools", "blue", label_pos="bottom")
    bm = b.card(280, 170, 160, 110, "before_model", ["runs before each", "model call"], "blue")
    m = b.card(480, 170, 130, 110, "model", ["LLM call"], "yellow", title_size=16)
    am = b.card(650, 170, 160, 110, "after_model", ["runs after each", "model reply"], "blue")
    t = b.card(860, 170, 100, 110, "tools", ["run if the", "model asked"], "purple", title_size=14)
    aa = b.card(1030, 170, 170, 110, "after_agent", ["runs once, at the", "very end"], "green")
    b.arrow(ba.right(), bm.left(), color="green")
    b.arrow(bm.right(), m.left())
    b.arrow(m.right(), am.left())
    b.arrow(am.right(), t.left(), color="purple")
    b.arrow(t.top(0.5), bm.top(0.5), via=[(910, 150), (360, 150)], color="blue", dashed=True,
            label="tool results go back round to the model", label_dy=-4)
    b.arrow(am.bottom(0.5), aa.bottom(0.5), via=[(am.cx, 330), (aa.cx, 330)], color="green",
            label="no more tool calls: finish", label_dy=-4)

    b.card(20, 470, 260, 180, "Input guardrails",
           ["ContentFilterMiddleware", "(custom, before_agent)", "PIIMiddleware on input", "(built in, before the model)"],
           "orange", size=12)
    b.card(310, 470, 290, 180, "Action guardrails",
           ["HumanInTheLoopMiddleware", "(built in, after the model has", "proposed a tool call and", "before the tool runs)"],
           "red", size=12)
    b.card(630, 470, 290, 180, "Output guardrails",
           ["PIIMiddleware on output", "(built in, after the model)", "SafetyGuardrailMiddleware", "(custom, after_agent)"],
           "green", size=12)
    b.card(950, 470, 270, 180, "Order of hooks",
           ["before_* hooks run first to", "last in the list; after_*", "hooks run last to first"],
           "purple", size=12)
    return b


# ------------------------------------------------------------------ board 8


@board
def hitl_pause_resume():
    """Explanatory board (not shown in the video): pause and resume in human in the loop."""
    b = Board(1240, 700, "Human in the loop: pause, decide, resume on the same thread",
              "Explanatory board (not shown in the video)")
    s1 = b.card(20, 130, 220, 120, "1  invoke", ["messages + config with", "thread_id session_001"], "blue")
    s2 = b.card(280, 130, 220, 120, "2  model proposes", ["a send_email tool call", "(search_web would skip", "the pause)"], "yellow")
    s3 = b.card(540, 130, 220, 120, "3  middleware pauses", ["interrupt_on says", "send_email: True", "so nothing is sent yet"], "red")
    s4 = b.card(800, 130, 200, 120, "4  state saved", ["checkpointer", "InMemorySaver keeps", "the paused run"], "purple")
    s5 = b.card(1040, 130, 180, 120, "5  you see it", ["result carries an", "__interrupt__ with", "the proposed call"], "orange")
    b.arrow(s1.right(), s2.left())
    b.arrow(s2.right(), s3.left())
    b.arrow(s3.right(), s4.left())
    b.arrow(s4.right(), s5.left())

    h = b.person(130, 330, "red", 0.9, "human reviews")
    b.arrow(s5.bottom(0.5), (1130, 340), via=[(1130, 340)], color="orange")
    cmd = b.card(330, 340, 790, 70, "6  invoke(Command(resume=...), config=same thread_id)",
                 ["a list of decisions, one per paused tool call"], "teal", size=12)
    b.arrow(h.right(), cmd.left(), color="red")

    ap = b.card(250, 500, 260, 150, "approve", ["the tool runs as proposed", "email sent, model writes", "the final reply"], "green")
    ed = b.card(560, 500, 260, 150, "edit", ["change the tool arguments", "first, then it runs", "(allowed by the middleware)"], "yellow")
    rj = b.card(870, 500, 260, 150, "reject", ["the tool is skipped and the", "model is told it was refused", "so it can reply politely"], "red")
    b.arrow(cmd.bottom(0.2), ap.top(0.5), color="green")
    b.arrow(cmd.bottom(0.5), ed.top(0.5), color="yellow")
    b.arrow(cmd.bottom(0.8), rj.top(0.5), color="red")
    b.text(20, 590, "Without a checkpointer\nthere is nothing to resume", 12, "red", "700", anchor="start")
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"{PREFIX}{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
