"""Infographics for docs/projects/agentic-ai-complete-course/06-deep-agents.md.

Seven boards redraw the instructor's whiteboard pages and the notebook / docs
pictures he shows between 8:02:11 and 8:45:43 of the course video, and one is an
explanatory board added for the notes (labelled as such in the chapter).
Run from the repo root:

    python3 scripts/infographics/agentic_course_06.py            # all boards
    python3 scripts/infographics/agentic_course_06.py shallow    # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import FAINT, INK, MONO, PALETTE, Board, esc  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "agentic-course"
BOARDS = {}


def board(fn):
    BOARDS[fn.__name__] = fn
    return fn


# --------------------------------------------------------------------- helpers


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


def circle_num(b, cx, cy, n, color="orange", r=13):
    c = PALETTE[color]
    b.parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="2"/>'
    )
    b.parts.append(
        f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-family="{MONO}" font-size="14" '
        f'font-weight="700" fill="{c["text"]}">{n}</text>'
    )


# ---------------------------------------------------------------- board 1


@board
def shallow():
    b = Board(1100, 600, "Shallow agent: one pass, no plan",
              "Redrawn from the instructor's whiteboard, 8:03 to 8:07")

    b.group(30, 90, 1040, 270, "Ask, call a tool, answer", "blue")
    q = b.card(60, 160, 230, 96, "Input", ["What is the current", "temperature of", "Bangalore or Paris?"], "red")
    llm = b.card(350, 160, 210, 96, "LLM", ["acts as the brain", "decides: answer, or", "call a tool?"], "dark")
    tool = b.card(640, 118, 210, 80, "Tool", ["SERP API / Tavily /", "a weather API"], "blue")
    out = b.card(890, 190, 160, 80, "Output", ["Paris is", "so and so degrees"], "yellow")
    b.arrow(q.right(), llm.left())
    b.arrow(llm.top(0.8), tool.left(), via=[(518, 158)], label="no live data,\ncall a tool", label_dy=-22, color="blue")
    b.arrow(tool.right(), out.top(), via=[(970, 158)], label="tool result", color="blue")
    b.arrow(llm.bottom(), out.bottom(), via=[(455, 318), (970, 318)], dashed=True,
            label="or answer directly, if it already knows", color="grey")

    b.group(30, 390, 1040, 190, "Why the instructor calls it shallow", "orange")
    b.card(55, 440, 310, 110, "No explicit planning",
           ["A query arrives, the LLM", "reacts, an answer leaves.", "Nothing is broken into steps."], "orange")
    b.card(395, 440, 310, 110, "Complex queries fail",
           ["\"AI news today, how it links", "to economics, and physics", "advances\" needs splitting up."], "orange")
    b.card(735, 440, 310, 110, "Limited context retention",
           ["One flow, one output.", "Little context is built up", "or kept along the way."], "orange")
    return b


# ---------------------------------------------------------------- board 2


@board
def react():
    b = Board(1100, 620, "ReAct agent: a loop, but still shallow",
              "Redrawn from the instructor's whiteboard, 8:07 to 8:10")

    b.group(30, 90, 760, 330, "LLM and tools in a loop", "green")
    sysp = b.card(60, 130, 200, 56, "System prompt", [], "purple", title_size=13)
    q = b.card(60, 240, 220, 100, "Input", ["What is 2 + 2 and", "then multiply by 5?"], "blue")
    llm = b.card(340, 220, 170, 110, "LLM", ["picks which tool,", "reads the result"], "dark")
    tools = b.card(590, 120, 170, 250, "Tools", ["Wikipedia", "search API", "Tavily", "calculator", "... any number"], "teal", align="center")
    out = b.card(340, 355, 170, 40, "Output", [], "yellow", title_size=13)
    b.arrow(sysp.right(), llm.top(0.3), via=[(380, 158)], color="purple")
    b.arrow(q.right(), llm.left())
    b.arrow(llm.right(0.25), tools.left(0.35), label="act", color="green", label_dy=-14)
    b.arrow(tools.left(0.75), llm.right(0.75), label="observe", color="green", label_dy=14)
    b.arrow(llm.bottom(), out.top(), label="done", label_dx=22)
    b.text(675, 400, "loop: as many times as needed", 12, "green", "700")

    tip = brace(b, 810, 120, 400, "red", facing="right")
    b.card(860, 120, 210, 280, "Still a shallow agent",
           ["No planning", "No structured plan", "No deep reasoning", "No state management", "No persistent memory"],
           "red", bullets=True, align="left")

    b.group(30, 450, 1040, 150, "ReAct, in one line", "grey")
    b.text(550, 505, "Reason about the situation, Act with a tool, Observe the result, repeat.", 15, INK, "700")
    b.text(550, 540, "Better than a single pass because the loop can repeat, but it is still just LLM + tools.", 13, FAINT)
    b.text(550, 566, "(Aloud, the instructor expands the name as \"act\" and \"read\"; the usual expansion is Reason + Act.)", 12, FAINT, italic=True)
    return b


# ---------------------------------------------------------------- board 3


@board
def four_parts():
    b = Board(1100, 700, "Deep agent: four core components",
              "Redrawn from the instructor's whiteboard, 8:10 to 8:16")

    b.card(40, 90, 1020, 50, "Deep agent  [ Deep Research in ChatGPT, Claude, Manus AI ]  =>  his own product, Xenodocs",
           [], "grey", title_size=14)

    centre = b.card(430, 300, 240, 100, "Deep agent", ["built on LangGraph"], "dark", title_size=20)
    plan = b.card(90, 180, 260, 92, "1. Planning tool", ["a to-do list before", "any real work starts"], "orange")
    sub = b.card(750, 180, 260, 92, "2. Sub agents", ["workers that carry out", "the to-do items"], "pink")
    sysp = b.card(750, 440, 260, 92, "3. System prompt", ["tone, behaviour, coding", "style, rules"], "purple")
    fs = b.card(90, 440, 260, 92, "4. File system", ["persistent memory shared", "by all sub agents"], "teal")
    b.arrow(centre.left(0.2), plan.right(0.8), color="orange")
    b.arrow(centre.right(0.2), sub.left(0.8), color="pink")
    b.arrow(centre.right(0.8), sysp.left(0.2), color="purple")
    b.arrow(centre.left(0.8), fs.right(0.2), color="teal")
    b.arrow(plan.right(0.2), sub.left(0.2), via=[(390, 198), (710, 198)], color="grey",
            label="items handed out", label_dy=-14)

    b.pill(550, 428, "His example: Claude Code", "orange", size=13, anchor="middle")
    b.text(550, 484, "planning is a to-do list,", 12, INK)
    b.text(550, 502, "the system prompt is public,", 12, INK)
    b.text(550, 520, "work is decomposed and delegated", 12, INK)

    b.group(40, 570, 1020, 110, "Compare", "grey")
    b.text(550, 628, "Shallow agent: LLM + tools, one pass.   ReAct agent: LLM + tools, looped.", 14, INK)
    b.text(550, 652, "Deep agent: plan, delegate, remember, all steered by a system prompt.", 14, INK, "700")
    return b


# ---------------------------------------------------------------- board 4


@board
def planning_to_subagents():
    b = Board(1100, 760, "From one request to a to-do list to sub agents",
              "Redrawn from the instructor's whiteboard, 8:13 to 8:17")

    req = b.card(40, 100, 420, 80, "Request",
                 ["Plan a holiday to Paris, budget 100k rupees,", "3 nights and 4 days"], "blue")
    sysp = b.card(760, 100, 300, 80, "System prompt", ["how the agent should behave"], "purple")

    b.group(40, 215, 440, 330, "1. Planning tool: a to-do list", "orange")
    todo = b.card(65, 265, 390, 260, "To-do list",
                  ["Day 1: travel to Paris,", "        stay at this hotel, price",
                   "Day 2: breakfast, then go to", "        the Eiffel Tower",
                   "Day 3: visit another place",
                   "Day 4: fly back to India",
                   "Cost for each day, what to", "book and what not to book"], "orange", align="left", size=12)

    b.group(560, 215, 500, 330, "2. Sub agents execute the items", "pink")
    subs = []
    for i, txt in enumerate(["Sub agent 1: day 1", "Sub agent 2: day 2", "Sub agent 3: day 3", "Sub agent 4: day 4"]):
        subs.append(b.card(590, 262 + i * 66, 250, 50, txt, [], "pink", title_size=13))
    b.card(880, 262, 150, 252, "Each one",
           ["has its own", "slice of the", "to-do list and", "reports back"], "pink", size=12)

    mem = b.cylinder(300, 600, 500, 120, "3. File system = persistent memory",
                     ["every sub agent can read and write the same files", "(notes, drafts, results)"], "teal")
    b.arrow(req.bottom(), (250, 215), color="blue")
    b.arrow(sysp.bottom(), (810, 215), color="purple", dashed=True)
    b.arrow(todo.right(0.5), (558, 395), color="orange", label="hand out", label_dy=-14)
    b.arrow((715, 545), (715, 600), color="teal", both=True, label="read / write", label_dx=62)
    return b


# ---------------------------------------------------------------- board 5


@board
def blog_example():
    b = Board(1100, 640, "Example: research and write a blog",
              "Redrawn from the instructor's whiteboard, 8:17 to 8:19")

    topic = b.card(40, 90, 260, 70, "Topic on a blog", ["I give a topic, the deep agent", "produces the blog"], "blue")
    b.group(40, 190, 260, 420, "To-do list", "orange")
    items = []
    for i, t in enumerate(["Research", "More research", "Write the blog", "Copyright check"]):
        items.append(b.card(65, 240 + i * 90, 210, 60, f"{i + 1}. {t}", [], "orange", title_size=14))
    b.arrow(topic.bottom(), (170, 190), color="blue")

    b.group(350, 190, 710, 420, "One sub agent per item, each with the right access", "pink")
    rows = [
        ("Sub agent", "internet access", "searches the web"),
        ("Sub agent", "arXiv access", "pulls research papers"),
        ("Sub agent", "writing expert", "drafts the blog post"),
        ("Sub agent", "internet access", "checks for copied text"),
    ]
    for i, (a, bb, c) in enumerate(rows):
        y = 240 + i * 90
        sa = b.card(380, y, 170, 60, a, [], "pink", title_size=14)
        acc = b.card(640, y, 190, 60, bb, [], "teal", title_size=14)
        res = b.card(880, y, 160, 60, c, [], "grey", size=11, title_size=11)
        b.arrow(items[i].right(), sa.left(), color="orange")
        b.arrow(sa.right(), acc.left(), color="pink")
        b.arrow(acc.right(), res.left(), color="teal")
    b.text(700, 596, "the items can run in parallel, then the pieces come together", 13, "pink", "700")
    return b


# ---------------------------------------------------------------- board 6


def _node(b, x, y, w, label, color="purple", h=34, size=12, rounded=False):
    c = PALETTE[color]
    b.parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2 if rounded else 7}" fill="{c["fill"]}" '
        f'stroke="{c["stroke"]}" stroke-width="1.6"/>'
    )
    b.parts.append(
        f'<text x="{x + w / 2}" y="{y + h / 2 + 4}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="{size}" font-weight="700" fill="{c["text"]}">{esc(label)}</text>'
    )
    from board import Box
    return Box(x, y, w, h)


@board
def simple_vs_deep_graph():
    b = Board(1100, 780, "Same model, same tool: two different graphs",
              "Redrawn from the LangGraph pictures printed in the notebook, 8:38 to 8:45")

    b.group(30, 90, 360, 660, "create_agent(...)", "blue")
    st = _node(b, 150, 140, 120, "__start__", "grey", rounded=True)
    mo = _node(b, 150, 250, 120, "model", "blue")
    to = _node(b, 250, 450, 100, "tools", "teal")
    en = _node(b, 70, 450, 110, "__end__", "purple", rounded=True)
    b.arrow(st.bottom(), mo.top())
    b.arrow(mo.bottom(0.2), en.top(0.5), color="grey", dashed=True)
    b.arrow(mo.bottom(0.8), to.top(0.4), color="teal", dashed=True, label="tool call", label_dx=40)
    b.arrow(to.top(0.8), mo.right(0.9), via=[(330, 400), (330, 277)], color="teal", dashed=True)
    b.text(210, 560, "Just nodes, edges and a loop.", 13, INK, "700")
    b.text(210, 585, "No hooks around the model.", 13, INK)
    b.text(210, 625, "Customise it by adding", 13, FAINT)
    b.text(210, 645, "middleware yourself.", 13, FAINT)

    b.group(430, 90, 640, 660, "create_deep_agent(...)", "orange")
    st2 = _node(b, 620, 140, 120, "__start__", "grey", rounded=True)
    p = _node(b, 520, 215, 340, "PatchToolCallsMiddleware.before_agent", "orange")
    s = _node(b, 520, 300, 340, "SummarizationMiddleware.before_model", "orange")
    m2 = _node(b, 580, 385, 120, "model", "blue")
    t = _node(b, 520, 470, 340, "TodoListMiddleware.after_model", "orange")
    to2 = _node(b, 900, 470, 100, "tools", "teal")
    en2 = _node(b, 620, 560, 120, "__end__", "purple", rounded=True)
    b.arrow(st2.bottom(), p.top())
    b.arrow(p.bottom(), s.top())
    b.arrow(s.bottom(0.5), m2.top(0.5))
    b.arrow(m2.bottom(0.5), t.top(0.5))
    b.arrow(t.bottom(0.5), en2.top(0.5), color="grey", dashed=True)
    b.arrow(t.right(0.5), to2.left(0.5), color="teal", dashed=True, label="tool call", label_dy=-20)
    b.arrow(to2.top(0.5), s.right(0.5), via=[(950, 317)], color="teal", dashed=True)
    b.arrow(t.left(0.3), s.left(0.5), via=[(490, 480), (490, 317)], color="orange", dashed=True)
    b.text(980, 345, "results go back\nthrough the\nsummariser", 11, "teal", "700", anchor="end")
    b.text(500, 405, "loop back\nfor the next\nstep", 11, "orange", "700", anchor="end")

    b.card(460, 610, 590, 120, "What each hook is for",
           ["before_agent: repair tool calls that never got an answer", "before_model: summarise when the history grows too long",
            "after_model: allow at most one to-do update per model turn",
            "(the write_todos tool itself is added by the same middleware)"], "yellow", size=12, align="left")
    return b


# ---------------------------------------------------------------- board 7


@board
def customisation_map():
    b = Board(1100, 640, "Customising create_deep_agent",
              "Redrawn from the LangChain docs page the instructor opens at 8:45, as the map for part two")

    cd = b.card(50, 270, 220, 70, "create_deep_agent", [], "dark", title_size=15)
    cfg = b.card(340, 160, 180, 56, "Core config", [], "blue", title_size=14)
    feat = b.card(340, 410, 180, 56, "Features", [], "orange", title_size=14)
    mdl = b.card(620, 80, 200, 50, "Model", [], "blue", title_size=14)
    sysp = b.card(620, 160, 200, 50, "System prompt", [], "blue", title_size=14)
    tls = b.card(620, 240, 200, 50, "Tools", [], "blue", title_size=14)
    be = b.card(620, 360, 200, 50, "Backend", [], "orange", title_size=14)
    sa = b.card(620, 440, 200, 50, "Subagents", [], "orange", title_size=14)
    it = b.card(620, 520, 200, 50, "Interrupts", [], "orange", title_size=14)
    out = b.card(890, 290, 170, 80, "Customised", ["agent"], "green", title_size=15)
    b.arrow(cd.right(0.2), cfg.left(), color="blue")
    b.arrow(cd.right(0.8), feat.left(), color="orange")
    for t, col in [(mdl, "blue"), (sysp, "blue"), (tls, "blue")]:
        b.arrow(cfg.right(), t.left(), color=col, width=1.4)
        b.arrow(t.right(), out.left(0.3), color=col, width=1.2)
    for t, col in [(be, "orange"), (sa, "orange"), (it, "orange")]:
        b.arrow(feat.right(), t.left(), color=col, width=1.4)
        b.arrow(t.right(), out.left(0.7), color=col, width=1.2)

    b.card(50, 590, 1010, 36,
           "Part one (this chapter) used model, system prompt and tools. Backend, subagents and interrupts are promised for part two.",
           [], "grey", size=12, title_size=12)
    return b


# ---------------------------------------------------------------- board 8


@board
def invoke_flow():
    b = Board(1100, 800, "What one invoke() does",
              "Explanatory board (not shown in the video), based on the notebook run at 8:40 to 8:44")

    b.group(30, 90, 1040, 310, "The loop inside the graph", "orange")
    user = b.card(50, 140, 210, 80, "invoke(...)", ["\"What is deepagent?\"", "as a user message"], "blue", size=11)
    mw = b.card(295, 140, 200, 80, "Middleware hooks", ["repair dangling calls,", "summarise if long"], "orange", size=11)
    mo = b.card(530, 140, 220, 80, "Model", ["Qwen3 32B on Groq,", "decides the next step"], "dark", size=11)
    choose = b.diamond(910, 180, 230, 110, "plan, call a\ntool, or answer?", "yellow", size=12)
    tool = b.card(530, 295, 220, 70, "web_search (Tavily)", ["returns a big JSON result"], "teal", size=11)
    b.arrow(user.right(), mw.left())
    b.arrow(mw.right(), mo.left())
    b.arrow(mo.right(), choose.left(), color="grey")
    b.arrow(choose.bottom(), tool.right(), via=[(910, 330)], color="teal", label="tool call", label_dx=-4, label_dy=-14)
    b.arrow(tool.top(), mo.bottom(), color="teal", dashed=True, label="result goes back", label_dx=60)

    ev = b.card(40, 450, 470, 190, "If the result is too large (over about 20,000 tokens)",
                ["1. The full text is written to the virtual file system", "   at /large_tool_results/<tool-call id>",
                 "2. The model only sees a short note: \"Tool result too", "   large ... saved in the filesystem at this path ...\"",
                 "3. It can read_file / grep the file later if it needs it"], "teal", size=11, align="left")
    res = b.card(560, 450, 500, 190, "What comes back in result",
                 ["messages  the whole conversation, last one is the answer",
                  "files     the virtual files, path -> content, timestamps",
                  "todos     only if the model chose to write a to-do list",
                  "(his run printed messages and files, no todos)"], "green", size=11, align="left")
    b.arrow(tool.bottom(0.3), ev.top(0.9), via=[(596, 425), (463, 425)], color="teal", dashed=True, label="too big?", label_dx=60)
    b.arrow(choose.right(), res.top(0.96), via=[(res.x + res.w * 0.96, 180)], color="green", label="answer", label_dy=-4)
    b.card(40, 680, 1020, 80, "Why this matters",
           ["A deep agent keeps the conversation small by moving bulky material into files and reading it back on demand.",
            "That is how it can run for a long time without the context window filling up."], "yellow", size=12, align="left")
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"06-{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
