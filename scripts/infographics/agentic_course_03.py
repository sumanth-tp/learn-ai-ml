"""Infographics for docs/projects/agentic-ai-complete-course/03-langgraph.md.

Each function redraws one board from the LangGraph section of the course
(Krish Naik's Excalidraw pages and the rendered graph images in his notebooks),
or, where the instructor only talks, adds one clearly labelled explanatory board.
Run from the repo root:

    python3 scripts/infographics/agentic_course_03.py              # all boards
    python3 scripts/infographics/agentic_course_03.py roadmap      # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import FAINT, INK, MONO, PALETTE, Board, Box, esc  # noqa: E402

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


def cross(b, cx, cy, s=13, color="red", width=4):
    line(b, [(cx - s, cy - s), (cx + s, cy + s)], color, width)
    line(b, [(cx - s, cy + s), (cx + s, cy - s)], color, width)


def tick(b, cx, cy, color="green", s=10, width=4):
    line(b, [(cx - s, cy), (cx - s / 3, cy + s * 0.8), (cx + s, cy - s * 0.8)], color, width)


def brace(b, x, y1, y2, color=INK, d=12, facing="right"):
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


def mono(b, x, y, text, size=12, color=INK, weight="400", anchor="start"):
    fill = PALETTE[color]["text"] if color in PALETTE else color
    b.parts.append(
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{MONO}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" xml:space="preserve">{esc(text)}</text>'
    )


def frame(b, x, y, w, h, color="grey", fill="#ffffff", width=2.0, rx=10, dashed=False):
    dash = ' stroke-dasharray="8 5"' if dashed else ""
    b.parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{_col(color)}" stroke-width="{width}"{dash}/>'
    )


def gnode(b, cx, cy, text, kind="node", w=None, h=38, size=13):
    """A node the way LangGraph's own diagram draws it: pills for start/end, rounded boxes for nodes."""
    w = w or max(96, len(text) * size * 0.62 + 30)
    fills = {
        "start": ("#f8f9fa", "#868e96", INK),
        "end": ("#e5dbff", "#7048e8", "#5f3dc4"),
        "node": ("#e7f5ff", "#1c7ed6", "#1864ab"),
        "tool": ("#fff4e6", "#e8590c", "#c2410c"),
        "human": ("#ebfbee", "#2f9e44", "#2b8a3e"),
    }
    fill, stroke, fg = fills[kind]
    rx = h / 2 if kind in ("start", "end") else 8
    b.parts.append(
        f'<rect x="{cx - w / 2}" y="{cy - h / 2}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.8"/>'
    )
    mono(b, cx, cy + size * 0.36, text, size, fg, "700", "middle")
    return Box(cx - w / 2, cy - h / 2, w, h)


def caption_pill(b, x, y, text, color="green", solid=True, size=12):
    return b.pill(x, y, text, color, size=size, solid=solid)


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"03-{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


# ------------------------------------------------------------------- boards


@board
def roadmap():
    b = Board(1120, 640, "LangGraph crash course: the three parts",
              "Redrawn from the instructor's roadmap page. This video covers Part 1.")
    p1 = b.group(30, 100, 480, 300, "Part 1 · Fundamentals (this video)", "blue")
    items = ["Build a basic chatbot", "Tools, multiple tools", "Adding memory",
             "Add human in the loop", "Streaming techniques", "MCP, built from scratch"]
    for i, t in enumerate(items):
        y = 140 + i * 40
        b.text(70, y + 20, f"{i + 1})", 15, "blue", "700", anchor="start")
        b.text(105, y + 20, t, 15, INK, "400", anchor="start")
    caption_pill(b, 400, 160, "Graph API", "blue", solid=False)

    b.group(30, 430, 480, 180, "Part 2 · Advanced LangGraph", "orange")
    for i, t in enumerate(["Workflows and agents", "Multi agents (agents talking to agents)",
                           "Functional API", "Debugging and monitoring"]):
        b.text(60, 482 + i * 28, f"{i + 1})  {t}", 14, INK, "400", anchor="start")

    b.group(570, 100, 520, 300, "Part 3 · LangGraph agents, end to end projects", "purple")
    rows = [("LLMOps pipeline", "Hugging Face Spaces"), ("Deployments", "MLflow"),
            ("LLM evaluation metrics", "AWS, Grafana")]
    for i, (a, c) in enumerate(rows):
        y = 160 + i * 70
        b.card(600, y, 230, 48, a, [], "purple", title_size=14)
        b.arrow((830, y + 24), (870, y + 24), color="purple")
        b.card(870, y, 190, 48, c, [], "pink", title_size=13)
    b.text(830, 385, "projects, deployment, evaluation dashboards", 12, FAINT, anchor="middle", italic=True)

    b.card(570, 430, 520, 70, "Approx. length (his estimate)",
           ["Part 1 about 2 h 50 m  ·  Part 2 about 2 h  ·  Part 3 to follow"], "yellow", size=13)
    b.card(570, 520, 520, 90, "In this recording", [
        "Part 1 is complete (including MCP from scratch).",
        "Parts 2 and 3 are announced but are not in this video."], "green", size=13)
    return b


@board
def blog_workflow():
    b = Board(1120, 750, "Components of LangGraph, shown on a YouTube-to-blog workflow",
              "Redrawn from the instructor's whiteboard (2:46:00 to 2:57:00). Nodes do the work, edges carry the flow.")
    b.card(30, 100, 300, 130, "Components of LangGraph", ["1  Edges", "2  Nodes", "3  State"], "blue",
           size=15, align="left", title_size=16)
    b.card(30, 250, 300, 108, "Two ways to build", ["1  Graph API  (used here)", "2  Functional API (later)"],
           "orange", size=14, align="left")
    b.card(30, 380, 300, 170, "The workflow (YT video to blog)", [
        "1  YT video  to  transcript",
        "2  transcript  to  title",
        "3  title + transcript  to  content"], "purple", size=13, align="left", title_size=14)
    b.text(30, 585, "LLMs are strong at content generation,", 13, FAINT, anchor="start", italic=True)
    b.text(30, 606, "so the title and content nodes call an LLM.", 13, FAINT, anchor="start", italic=True)

    cx = 560
    yt = b.card(cx - 110, 100, 220, 44, "YT URL", ["(the input, I/P)"], "red", size=12)
    st = gnode(b, cx, 185, "START", "start", w=130)
    n1 = b.card(cx - 110, 245, 220, 70, "Transcript", ["node"], "blue", size=13)
    n2 = b.card(cx - 110, 390, 220, 70, "Title generator", ["node"], "blue", size=13)
    n3 = b.card(cx - 110, 545, 220, 70, "Content generator", ["node"], "blue", size=13)
    en = gnode(b, cx, 690, "END", "end", w=130)
    b.text(cx + 90, 696, "output (O/P)", 13, "green", "700", anchor="start")
    b.arrow(yt.bottom(), st.top(0.5))
    b.arrow(st.bottom(), n1.top(), label="edge", color="orange")
    b.arrow(n1.bottom(), n2.top(), label="edge: transcript", color="orange")
    b.arrow(n2.bottom(), n3.top(), label="edge: title, transcript", color="orange")
    b.arrow(n3.bottom(), en.top(), label="edge", color="orange")

    impl = [
        (n1, "Node implementation", ["LangChain YT loader:", "takes the YT URL,", "returns the transcript"], "teal"),
        (n2, "LLM + prompt", ["input: transcript", "output: a blog title"], "green"),
        (n3, "LLM + prompt", ["input: title + transcript", "output: the blog content"], "green"),
    ]
    for node, t, ls, c in impl:
        card = b.card(790, node.y - 12, 290, 96, t, ls, c, size=12)
        b.arrow(card.left(), node.right(), dashed=True, color=c)
    return b


@board
def state_shared():
    b = Board(1120, 600, "State: the variables every node can read and write",
              "Redrawn from the instructor's whiteboard (2:55:00 to 2:57:30). A graph that keeps state is a StateGraph.")
    b.group(30, 90, 1060, 170, "StateGraph", "blue")
    xs = [200, 560, 920]
    names = [("Transcript", "node 1"), ("Title generator", "node 2"), ("Content generator", "node 3")]
    nodes = []
    for x, (t, s_) in zip(xs, names):
        nodes.append(b.card(x - 120, 130, 240, 74, t, [s_], "blue", size=12))
    b.arrow(nodes[0].right(), nodes[1].left(), label="edge", color="orange")
    b.arrow(nodes[1].right(), nodes[2].left(), label="edge", color="orange")

    b.group(30, 380, 1060, 150, "State  (one shared dictionary)", "yellow")
    for x, k in zip(xs, ["transcript", "title", "content"]):
        b.card(x - 100, 440, 200, 60, k, ['starts empty'], "yellow", size=12)
    # writes go down, reads come up; each node has its own lane
    b.arrow((xs[0], 204), (xs[0], 380), label="writes transcript", color="green")
    b.arrow((xs[1] - 50, 204), (xs[1] - 50, 380), label="writes title", color="green")
    b.arrow((xs[1] + 50, 380), (xs[1] + 50, 204), label="reads transcript", color="blue")
    b.arrow((xs[2] - 50, 204), (xs[2] - 50, 380), label="writes content", color="green")
    b.arrow((xs[2] + 50, 380), (xs[2] + 50, 204), label="reads title,\ntranscript", color="blue")
    b.text(560, 565, "Not the same thing as long-term memory: this state lives inside one run of the graph.",
           13, FAINT, italic=True)
    return b


@board
def chatbot_state():
    b = Board(1120, 640, "The basic chatbot: one node, and a messages list in the state",
              "Redrawn from the instructor's whiteboard (3:00:30 to 3:14:15). The reducer appends, it never replaces.")
    # left: state
    b.group(30, 100, 470, 340, "State", "orange")
    b.text(265, 160, "{ messages = [ ... ] }", 17, "orange", "700")
    slots = [("Hi, how are you?", "human"), ("I am good", "AI"), ("What is your name?", "human"),
             ("I am a basic chatbot", "AI")]
    for i, (t, who) in enumerate(slots):
        y = 185 + i * 52
        b.card(60, y, 300, 40, t, [], "yellow" if who == "human" else "green", title_size=12)
        b.pill(372, y + 6, who, "grey", size=11)
    b.text(265, 410, "each new message is appended", 13, "orange", "700")
    b.card(30, 460, 470, 150, "Reducer: add_messages", [
        "A reducer decides how a state key is updated.",
        "Without one, a node's return value replaces the key.",
        "With add_messages the new messages are added to the list."], "purple", size=12, align="left")
    # right: graph
    b.group(560, 100, 530, 510, "StateGraph", "blue")
    st = gnode(b, 825, 170, "START", "start", w=130)
    ch = b.card(715, 240, 220, 70, "chatbot", ["a node"], "blue", size=13)
    en = gnode(b, 825, 400, "END", "end", w=130)
    b.arrow(st.bottom(), ch.top(), color="orange")
    b.arrow(ch.bottom(), en.top(), color="orange")
    llm = b.card(960, 240, 120, 70, "LLM + prompt", ["takes input,", "gives output"], "green", size=11)
    b.arrow(ch.right(), llm.left(), dashed=True, color="green")
    b.text(825, 460, "input in, state['messages'] goes to the LLM,", 12, INK)
    b.text(825, 478, "the reply is appended to the list", 12, INK)
    caption_pill(b, 600, 530, "next: Tools", "orange", solid=False)
    caption_pill(b, 740, 530, "later: ReAct agent", "purple", solid=False)
    caption_pill(b, 920, 530, "Reducers", "pink", solid=False)
    return b


@board
def reducer_explained():
    b = Board(1120, 560, "Replace versus append: what a reducer changes",
              "Explanatory board (not shown in the video)")
    b.group(30, 90, 520, 430, "No reducer: the key is overwritten", "red")
    b.card(60, 140, 460, 56, "after turn 1", ["messages = [human: Hi, AI: Hello]"], "grey", size=12)
    b.arrow((290, 196), (290, 250), label="node returns [AI: I am a bot]", color="red")
    b.card(60, 250, 460, 56, "after turn 2", ["messages = [AI: I am a bot]"], "red", size=12)
    b.text(290, 345, "The earlier turns are gone, so the chatbot", 13, "red", "700")
    b.text(290, 365, "cannot see what was said before.", 13, "red", "700")
    cross(b, 500, 285, 10)
    b.group(570, 90, 520, 430, "add_messages: the list grows", "green")
    b.card(600, 140, 460, 56, "after turn 1", ["messages = [human: Hi, AI: Hello]"], "grey", size=12)
    b.arrow((830, 196), (830, 250), label="node returns [AI: I am a bot]", color="green")
    b.card(600, 250, 460, 80, "after turn 2", ["messages = [human: Hi, AI: Hello,", "           AI: I am a bot]"], "green", size=12)
    b.text(830, 372, "Everything said so far stays in the list, so", 13, "green", "700")
    b.text(830, 392, "the next LLM call sees the whole conversation.", 13, "green", "700")
    b.text(830, 450, "In code:  messages: Annotated[list, add_messages]", 13, INK)
    b.text(830, 474, "(a message that reuses an existing id replaces that one)", 12, FAINT, italic=True)
    tick(b, 1040, 290)
    return b


@board
def tool_need():
    b = Board(1120, 720, "Why a chatbot needs external tools",
              "Redrawn from the instructor's whiteboard (3:24:15 to 3:28:45).")
    q = b.card(60, 100, 330, 44, "Provide me the recent AI news", [], "red", title_size=13)
    st = gnode(b, 225, 190, "START", "start", w=130)
    ch = b.card(130, 240, 190, 70, "Chatbot", ["LLM + prompt"], "dark", size=12)
    b.arrow(q.bottom(), st.top(), color="red")
    b.arrow(st.bottom(), ch.top())
    caption_pill(b, 345, 255, "No", "red", solid=True)
    b.text(345, 295, "no live data,", 12, "red", anchor="start")
    b.text(345, 312, "may be stale", 12, "red", anchor="start")
    tn = b.card(110, 440, 230, 70, "ToolNode", ["makes the tool call"], "blue", size=12)
    b.arrow(ch.bottom(0.5), tn.top(0.5), label="tool call", color="orange")
    tools = b.card(20, 540, 400, 66, "Tools it can hold", ["Tavily, add, subtract, custom ..."], "orange", size=12)
    b.arrow(tn.bottom(), tools.top(), dashed=True, color="orange")
    en = gnode(b, 225, 665, "END", "end", w=130)
    b.arrow(tools.bottom(0.5), en.top(), color="orange", label="output")

    b.group(560, 100, 540, 590, "External tool", "orange")
    c1 = b.card(600, 160, 460, 70, "LLM + binded tools", ["the LLM is told which tools it owns"], "purple", size=12)
    f = b.card(600, 270, 460, 150, "A custom tool is a function", [
        "def add(a, b):",
        '    """Doc string:',
        "    what it does, and its inputs/args",
        '    """',
    ], "yellow", size=13, align="left")
    b.arrow(c1.bottom(), f.top(), color="purple")
    c2 = b.card(600, 460, 460, 120, "How the LLM picks a tool", [
        "It reads each tool's doc string. If the user's",
        "request matches a tool's inputs and purpose,",
        "the LLM emits a tool call for it."], "green", size=12, align="left")
    b.arrow(f.bottom(), c2.top(), color="orange", label="doc string is what it reads")
    return b


@board
def tool_calling_graph():
    b = Board(1120, 760, "The tool-calling graph and what each part does",
              "Redrawn from the graph image and annotations at 3:38:00 to 3:45:00.")
    cx = 330
    s = gnode(b, cx, 130, "__start__", "start", w=150)
    t = gnode(b, cx, 250, "tool_calling_llm", "node", w=210)
    tl = gnode(b, cx + 90, 400, "tools", "tool", w=120)
    e = gnode(b, cx, 560, "__end__", "end", w=150)
    b.arrow(s.bottom(), t.top(), label="input", color="orange")
    b.arrow(t.bottom(0.8), tl.top(0.5), label="tool call", color="orange")
    b.arrow(t.left(), e.left(), via=[(cx - 190, 250), (cx - 190, 560)], dashed=True,
            label="no tool call", color="grey", label_at=0.5)
    b.arrow(tl.bottom(), e.right(), via=[(cx + 90, 560)], color="orange", label="output", label_at=0.4)
    # right side annotations
    n1 = b.card(600, 215, 480, 70, "Node:  LLM + tools", ["the LLM has the tools bound to it"], "blue", size=12)
    b.arrow(n1.left(), t.right(), dashed=True, color="blue")
    n2 = b.card(600, 340, 230, 120, "ToolNode", ["Tavily search", "custom tools", "(multiply ...)"], "orange", size=12)
    b.arrow(n2.left(), tl.right(), dashed=True, color="orange")
    n3 = b.card(850, 340, 230, 120, "LLM  { doc string }", ["each tool carries a", "doc string the LLM", "reads"], "yellow", size=12)
    b.arrow(n3.left(), n2.right(), color="yellow")
    b.card(600, 500, 480, 200, "Two ideas to keep apart", [
        "1  Binding: LLM + tools. The LLM learns which",
        "   tools exist (like weapons it can reach for).",
        "2  Tool node: when the LLM does call a tool,",
        "   this node actually runs it and returns a",
        "   ToolMessage."], "green", size=12, align="left")
    return b


@board
def react_brain():
    b = Board(1120, 600, "The ReAct agent: the LLM as the brain, the tool node as its hands",
              "Redrawn from the instructor's whiteboard (3:53:15 to 4:00:30).")
    inp = b.card(40, 210, 220, 100, "Natural input", ["AI news,", "and multiply 5 by 5"], "red", size=13)
    brain = b.card(420, 190, 250, 140, "LLM  [ BRAIN ]", ["decides which tool,", "and when to stop"], "purple", size=13)
    b.card(430, 80, 230, 44, "binding tools", [], "yellow", title_size=13)
    b.arrow((545, 124), (545, 190), color="yellow")
    tn = b.card(800, 390, 220, 70, "ToolNode", ["Tavily, multiply ..."], "blue", size=12)
    en = gnode(b, 930, 218, "END", "end", w=130)
    b.arrow(inp.right(), brain.left(), label="input", color="red")
    b.arrow(brain.bottom(0.85), tn.left(), via=[(brain.x + brain.w * 0.85, 425)], label="act: tool call", color="orange")
    b.arrow(tn.top(), brain.right(0.8), via=[(910, 290)], label="observe: tool output", color="green", curve=False, label_dx=-40, label_dy=-18)
    b.arrow(brain.right(0.2), en.left(), label="nothing left: final answer", color="purple", label_dy=-14)
    b.card(40, 400, 330, 150, "Reason, act, observe", [
        "Reason: the LLM works out what is",
        "still unanswered.",
        "Act: it calls a tool.",
        "Observe: it reads the tool result and",
        "decides again."], "teal", size=12, align="left")
    b.text(660, 510, "loop until the LLM has no tool call left", 13, "purple", "700", anchor="middle")
    return b


@board
def react_graph():
    b = Board(1120, 640, "From one-shot tools to a ReAct loop: one edge changes",
              "Redrawn from the two rendered graphs at 3:46:45 and 3:59:45.")
    b.group(30, 90, 500, 520, "Before: tools goes to END", "red")
    s = gnode(b, 280, 150, "__start__", "start", w=150)
    t = gnode(b, 280, 260, "tool_calling_llm", "node", w=210)
    tl = gnode(b, 370, 390, "tools", "tool", w=120)
    e = gnode(b, 280, 520, "__end__", "end", w=150)
    b.arrow(s.bottom(), t.top())
    b.arrow(t.bottom(0.8), tl.top(), color="orange")
    b.arrow(t.left(), e.left(), via=[(120, 260), (120, 520)], dashed=True, color="grey")
    b.arrow(tl.bottom(), e.right(), via=[(370, 520)], color="red", label="add_edge('tools', END)", label_at=0.2, label_dy=22)
    b.text(280, 580, "second question in the sentence is never answered", 12, "red", "700")

    b.group(590, 90, 500, 520, "After: tools goes back to the LLM", "green")
    s2 = gnode(b, 840, 150, "__start__", "start", w=150)
    t2 = gnode(b, 840, 260, "tool_calling_llm", "node", w=210)
    tl2 = gnode(b, 930, 390, "tools", "tool", w=120)
    e2 = gnode(b, 840, 520, "__end__", "end", w=150)
    b.arrow(s2.bottom(), t2.top())
    b.arrow(t2.bottom(0.8), tl2.top(), color="orange")
    b.arrow(tl2.right(), t2.right(), via=[(1030, 390), (1030, 260)], color="green",
            label="add_edge('tools',\n'tool_calling_llm')", label_at=0.5, label_dx=8)
    b.arrow(t2.left(), e2.left(), via=[(680, 260), (680, 520)], dashed=True, color="grey", label="no tool call", label_at=0.5)
    b.text(840, 580, "the LLM keeps deciding until no tool call remains", 12, "green", "700")
    return b


@board
def memory_threads():
    b = Board(1120, 600, "Memory: a checkpointer saves state per thread_id",
              "Explanatory board (not shown in the video)")
    b.group(30, 90, 400, 470, "Your code", "blue")
    c1 = b.card(60, 140, 340, 90, "graph = builder.compile(", ["    checkpointer=memory)"], "blue", size=13, align="left")
    c2 = b.card(60, 260, 340, 110, "config = {", ['  "configurable": {', '    "thread_id": "1"}}'], "yellow", size=13, align="left")
    c3 = b.card(60, 400, 340, 120, "graph.invoke(", ['  {"messages": "Hi, I am Krish"},', "  config=config)"], "green", size=13, align="left")
    b.arrow(c1.bottom(), c2.top(), color="grey")
    b.arrow(c2.bottom(), c3.top(), color="grey")

    st = b.group(520, 90, 570, 470, "MemorySaver (in RAM)", "purple")
    t1 = b.card(550, 150, 510, 150, "thread_id = \"1\"", [
        "human: Hi, my name is Krish",
        "AI: Nice to meet you, Krish",
        "human: what is my name?",
        "AI: Your name is Krish"], "purple", size=12, align="left")
    t2 = b.card(550, 340, 510, 100, "thread_id = \"2\"", ["(a different user or chat)", "starts with an empty list"], "grey", size=12, align="left")
    b.arrow(c3.right(), t1.left(), label="load, run, save", color="green", via=[(470, 460), (470, 225)], label_at=0.5)
    b.text(805, 485, "Same thread_id: the graph sees the earlier turns.", 13, "green", "700")
    b.text(805, 508, "New thread_id: a fresh conversation.", 13, "grey", "700")
    b.text(805, 535, "MemorySaver is for tests and demos; use a database saver in production.", 11, FAINT, italic=True)
    return b


@board
def streaming_modes():
    b = Board(1120, 700, "Streaming modes: updates versus values",
              "Redrawn from the instructor's whiteboard (4:11:15 to 4:18:45).")
    b.text(150, 118, "Graph", 15, "red", "700")
    g = b.group(60, 135, 180, 380, "", "red")
    n3 = b.card(90, 160, 120, 60, "Node 3", [], "grey", title_size=14)
    n2 = b.card(90, 280, 120, 60, "Node 2", [], "grey", title_size=14)
    n1 = b.card(90, 400, 120, 60, "Node 1", [], "grey", title_size=14)
    b.arrow(n1.top(), n2.bottom(), color="orange")
    b.arrow(n2.top(), n3.bottom(), color="orange")
    b.arrow((150, 560), n1.bottom(), color="orange", label="input")
    b.text(150, 600, "stream() and astream()", 12, FAINT)
    cols = [(300, "state change this node writes"), (560, "stream_mode = \"updates\""), (850, "stream_mode = \"values\"")]
    for x, t in cols:
        b.text(x + 110, 125, t, 13, "purple", "700")
    rows = [
        (n3, "messages = [Krish]", "{msg: [Krish]}", "{msg: [Hi, My name is, Krish]}"),
        (n2, "messages = [My name is]", "{msg: [My name is]}", "{msg: [Hi, My name is]}"),
        (n1, "messages = [Hi]", "{msg: [Hi]}", "{msg: [Hi]}"),
    ]
    for node, a, u, v in rows:
        y = node.y - 8
        b.card(300, y, 230, 76, a, [], "yellow", title_size=12)
        b.card(560, y, 250, 76, u, ["only the newest change"], "blue", title_size=12, size=11)
        b.card(850, y, 230, 76, v, ["the whole state so far"], "green", title_size=12, size=11)
        b.arrow(node.right(), (300, node.cy), color="grey", dashed=True)
    b.arrow((530, rows[0][0].cy), (560, rows[0][0].cy), color="grey")
    b.arrow((530, rows[1][0].cy), (560, rows[1][0].cy), color="grey")
    b.arrow((530, rows[2][0].cy), (560, rows[2][0].cy), color="grey")
    b.arrow((810, rows[0][0].cy), (850, rows[0][0].cy), color="grey")
    b.arrow((810, rows[1][0].cy), (850, rows[1][0].cy), color="grey")
    b.arrow((810, rows[2][0].cy), (850, rows[2][0].cy), color="grey")
    b.card(300, 540, 780, 110, "Third way: astream_events", [
        "Streams every internal event (chain start, model start, each token chunk, tool events).",
        "Use it when you need very detailed output for debugging or a live typing effect."],
        "orange", size=12, align="left")
    return b


@board
def human_loop():
    b = Board(1120, 660, "Human feedback in the loop",
              "Redrawn from the instructor's whiteboard (4:20:00 to 4:22:00).")
    b.group(280, 90, 420, 540, "", "grey")
    b.text(530, 112, "I/P", 13, "yellow", "700")
    st = gnode(b, 530, 160, "START", "start", w=130)
    ch = b.card(420, 215, 220, 70, "Chatbot", ["LLM + tools"], "dark", size=12)
    tn = b.card(320, 380, 190, 70, "ToolNode", ["Tavily", "human assistance"], "blue", size=11)
    en = gnode(b, 570, 560, "END", "end", w=130)
    b.arrow(st.bottom(), ch.top(), color="yellow")
    b.arrow(ch.bottom(0.25), tn.top(0.4), color="yellow", label="tool call")
    b.arrow(tn.top(0.8), ch.bottom(0.55), color="yellow", label="feedback", label_dx=26)
    b.arrow(ch.bottom(0.85), en.top(), via=[(607, 480)], color="yellow")
    b.card(30, 330, 220, 70, "Tavily", ["web search tool"], "red", size=12)
    b.card(30, 430, 220, 70, "Human assistance", ["a custom tool"], "orange", size=12)
    b.arrow((250, 365), tn.left(0.3), color="red", dashed=True)
    b.arrow((250, 465), tn.left(0.7), color="orange", dashed=True)
    b.card(750, 215, 340, 110, "LLM + tools", [
        "The LLM decides: search the web,",
        "or ask a human."], "yellow", size=12)
    b.arrow(ch.right(), (750, 270), color="yellow", dashed=True)
    b.card(750, 370, 340, 250, "Complex workflow", [
        "Some steps must not run unless a",
        "person approves.",
        "",
        "step A runs",
        "   |",
        "INTERRUPT  (graph pauses here)",
        "   |",
        "human says yes / gives feedback",
        "   |",
        "step B continues"], "teal", size=12, align="left")
    return b


@board
def interrupt_resume():
    b = Board(1120, 700, "Interrupt and resume, step by step",
              "Explanatory board (not shown in the video)")
    lanes = [(40, "Your script", "blue"), (400, "The graph", "purple"), (780, "The human", "green")]
    for x, t, c in lanes:
        b.group(x, 90, 300, 580, t, c)
    steps = [
        (130, 0, 1, "graph.stream(user input, config, stream_mode='values')", "blue"),
        (230, 1, 1, "chatbot node: the LLM emits a tool call for human_assistance", "purple"),
        (330, 1, 1, "tool runs interrupt({'query': ...}); graph PAUSES, state is checkpointed", "purple"),
        (430, 2, 2, "reads the query, writes an answer", "green"),
        (530, 0, 1, "graph.stream(Command(resume={'data': answer}), config)", "blue"),
        (610, 1, 1, "interrupt() returns the answer; tool message; LLM replies", "purple"),
    ]
    boxes = []
    for y, lane, _, t, c in steps:
        x = lanes[lane][0] + 15
        boxes.append(b.card(x, y, 270, 70, t, [], c, size=12, title_size=12))
    b.arrow(boxes[0].right(), boxes[1].left(), color="grey")
    b.arrow(boxes[1].bottom(), boxes[2].top(), color="purple")
    b.arrow(boxes[2].right(), boxes[3].left(), color="green", label="query")
    b.arrow(boxes[3].left(0.8), boxes[4].right(), color="green", label="answer", via=[(700, 485), (700, 585)])
    b.arrow(boxes[4].right(0.2), boxes[5].left(), color="blue", via=[(350, 545), (350, 645)])
    return b


@board
def mcp_architecture():
    b = Board(1120, 600, "MCP servers, MCP clients and the app",
              "Redrawn from the instructor's slide at 4:27:30 (Building MCP server, MCP integration).")
    b.text(210, 118, "MCP Servers", 20, INK, "700")
    b.text(210, 142, "provide context, tools and prompts", 12, FAINT)
    b.text(210, 158, "to clients", 12, FAINT)
    b.text(560, 118, "MCP Clients", 20, INK, "700")
    b.text(560, 142, "keep a 1:1 connection with a server,", 12, FAINT)
    b.text(560, 158, "inside the host app", 12, FAINT)
    b.text(930, 118, "App", 20, INK, "700")

    b.group(80, 190, 260, 320, "", "grey")
    math = b.card(110, 215, 200, 100, "Math server", ["add()", "divide() ..."], "blue", size=12)
    data = b.card(110, 360, 200, 100, "Data server", ["..."], "teal", size=12)
    b.text(210, 495, "each server can have many tools", 12, FAINT, italic=True)

    cd = b.card(440, 215, 240, 90, "Claude desktop", ["app client"], "purple", size=12)
    py = b.card(440, 380, 240, 90, "Py client", ["Python MCP client"], "purple", size=12)
    app = b.card(810, 215, 240, 90, "Claude desktop app", ["a host app"], "orange", size=12)
    ag = b.card(810, 380, 240, 90, "LangGraph agent", ["your own app"], "red", size=12)
    b.arrow(math.right(0.4), cd.left(0.5), both=True, color="grey")
    b.arrow(data.right(0.5), py.left(0.5), both=True, color="grey")
    b.arrow(cd.right(), app.left(), both=True, color="grey")
    b.arrow(py.right(), ag.left(), both=True, color="red", label="load_mcp_tools", label_dy=-16)
    b.card(80, 535, 960, 44, "Server and client talk the MCP protocol; the agent only sees ordinary LangChain tools.", [], "yellow", title_size=12)
    return b


@board
def mcp_app_board():
    b = Board(1120, 640, "The MCP demo we build: one chatbot app, MCP server(s), two transports",
              "Redrawn from the instructor's whiteboard (4:28:45 to 4:41:15).")
    inp = b.text(70, 330, "I/P", 17, "red", "700")
    app = b.card(130, 220, 250, 230, "Application", ["CHATBOT", "", "LLM", "+ MCP client"], "pink", size=15, title_size=17)
    b.arrow((85, 335), app.left(0.5), color="red")
    srv = b.card(640, 240, 190, 100, "MCP Server", ["FastMCP"], "red", size=13)
    b.arrow(app.right(0.3), srv.left(0.3), color="grey", label="MCP protocol", label_dy=-16)
    b.arrow(srv.left(0.75), app.right(0.75), color="grey")
    t1 = b.card(900, 220, 190, 50, "Add, multiplication", [], "yellow", title_size=12)
    t2 = b.card(900, 300, 190, 60, "Weather call API", [], "yellow", title_size=12)
    b.arrow(srv.right(0.3), t1.left(), color="yellow")
    b.arrow(srv.right(0.7), t2.left(), color="yellow")
    b.card(620, 400, 470, 120, "Two parts to build", [
        "1  MCP server  (with FastMCP)",
        "2  MCP client  (langchain-mcp-adapters)"], "purple", size=13, align="left")
    b.card(620, 90, 470, 100, "Transport = how client and server talk", [
        "stdio  (standard input and output)",
        "http   (streamable HTTP)"], "orange", size=13, align="left")
    b.text(255, 520, "A user's question reaches the LLM; if it needs a tool,", 12, FAINT)
    b.text(255, 540, "the call goes out through the MCP client.", 12, FAINT)
    return b


@board
def mcp_transports():
    b = Board(1120, 640, "The two transports in the demo, side by side",
              "Explanatory board (not shown in the video)")
    b.group(30, 90, 520, 520, "stdio  (mathserver.py)", "blue")
    c = b.card(60, 150, 200, 80, "client.py", ["MultiServerMCPClient"], "purple", size=12)
    p = b.card(60, 330, 200, 80, "child process", ["python mathserver.py"], "blue", size=12)
    b.arrow(c.bottom(0.3), p.top(0.3), color="blue", label="spawns it", label_dx=-4)
    b.arrow(c.bottom(0.7), p.top(0.7), color="blue", label="stdin / stdout", both=True, label_dx=10)
    b.card(300, 150, 230, 120, "Why use it", ["Runs on your machine.", "No port, no URL.", "Best for local testing."], "green", size=12, align="left")
    b.card(60, 450, 470, 130, "In the client config", [
        '"math": {"command": "python",',
        '         "args": ["mathserver.py"],',
        '         "transport": "stdio"}'], "yellow", size=12, align="left")
    b.group(570, 90, 520, 520, "streamable HTTP  (weather.py)", "orange")
    c2 = b.card(600, 150, 200, 80, "client.py", ["MultiServerMCPClient"], "purple", size=12)
    s2 = b.card(600, 330, 200, 80, "web server", ["python weather.py", "localhost:8000"], "orange", size=12)
    b.arrow(c2.bottom(), s2.top(), color="orange", both=True, label="HTTP requests", label_dx=0)
    b.card(830, 150, 230, 120, "Why use it", ["Runs as an API service.", "Reachable by URL,", "can live on another host."], "green", size=12, align="left")
    b.card(600, 450, 470, 130, "In the client config", [
        '"weather": {"url": "http://localhost:8000/mcp",',
        '            "transport": "streamable_http"}'], "yellow", size=12, align="left")
    return b


@board
def rendered_graphs():
    b = Board(1120, 560, "The graphs the notebook draws for the other examples",
              "Redrawn from the rendered Mermaid images at 3:17:00, 4:10:00 and 4:24:00.")
    # 1 llmchatbot
    cx = 190
    b.group(30, 80, 320, 440, "basic chatbot (3:17)", "blue")
    s = gnode(b, cx, 160, "__start__", "start", w=140)
    n = gnode(b, cx, 280, "llmchatbot", "node", w=170)
    e = gnode(b, cx, 400, "__end__", "end", w=140)
    b.arrow(s.bottom(), n.top(), color="orange")
    b.arrow(n.bottom(), e.top(), color="orange")
    b.text(cx, 470, "START to one node to END", 12, FAINT)
    # 2 SuperBot
    cx = 560
    b.group(380, 80, 320, 440, "streaming demo (4:10)", "purple")
    s = gnode(b, cx, 160, "__start__", "start", w=140)
    n = gnode(b, cx, 280, "SuperBot", "node", w=170)
    e = gnode(b, cx, 400, "__end__", "end", w=140)
    b.arrow(s.bottom(), n.top(), color="orange")
    b.arrow(n.bottom(), e.top(), color="orange")
    b.text(cx, 470, "same shape, plus a checkpointer", 12, FAINT)
    # 3 hitl
    cx = 920
    b.group(730, 80, 360, 440, "human in the loop (4:24)", "green")
    s = gnode(b, cx - 20, 150, "__start__", "start", w=140)
    n = gnode(b, cx - 20, 250, "chatbot", "node", w=150)
    t = gnode(b, cx + 60, 360, "tools", "tool", w=110)
    e = gnode(b, cx - 40, 450, "__end__", "end", w=130)
    b.arrow(s.bottom(), n.top(), color="orange")
    b.arrow(n.bottom(0.8), t.top(0.4), color="orange")
    b.arrow(t.right(), n.right(), via=[(cx + 150, 360), (cx + 150, 250)], color="orange")
    b.arrow(n.left(), e.left(), via=[(cx - 125, 250), (cx - 125, 450)], dashed=True, color="grey")
    return b


@board
def studio_debug():
    b = Board(1120, 600, "Debugging in LangGraph Studio: the three files and one command",
              "Explanatory board (not shown in the video). From the 3-Debugging folder of the course repo.")
    b.group(30, 90, 400, 490, "3-Debugging folder", "blue")
    b.card(55, 140, 350, 150, "agent.py", [
        "State, tools, llm",
        "def make_tool_graph(): ... build",
        "tool_agent = make_tool_graph()",
        "(a compiled graph in a module variable)"], "blue", size=12, align="left")
    b.card(55, 310, 350, 140, "langgraph.json", [
        '"dependencies": ["."]',
        '"graphs": {"tool_agent":',
        '           "./agent.py:tool_agent"}',
        '"env": "../.env"'], "yellow", size=12, align="left")
    b.card(55, 470, 350, 90, ".env  (never committed)", ["GROQ_API_KEY", "LANGCHAIN_API_KEY"], "red", size=12, align="left")
    cmd = b.card(480, 250, 190, 110, "langgraph dev", ["run in the folder", "that holds", "langgraph.json"], "dark", size=12)
    b.arrow((405, 215), cmd.left(0.25), color="blue")
    b.arrow((405, 380), cmd.left(0.6), color="orange")
    b.arrow((405, 515), cmd.left(0.9), color="red", via=[(450, 515), (450, 348)])
    srv = b.card(730, 110, 330, 110, "Local dev server", ["loads graph 'tool_agent'", "from agent.py"], "purple", size=12)
    b.arrow(cmd.right(0.2), srv.left(0.7), color="purple")
    st = b.card(730, 270, 330, 160, "LangGraph Studio (browser)", [
        "see the graph drawn",
        "type an input, run it",
        "watch each node execute",
        "inspect state after every step"], "green", size=12, align="left")
    b.arrow(cmd.right(0.5), st.left(0.5), color="green")
    ls = b.card(730, 470, 330, 100, "LangSmith traces", ["project 'TestProject'", "LANGSMITH_TRACING=true"], "orange", size=12)
    b.arrow(cmd.right(0.85), ls.left(0.5), color="orange", via=[(700, 345), (700, 520)])
    return b


@board
def multi_agent():
    b = Board(1120, 760, "Two multi-agent shapes from the repo notebook",
              "Explanatory board (not shown in the video). From Agents/multiaiagent.ipynb.")
    b.group(30, 90, 480, 650, "1  Simple pipeline", "blue")
    st = gnode(b, 270, 160, "START", "start", w=120)
    r = b.card(155, 215, 230, 90, "researcher", ["LLM + search_web tool", "next_agent = writer"], "blue", size=12)
    w = b.card(155, 360, 230, 90, "writer", ["LLM, no tools", "writes the summary"], "green", size=12)
    en = gnode(b, 270, 510, "END", "end", w=120)
    b.arrow(st.bottom(), r.top(), color="orange")
    b.arrow(r.bottom(), w.top(), color="orange")
    b.arrow(w.bottom(), en.top(), color="orange")
    b.card(55, 560, 430, 150, "Shared state", [
        "AgentState(MessagesState)",
        "   messages  (added by the base class)",
        "   next_agent: str",
        "Each agent returns its message and",
        "who goes next."], "yellow", size=12, align="left")

    b.group(540, 90, 550, 650, "2  Supervisor", "purple")
    sup = b.card(650, 215, 200, 80, "supervisor", ["LLM picks next agent"], "purple", size=12)
    st2 = gnode(b, 750, 160, "START", "start", w=120)
    b.arrow(st2.bottom(), sup.top(), color="orange")
    rs = b.card(570, 400, 150, 70, "researcher", ["research_data"], "blue", size=12)
    an = b.card(745, 400, 150, 70, "analyst", ["analysis"], "teal", size=12)
    wr = b.card(920, 400, 150, 70, "writer", ["final_report"], "green", size=12)
    b.arrow(sup.bottom(0.2), rs.top(), color="purple")
    b.arrow(sup.bottom(0.5), an.top(), color="purple")
    b.arrow(sup.bottom(0.8), wr.top(), color="purple")
    en2 = gnode(b, 1000, 255, "END", "end", w=110)
    b.arrow(sup.right(), en2.left(), color="grey", dashed=True, label="done")
    b.text(815, 520, "every agent hands control back to the supervisor", 12, "purple", "700")
    b.text(815, 540, "(conditional edges through a router function)", 12, FAINT, italic=True)
    b.card(570, 570, 500, 150, "Shared state", [
        "SupervisorState(MessagesState)",
        "   next_agent, research_data, analysis,",
        "   final_report, task_complete, current_task",
        "The router reads next_agent and returns the",
        "node to run next, or END."], "yellow", size=12, align="left")
    return b


@board
def multimodal_rag():
    b = Board(1120, 640, "Multimodal RAG over a PDF with text and images",
              "Explanatory board (not shown in the video). From 4-Multimodal/1-multimodalopenai.ipynb.")
    pdf = b.card(30, 110, 160, 110, "PDF", ["text and charts"], "grey", size=12)
    tx = b.card(260, 100, 220, 80, "Text chunks", ["500 chars, overlap 100"], "blue", size=12)
    im = b.card(260, 210, 220, 80, "Images", ["pulled out with PyMuPDF"], "orange", size=12)
    b.arrow(pdf.right(0.3), tx.left(), color="blue")
    b.arrow(pdf.right(0.7), im.left(), color="orange")
    clip = b.card(550, 140, 220, 110, "CLIP", ["one model, one space:", "text and images both", "become 512-d vectors"], "purple", size=12)
    b.arrow(tx.right(), clip.left(0.3), color="blue")
    b.arrow(im.right(), clip.left(0.7), color="orange")
    fa = b.cylinder(840, 120, 200, 130, "FAISS", ["vectors + metadata", "(type: text / image)"], "teal")
    b.arrow(clip.right(), (840, 185), color="purple")
    q = b.card(30, 400, 200, 70, "Question", ["embedded with CLIP"], "red", size=12)
    ret = b.card(300, 390, 230, 90, "Top-k search", ["similarity search by", "the query vector"], "teal", size=12)
    b.arrow(q.right(), ret.left(), color="red")
    b.arrow(fa.bottom(), ret.top(), via=[(940, 330), (415, 330)], color="teal")
    msg = b.card(600, 380, 240, 110, "Build one message", ["question + text excerpts", "+ images as base64"], "yellow", size=12)
    b.arrow(ret.right(), msg.left(), color="teal")
    llm = b.card(900, 390, 190, 90, "Vision LLM", ["gpt-4.1"], "dark", size=12)
    b.arrow(msg.right(), llm.left(), color="grey")
    ans = b.card(900, 530, 190, 60, "Answer", [], "green", title_size=13)
    b.arrow(llm.bottom(), ans.top(), color="green")
    b.text(300, 560, "Images are stored as base64 so the vision model can look at the chart itself.", 12, FAINT, italic=True, anchor="start")
    return b


if __name__ == "__main__":
    main(sys.argv[1:])
