"""Infographics for docs/projects/agentic-ai-complete-course, chapter 02
(LangChain messages, structured output and middleware, 1:06:40 to 2:35:12).

Boards the instructor actually drew or showed (redrawn here as original images):
    airport_security        1:53:45 to 1:58:00   Excalidraw scribble
    agent_hooks             1:57:00 to 1:58:00   LangChain docs diagram, annotated "hooks"
    builtin_summarization   1:58:30 to 2:00:30   Excalidraw scribble
    hitl_scribble           2:20:40 to 2:22:40   Excalidraw scribble

Explanatory boards added for this chapter (he explains these aloud or in code only):
    msg_types, msg_tool_roundtrip, schema_routes, nested_schema,
    summarization_runs, hitl_flow

Run from the repo root:
    python3 scripts/infographics/agentic_course_02.py                 # all boards
    python3 scripts/infographics/agentic_course_02.py airport_security
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


def circle_num(b, cx, cy, n, color="orange", r=13):
    c = PALETTE[color]
    b.parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="2"/>'
    )
    b.parts.append(
        f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-family="{MONO}" font-size="14" '
        f'font-weight="700" fill="{c["text"]}">{n}</text>'
    )


def mono(b, x, y, text, size=12, color=INK, weight="400", anchor="start"):
    fill = PALETTE[color]["text"] if color in PALETTE else color
    b.parts.append(
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{MONO}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" xml:space="preserve">{esc(text)}</text>'
    )


def wide_card(b, x, y, w, h, title, color, size=14):
    """A one-line card with a bold title centred in it."""
    return b.card(x, y, w, h, title, [], color, title_size=size)


# ----------------------------------------------------------- explanatory: messages


@board
def msg_types():
    b = Board(1100, 700, "Messages: the unit of context",
              "Explanatory board (not shown in the video): what a message is, the four kinds, two ways to call a model")

    b.group(20, 90, 1060, 150, "Every message carries three things", "grey")
    b.card(45, 130, 320, 90, "role", ["which kind of message it is", "system, human, ai, tool"], "blue")
    b.card(390, 130, 320, 90, "content", ["the payload itself", "text, images, audio, documents"], "green")
    b.card(735, 130, 320, 90, "metadata", ["optional extras: name, id,", "token usage, response info"], "purple")

    b.group(20, 260, 1060, 215, "The four message types", "blue")
    sysm = b.card(40, 300, 245, 150, "SystemMessage",
                  ["sets the model's role", "and ground rules", "", "SystemMessage(", "  \"You are a poetry", "  expert\")"],
                  "blue", size=12, align="center")
    hum = b.card(305, 300, 245, 150, "HumanMessage",
                 ["what the user says", "", "HumanMessage(", "  \"Write a poem on", "  AI\")"],
                 "green", size=12)
    ai = b.card(570, 300, 245, 150, "AIMessage",
                ["what the model answers:", "text, tool_calls,", "usage_metadata", "", "model.invoke(...)", "returns one"],
                "purple", size=12)
    tool = b.card(835, 300, 225, 150, "ToolMessage",
                  ["output of ONE tool", "call, sent back to", "the model", "", "needs tool_call_id"],
                  "orange", size=12)

    b.group(20, 495, 1060, 185, "Two ways to call model.invoke", "teal")
    t = b.card(45, 540, 400, 110, "Text prompt",
               ["model.invoke(\"what is langchain\")", "one string, treated as a single", "HumanMessage, no history"],
               "teal", size=12)
    m = b.card(520, 540, 400, 110, "Message prompt",
               ["model.invoke([system, human, ai, ...])", "a list = a conversation history,", "you control every role"],
               "teal", size=12)
    out = b.card(960, 560, 100, 70, "AIMessage", ["comes back"], "purple", size=11, title_size=12)
    b.arrow(t.bottom(0.5), out.bottom(0.5), via=[(t.cx, 664), (out.cx, 664)], color="teal")
    b.arrow(m.right(), out.left(0.5), color="teal")
    return b


@board
def msg_tool_roundtrip():
    b = Board(1100, 600, "A tool call, written out as messages",
              "Explanatory board (not shown in the video): the exact list he builds by hand for get_weather")

    b.group(20, 90, 770, 370, "messages = [ ... ]  (a list of message objects, in order)", "blue")
    h = b.card(40, 140, 220, 190, "1. HumanMessage",
               ["\"What's the weather", "in San Francisco?\""], "green", size=12)
    a = b.card(285, 140, 250, 190, "2. AIMessage",
               ["content=[]", "tool_calls=[{", "  name: get_weather", "  args: {location:", "     San Francisco}", "  id: call_123", "}]"],
               "purple", size=12)
    t = b.card(560, 140, 210, 190, "3. ToolMessage",
               ["content=", "\"Sunny, 72F\"", "", "tool_call_id=", "\"call_123\""],
               "orange", size=12)
    b.arrow(h.right(), a.left(), color="grey")
    b.arrow(a.right(), t.left(), color="grey")
    b.text(380, 358, "the model asked for the tool", 12, "purple", "700")
    b.text(665, 365, "your code ran it, then wrapped", 12, "orange", "700")
    b.text(665, 383, "the result in a ToolMessage", 12, "orange", "700")

    # id matching
    line(b, [(a.x + a.w * 0.8, 330), (a.x + a.w * 0.8, 425)], "red", 1.8, dashed=True)
    line(b, [(a.x + a.w * 0.8, 425), (t.x + t.w * 0.5, 425)], "red", 1.8, dashed=True)
    b.arrow((t.x + t.w * 0.5, 425), t.bottom(), color="red", dashed=True)
    b.text(415, 419, "ids must match", 12, "red", "700")

    inv = b.card(840, 160, 220, 70, "model.invoke(messages)", [], "dark", title_size=12)
    fin = b.card(840, 290, 220, 110, "AIMessage", ["final answer, for example:", "\"Sunny, 72F in SF\""],
                 "purple", size=12)
    b.arrow((790, 255), inv.bottom(0.2), via=[(815, 255), (815, 230)], color="grey")
    b.arrow(inv.bottom(), fin.top(), label="model reads the result", label_dy=0, color="purple")

    b.card(20, 485, 1060, 90, "Why this matters",
           ["A model never runs tools. It writes a tool_call, your code (or create_agent) runs the tool,",
            "and the result goes back as a ToolMessage so the model can phrase the final reply."],
           "yellow", size=13)
    return b


# ----------------------------------------------------- explanatory: structured output


@board
def schema_routes():
    b = Board(1160, 600, "Structured output: three ways to describe the schema",
              "Explanatory board (not shown in the video): what he compared in 5-structuredoutput.ipynb")

    q = b.card(30, 100, 250, 70, "\"Provide details about", ["the movie Inception\""], "grey", size=12, title_size=12)
    m = b.card(350, 100, 330, 70, "model.with_structured_output(Schema)", [], "dark", title_size=12)
    r = b.card(760, 100, 370, 70, "an object that matches Schema", ["not a paragraph of prose"], "green", size=12)
    b.arrow(q.right(), m.left())
    b.arrow(m.right(), r.left())

    b.table(40, 210, [190, 300, 300, 300], [
        ["", "Pydantic BaseModel", "TypedDict", "dataclass"],
        ["checks types\nat runtime?", "Yes. A wrong type\nraises a validation error", "No. It is a plain dict\n(rating came back as 8)", "No. Nothing is enforced\nby the class itself"],
        ["what you get\nback", "Movie(title='Inception',\nyear=2010, ...)", "{'director': 'Joss Whedon',\n'title': 'The Avengers', ...}", "ContactInfo(name='John\nDoe', email=..., phone=...)"],
        ["field\ndescriptions", "Field(description=...)", "Annotated[str, ...,\n\"The title of the movie\"]", "only a class docstring;\nfield comments are not sent"],
        ["nested\nstructures", "Yes: cast: list[Actor]", "Yes, but no validation\ninside", "Yes, with other\ndataclasses"],
        ["pick it when", "you need guarantees\nbefore using the data", "you only want a dict\nshape, fast", "you like plain classes\nwith attribute access"],
    ], header_color="blue", size=12)

    b.card(40, 520, 1080, 60, "Where he used each",
           ["with_structured_output on Groq Qwen: Pydantic, TypedDict.  create_agent(response_format=...) on GPT-5: all three."],
           "yellow", size=12)
    return b


@board
def nested_schema():
    b = Board(1100, 560, "Nested structure: Actor inside MovieDetails",
              "Explanatory board (not shown in the video): the Pydantic nested example, and what came back")

    md = b.card(30, 100, 400, 230, "class MovieDetails(BaseModel)",
                ["title: str", "year: int", "cast: list[Actor]", "genres: list[str]",
                 "budget: float | None", "   = Field(None, description=", "     \"Budget in millions USD\")"],
                "blue", size=13, align="left")
    ac = b.card(30, 370, 400, 140, "class Actor(BaseModel)", ["name: str", "role: str"],
                "teal", size=13, align="left")
    b.arrow(ac.top(0.5), (230, 330), via=[], color="teal", label="cast holds many", label_dx=95)

    b.group(500, 90, 570, 430, "MovieDetails returned for \"Inception\"", "green")
    b.card(525, 135, 520, 36, "title='Inception'   year=2010", [], "green", title_size=13)
    a1 = b.card(525, 190, 520, 36, "Actor(name='Leonardo DiCaprio', role='Dom Cobb')", [], "teal", title_size=12)
    a2 = b.card(525, 235, 520, 36, "Actor(name='Joseph Gordon-Levitt', role='Arthur')", [], "teal", title_size=12)
    a3 = b.card(525, 280, 520, 36, "Actor(name='Elliot Page', role='Ariadne')", [], "teal", title_size=12)
    a4 = b.card(525, 325, 520, 36, "Actor(name='Tom Hardy', role='Bane')", [], "teal", title_size=12)
    b.text(535, 384, "cast = [ the four Actor objects above ]", 12, "teal", "700", anchor="start")
    b.card(525, 400, 520, 36, "genres=['Science Fiction', 'Action', 'Heist']", [], "green", title_size=12)
    b.card(525, 450, 520, 36, "budget=160.0   (millions of USD)", [], "green", title_size=13)
    return b


# -------------------------------------------------------------- middleware boards


@board
def airport_security():
    b = Board(1180, 640, "Airport security = middleware",
              "Redrawn from the instructor's Excalidraw page at 1:53:45 to 1:58:00")

    b.group(20, 90, 1140, 290, "The passenger's journey: checks happen before you reach the gate", "yellow")
    p = b.person(85, 150, "grey", 1.0, "Passenger")
    sec = b.card(190, 170, 190, 66, "Security check", [], "orange", title_size=15)
    imm = b.card(450, 170, 190, 66, "Immigration", [], "orange", title_size=15)
    brd = b.card(710, 170, 190, 66, "Board", [], "orange", title_size=15)
    fl = b.card(985, 170, 140, 66, "Flight", [], "grey", title_size=15)
    gate = b.card(1055, 120, 70, 34, "18", [], "grey", title_size=14)
    b.arrow((120, 203), sec.left(), color="grey")
    b.arrow(sec.right(), imm.left(), color="grey")
    b.arrow(imm.right(), brd.left(), color="grey")
    b.arrow(brd.right(), fl.left(), color="grey")
    b.arrow(fl.top(0.78), gate.bottom(0.5), color="grey", width=1.4)

    m1 = b.card(170, 285, 230, 80, "Middleware 1",
                ["luggage check:", "no batteries"], "red", size=12)
    m2 = b.card(430, 285, 230, 80, "Middleware 2",
                ["passport check:", "is it still valid"], "red", size=12)
    m3 = b.card(690, 285, 230, 80, "Middleware 3",
                ["boarding pass", "is it the right flight"], "red", size=12)
    b.arrow(m1.top(), sec.bottom(), color="red")
    b.arrow(m2.top(), imm.bottom(), color="red")
    b.arrow(m3.top(), brd.bottom(), color="red")

    b.group(20, 410, 1140, 205, "The same idea around an agent", "blue")
    req = b.card(45, 475, 130, 60, "request", [], "grey", title_size=14)
    w1 = b.card(225, 475, 170, 60, "Middleware 1", ["logging"], "red", size=12)
    w2 = b.card(435, 475, 170, 60, "Middleware 2", ["checks, retries"], "red", size=12)
    w3 = b.card(645, 475, 170, 60, "Middleware 3", ["guardrails, PII"], "red", size=12)
    ag = b.card(855, 455, 150, 100, "Agent", ["model + tools"], "purple", size=12)
    res = b.card(1040, 475, 100, 60, "result", [], "green", title_size=14)
    for a, c in [(req, w1), (w1, w2), (w2, w3), (w3, ag), (ag, res)]:
        b.arrow(a.right(), c.left())
    b.text(590, 590, "each stage can inspect, change or stop the request before the agent does its job",
           12, FAINT, "400")
    return b


@board
def agent_hooks():
    b = Board(1180, 780, "Agent vs agent with middleware: the hooks",
              "Redrawn from the LangChain docs diagram he showed at 1:56:40 to 1:58:00 and labelled \"hooks\"")

    b.group(20, 90, 400, 660, "Agent", "orange")
    rq = b.card(120, 135, 200, 46, "request", [], "grey", title_size=14)
    md = b.card(120, 240, 200, 56, "model", [], "purple", title_size=15)
    tl = b.card(60, 480, 140, 56, "tools", [], "blue", title_size=15)
    rs = b.card(250, 480, 140, 56, "result", [], "green", title_size=15)
    b.arrow(rq.bottom(), md.top())
    b.arrow(md.bottom(0.15), tl.top(0.64), label="action", color="blue", label_dx=-42)
    b.arrow(tl.top(0.86), md.bottom(0.3), label="observation", dashed=True, color="blue", label_dx=48)
    b.arrow(md.bottom(0.85), rs.top(0.29), dashed=True, color="green")
    b.text(220, 640, "a plain loop: ask the model,", 12, FAINT)
    b.text(220, 658, "run tools, repeat, answer", 12, FAINT)

    b.group(460, 90, 700, 660, "Agent with middleware", "red")
    b.text(1110, 126, "hooks", 20, "red", "700")
    rq2 = b.card(660, 130, 200, 44, "request", [], "grey", title_size=14)
    ba = b.card(610, 215, 300, 50, "before_agent", [], "red", title_size=14)
    bm = b.card(610, 300, 300, 50, "before_model", [], "red", title_size=14)
    wt = b.card(500, 410, 230, 90, "wrap_tool_call", ["tools"], "blue", size=13, title_size=14)
    wm = b.card(790, 410, 230, 90, "wrap_model_call", ["model"], "purple", size=13, title_size=14)
    am = b.card(610, 560, 300, 50, "after_model", [], "red", title_size=14)
    aa = b.card(610, 640, 300, 50, "after_agent", [], "red", title_size=14)
    rs2 = b.card(660, 710, 200, 34, "result", [], "green", title_size=13)
    b.arrow(rq2.bottom(), ba.top())
    b.arrow(ba.bottom(), bm.top())
    b.arrow(bm.bottom(0.2), wt.top(), via=[(bm.x + bm.w * 0.2, 380), (wt.cx, 380)])
    b.arrow(bm.bottom(0.8), wm.top(), via=[(bm.x + bm.w * 0.8, 380), (wm.cx, 380)])
    b.arrow(wt.right(), wm.left(), both=True, dashed=True, color="grey", label="loop", label_dy=-14)
    b.arrow(wt.bottom(), am.top(0.2), via=[(wt.cx, 535), (am.x + am.w * 0.2, 535)])
    b.arrow(wm.bottom(), am.top(0.8), via=[(wm.cx, 535), (am.x + am.w * 0.8, 535)])
    b.arrow(am.bottom(), aa.top())
    b.arrow(aa.bottom(), rs2.top(), width=1.4)
    b.text(1100, 235, "run once", 11, FAINT, anchor="end")
    b.text(1100, 255, "per agent run", 11, FAINT, anchor="end")
    b.text(1100, 320, "run each", 11, FAINT, anchor="end")
    b.text(1100, 340, "model turn", 11, FAINT, anchor="end")
    return b


@board
def builtin_summarization():
    b = Board(1180, 640, "Built-in middleware: summarization first",
              "Redrawn from the instructor's Excalidraw page at 1:58:30 to 2:00:30")

    b.group(20, 90, 330, 520, "Built-in middleware", "red")
    c1 = b.card(45, 150, 280, 70, "1) Summarization", ["wraps the agent"], "red", size=12, title_size=15)
    c2 = b.card(45, 240, 280, 70, "2) Human in the loop", ["approve tool calls"], "red", size=12, title_size=15)
    c3 = b.card(45, 330, 280, 70, "3) Model call limit", ["cap model calls to cap cost"], "red", size=12, title_size=15)
    b.text(185, 445, "... and more: tool call limit,", 12, FAINT)
    b.text(185, 463, "model fallback, PII detection,", 12, FAINT)
    b.text(185, 481, "to-do list, tool selector, retries", 12, FAINT)

    b.group(390, 90, 770, 520, "Summarization middleware around an agent", "blue")
    b.text(450, 246, "I/P", 18, "grey", "700")
    ag = b.card(520, 190, 170, 100, "Agent", [], "purple", title_size=18)
    tool = b.card(760, 150, 130, 50, "Tool", [], "blue", title_size=14)
    b.text(1010, 246, "O/P", 18, "grey", "700")
    b.arrow((480, 240), ag.left())
    b.arrow(ag.right(), (985, 240))
    b.arrow(ag.top(0.8), tool.left(), via=[(ag.x + ag.w * 0.8, 175)], color="blue")
    b.arrow(tool.bottom(0.2), ag.right(0.25), via=[(tool.x + tool.w * 0.2, 215)], color="blue", dashed=True)

    msgs = b.card(430, 350, 190, 130, "{ messages }", ["human, ai,", "tool, ai, ...", "the list keeps", "growing"], "yellow", size=12)
    ten = b.card(430, 510, 190, 70, "{ 10 messages }", ["the trigger"], "orange", size=12)
    llm = b.card(740, 380, 130, 50, "LLM", ["(cheaper model)"], "dark", title_size=14, size=11)
    sm = b.card(930, 350, 190, 130, "Summarised", ["message", "", "old turns squeezed", "into one"], "green", size=12)
    b.arrow(msgs.right(), llm.left(0.5), label="count hits 10", label_dy=-18)
    b.arrow(llm.right(), sm.left(0.5))
    b.arrow(ten.top(), msgs.bottom(), color="orange", dashed=True)
    b.arrow(sm.top(), ag.bottom(0.6), via=[(sm.cx, 320), (ag.x + ag.w * 0.6, 320)],
            color="green", label="replaces the old messages", label_at=0.3, label_dy=-14)
    return b


@board
def summarization_runs():
    b = Board(1160, 700, "Summarisation in action: three triggers, three runs",
              "Explanatory board (not shown in the video): message counts after each turn, from his notebook outputs")

    cols = [
        ("Run 1: messages", "trigger=(\"messages\", 10)\nkeep=(\"messages\", 4)", "blue",
         [("Q1", 2, ""), ("Q2", 4, ""), ("Q3", 6, ""), ("Q4", 8, ""), ("Q5", 10, ""), ("Q6", 6, "summarised")]),
        ("Run 2: tokens", "trigger=(\"tokens\", 550)\nkeep=(\"tokens\", 200)", "green",
         [("Paris", 4, "~149 tok"), ("London", 8, "~302 tok"), ("Tokyo", 12, "~456 tok"),
          ("New York", 8, "~396 tok  summarised"), ("Dubai", 5, "~232 tok  summarised"), ("Singapore", 9, "~361 tok")]),
        ("Run 3: fraction", "trigger=(\"fraction\", 0.005)\nkeep=(\"fraction\", 0.002)", "purple",
         [("Paris", 4, "~64 tok"), ("London", 8, "~133 tok"), ("Tokyo", 12, "~203 tok"),
          ("New York", 16, "~276 tok"), ("Dubai", 20, "~349 tok"), ("Singapore", 12, "~365 tok  summarised")]),
    ]
    cw, gap, x0 = 360, 20, 20
    for i, (title, params, color, rows) in enumerate(cols):
        x = x0 + i * (cw + gap)
        b.group(x, 90, cw, 520, title, color)
        b.text(x + cw / 2, 135, params, 11, color, "700", line_gap=1.3)
        yy = 190
        for label, n, note in rows:
            b.text(x + 14, yy + 13, label, 12, INK, "700", anchor="start")
            bw = 170
            drop = "summarised" in note
            b.bar(x + 95, yy, bw, n / 20, color="orange" if drop else color, h=18)
            b.text(x + 95 + bw * (n / 20) + 8, yy + 13, str(n), 12, INK, "700", anchor="start")
            if note:
                b.text(x + 14, yy + 40, note, 11, "orange" if drop else FAINT, "700" if drop else "400", anchor="start")
            yy += 62
    b.text(580, 640, "Bar = number of messages in the thread after that turn. A sudden drop (orange) means the middleware summarised.",
           12, FAINT)
    b.text(580, 662, "Token figures are his rough chars-divided-by-4 helper, not the middleware's own count, so do not expect the drop to line up exactly with the trigger value.",
           12, FAINT)
    return b


@board
def hitl_scribble():
    b = Board(1100, 600, "Human in the loop middleware",
              "Redrawn from the instructor's Excalidraw page at 2:20:40 to 2:22:40")

    b.group(20, 90, 620, 480, "Autonomous agent: nobody checks the result", "grey")
    b.person(335, 140, "yellow", 0.9)
    b.text(385, 170, "Human intervention", 14, "yellow", "700", anchor="start")
    ag = b.card(250, 270, 170, 90, "Agent", [], "dark", title_size=18)
    b.arrow((335, 212), ag.top(), color="yellow")
    b.text(70, 322, "I/P", 18, "grey", "700")
    b.text(580, 322, "O/P", 18, "grey", "700")
    b.arrow((100, 315), ag.left())
    b.arrow(ag.right(), (550, 315))
    b.arrow(ag.bottom(), (335, 440))
    b.text(335, 465, "autonomous agent", 15, "grey", "700")
    b.text(330, 530, "the human approves before the critical step runs", 12, FAINT)

    b.group(680, 90, 400, 480, "Why: a critical task", "yellow")
    f = b.card(720, 150, 320, 80, "Financial transaction", [], "yellow", title_size=16)
    s = b.card(720, 280, 320, 80, "Stock buy", ["the agent buys stock for you"], "yellow", size=12, title_size=16)
    c = b.card(720, 410, 320, 90, "Critical task", ["a mistake here means real", "financial loss"], "red", size=12, title_size=16)
    b.arrow(f.bottom(), s.top())
    b.arrow(s.bottom(), c.top())
    return b


@board
def hitl_flow():
    b = Board(1200, 800, "Human-in-the-loop: pause, decide, resume",
              "Explanatory board (not shown in the video): the interrupt and Command(resume=...) cycle from 6-middleware.ipynb")

    s1 = b.card(30, 100, 320, 100, "1. agent.invoke(messages, config)",
                ["same thread_id every time,", "the checkpointer remembers the pause"], "blue", size=12, title_size=13)
    s2 = b.card(420, 100, 340, 100, "2. model picks send_email_tool",
                ["read_email_tool is set to False,", "so it would not pause"], "purple", size=12, title_size=13)
    s3 = b.card(830, 100, 340, 100, "3. middleware interrupts",
                ["send_email_tool is in interrupt_on", "the tool has NOT run yet"], "red", size=12, title_size=13)
    b.arrow(s1.right(), s2.left())
    b.arrow(s2.right(), s3.left())

    r = b.card(830, 250, 340, 130, "result[\"__interrupt__\"]",
               ["action_requests: tool name + args", "review_configs: allowed_decisions", "approve | edit | reject"],
               "yellow", size=12, title_size=13)
    b.arrow(s3.bottom(), r.top())
    hum = b.card(440, 280, 320, 70, "A human reviews the call", [], "yellow", title_size=14)
    b.arrow(r.left(), hum.right())
    b.person(385, 262, "yellow", 0.8)

    d = b.diamond(590, 450, 250, 90, "decision?", "yellow", 14)
    b.arrow(hum.bottom(), d.top())

    cmd = b.card(30, 560, 1140, 56, "agent.invoke(Command(resume={\"decisions\": [ ... ]}), config=config)    # same config, same thread", [],
                 "dark", title_size=13)

    ap = b.card(35, 650, 330, 120, "approve", ["{\"type\": \"approve\"}", "tool runs with the original args", "ToolMessage: Email sent to john@test.com"],
                "green", size=12, title_size=14)
    ed = b.card(425, 650, 330, 120, "edit", ["{\"type\": \"edit\", \"edited_action\":", "{name, args}}  tool runs with the", "human's corrected recipient"],
                "orange", size=12, title_size=14)
    rj = b.card(835, 650, 330, 120, "reject", ["{\"type\": \"reject\"}", "tool never runs", "ToolMessage: User rejected the tool call"],
                "red", size=12, title_size=14)
    b.arrow(d.left(), (200, 450), via=[], color="green")
    b.arrow((200, 450), (200, 560), color="green")
    b.arrow(d.bottom(), (590, 560), color="orange", label="", width=1.8)
    b.arrow(d.right(), (1000, 450), color="red")
    b.arrow((1000, 450), (1000, 560), color="red")
    b.arrow((200, 616), ap.top(0.5), color="green", via=[(200, 632), (ap.cx, 632)])
    b.arrow((590, 616), ed.top(0.5), color="orange")
    b.arrow((1000, 616), rj.top(0.5), color="red", via=[(1000, 632), (rj.cx, 632)])
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"02-{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
