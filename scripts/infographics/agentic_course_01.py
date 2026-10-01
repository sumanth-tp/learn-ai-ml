"""Infographics for docs/projects/agentic-ai-complete-course chapter 01
(Introduction, LangChain setup, models and tools; video 0:00:00 to 1:06:40).

Run from the repo root:

    python3 scripts/infographics/agentic_course_01.py                 # all boards
    python3 scripts/infographics/agentic_course_01.py agent_whiteboard   # one board

Output: static/img/agentic-course/01-<name>.svg
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


def mono(b, x, y, text, size=12, color=INK, weight="400", anchor="start"):
    fill = PALETTE[color]["text"] if color in PALETTE else color
    b.parts.append(
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{MONO}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" xml:space="preserve">{esc(text)}</text>'
    )


def line(b, pts, color=INK, width=1.8, dashed=False):
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    b.parts.append(
        f'<path d="{d}" fill="none" stroke="{_col(color)}" stroke-width="{width}"{dash} '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )


def cross(b, cx, cy, s=11, color="red", width=4):
    line(b, [(cx - s, cy - s), (cx + s, cy + s)], color, width)
    line(b, [(cx - s, cy + s), (cx + s, cy - s)], color, width)


def circle_num(b, cx, cy, n, color="orange", r=13):
    c = PALETTE[color]
    b.parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="2"/>'
    )
    b.parts.append(
        f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-family="{MONO}" font-size="14" '
        f'font-weight="700" fill="{c["text"]}">{n}</text>'
    )


def code(b, x, y, w, h, title, lines, color="grey", size=12):
    """A left-aligned monospace card, for code and command lines."""
    return b.card(x, y, w, h, title, lines, color, size=size, align="left")


# ------------------------------------------------------------------ board 1


@board
def course_map():
    b = Board(
        1200, 660,
        "Complete Agentic AI Course in 10 Hours: the nine sections",
        "Start times are positions in the single 11h13m video. Chapter 1 of these notes covers 0:00:00 to 1:06:40.",
    )

    intro = b.card(30, 100, 1140, 62, "1 · Introduction  (0:00:00)", [
        "the plan, the three repos, who the course is for: about a month of study, aimed at interview-ready answers",
    ], "grey", size=12)

    # three themed groups
    g1 = b.group(30, 190, 370, 440, "Build agents", "blue")
    g2 = b.group(415, 190, 370, 440, "Give agents knowledge", "green")
    g3 = b.group(800, 190, 370, 440, "Make them safe and shippable", "orange")

    def sec(x, y, n, title, start, lines, color, here=False):
        c = b.card(x, y, 340, 92, f"{n} · {title}", [f"starts {start}"] + lines, color, size=12)
        if here:
            b.pill(x + 186, y - 12, "chapter 1 starts here", "pink", size=11, solid=True)
        return c

    sec(45, 235, 2, "LangChain v1", "0:02:31",
        ["agents, models, tools, messages,", "memory, middleware"], "blue", here=True)
    sec(45, 350, 3, "LangGraph", "2:35:12",
        ["state graphs, agentic apps", "with LangGraph"], "blue")
    sec(45, 465, 6, "Deep Agents", "8:02:11",
        ["deep research agents,", "practical implementation"], "blue")

    sec(430, 235, 4, "RAG", "5:02:29",
        ["traditional RAG and", "agentic RAG"], "green")
    sec(430, 350, 5, "Vectorless RAG", "7:10:43",
        ["RAG without a vector store,", "compared with vector RAG"], "green")

    sec(815, 235, 7, "Guardrails", "8:45:43",
        ["AI security: keep inputs and", "outputs inside the rules"], "orange")
    sec(815, 350, 8, "LLM evaluation", "9:22:55",
        ["techniques to measure", "answer quality, open source"], "orange")
    sec(815, 465, 9, "LLM gateways", "10:30:25",
        ["one front door to many models,", "with its implementation"], "orange")

    b.arrow(intro.bottom(0.2), (215, 190), color="grey")
    b.arrow(intro.bottom(0.5), (600, 190), color="grey")
    b.arrow(intro.bottom(0.8), (985, 190), color="grey")
    return b


# ------------------------------------------------------------------ board 2


@board
def docs_map():
    b = Board(
        1200, 640,
        "The LangChain docs the series walks through",
        "Three frameworks on the docs home page, and the Core components list the LangChain notebooks follow.",
    )

    # left: the three frameworks and LangSmith
    b.group(20, 95, 560, 520, "docs.langchain.com home page", "blue")
    lc = b.card(40, 140, 520, 84, "LangChain (Python)", [
        "quickly build agents with any model provider",
    ], "blue")
    lg = b.card(40, 250, 520, 84, "LangGraph (Python)", [
        "control every step: low-level orchestration,",
        "memory, human-in-the-loop",
    ], "purple")
    da = b.card(40, 360, 520, 84, "Deep Agents (Python)", [
        "agents for complex, multi-step tasks",
    ], "teal")
    pill_x = 52
    mono(b, 52, 478, "LangSmith, the platform around them:", 13, "grey", "700")
    for t in ["Observability", "Evaluation", "Prompt engineering", "Deployment"]:
        p = b.pill(pill_x, 495, t, "grey", size=12)
        pill_x += p.w + 8
    mono(b, 52, 548, "The course returns to LangGraph at 2:35:12 and to Deep", 12, FAINT)
    mono(b, 52, 566, "Agents at 8:02:11; this chapter stays inside LangChain.", 12, FAINT)
    b.pill(452, 124, "this chapter", "pink", size=11, solid=True)

    # right: LangChain Core components
    b.group(610, 95, 570, 520, "LangChain v1 sidebar: Core components", "green")
    items = [
        ("Agents", "create_agent, the first notebook", True),
        ("Models", "init_chat_model, ChatOpenAI, ChatGroq ...", True),
        ("Messages", "human, AI, system and tool messages", False),
        ("Tools", "@tool, bind_tools, the tool loop", True),
        ("Short-term memory", "conversation state across turns", False),
        ("Streaming", "stream() and batch(), with the models", True),
        ("Structured output", "typed answers from the model", False),
        ("Middleware", "built-in and custom hooks, guardrails", False),
    ]
    y = 135
    for name, note, here in items:
        color = "green" if here else "grey"
        b.card(635, y, 520, 48, name, [note], color, size=12, align="left", dashed=not here)
        if here:
            b.pill(1040, y + 14, "chapter 1", "pink", size=11, solid=True)
        y += 59
    return b


# ------------------------------------------------------------------ board 3


@board
def uv_setup_flow():
    b = Board(
        1200, 700,
        "Setting up the project with uv (video 5:05 to 20:00)",
        "Explanatory board: the terminal walkthrough as one flow. Commands are the ones typed in the video.",
    )
    W, H = 255, 150
    xs = [30, 330, 630, 930]
    y1, y2 = 105, 330

    def step(x, y, n, title, lines, color):
        c = code(b, x, y, W, H, "     " + title, lines, color)
        circle_num(b, x + 20, y + 20, n, color)
        return c

    s1 = step(xs[0], y1, 1, "Install uv", [
        "once per machine",
        "macOS/Linux:",
        "curl -LsSf https://astral.sh/",
        "  uv/install.sh | sh",
        "Windows: PowerShell script",
    ], "grey")
    s2 = step(xs[1], y1, 2, "uv init", [
        "inside the empty folder",
        "writes pyproject.toml,",
        ".python-version (3.13),",
        "main.py, README.md",
    ], "blue")
    s3 = step(xs[2], y1, 3, "uv venv", [
        "creates the .venv folder",
        "picks CPython 3.13.2",
        "typo seen on camera:",
        "'uv venv/' is rejected",
    ], "blue")
    s4 = step(xs[3], y1, 4, "Activate", [
        "Windows:",
        ".venv\\Scripts\\activate",
        "macOS/Linux:",
        "source .venv/bin/activate",
    ], "blue")

    s5 = step(xs[0], y2, 5, "requirements.txt", [
        "next to .venv, not inside",
        "langchain, langchain_community",
        "langchain-openai, langchain-groq",
        "python-dotenv,",
        "langchain-google-genai",
    ], "green")
    s6 = step(xs[1], y2, 6, "uv add -r", [
        "uv add -r requirements.txt",
        "installs AND records each",
        "package in pyproject.toml",
        "(uv add <name> for one)",
    ], "green")
    s7 = step(xs[2], y2, 7, ".env file", [
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "values come from each console",
    ], "orange")
    s8 = step(xs[3], y2, 8, "uv add ipykernel", [
        "lets Jupyter use this venv",
        "then pick the kernel",
        ".venv (Python 3.13.2)",
        "in the notebook toolbar",
    ], "purple")

    for a, c in [(s1, s2), (s2, s3), (s3, s4), (s5, s6), (s6, s7), (s7, s8)]:
        b.arrow(a.right(), c.left())
    # wrap from row 1 to row 2
    b.arrow(s4.bottom(), s5.top(), via=[(s4.cx, 290), (s5.cx, 290)], color="grey", dashed=True)

    b.group(30, 515, 1140, 165, "Result: what pyproject.toml records (versions printed in the video)", "yellow")
    code(b, 60, 555, 520, 110, "", [
        "requires-python = \">=3.13\"",
        "dependencies = [",
        "    \"ipykernel>=7.1.0\", \"langchain>=1.1.0\",",
        "    \"langchain-community>=0.4.1\",",
        "    \"langchain-google-genai>=3.2.0\", ... ]",
    ], "grey")
    b.card(620, 555, 520, 110, "", [
        "Unpinned requirements.txt gave the newest releases;",
        "pyproject.toml now freezes what you actually got.",
        "Tomorrow's release will not silently change your",
        "project, and langchain.__version__ prints 1.1.0.",
    ], "yellow", size=12, align="left")
    return b


# ------------------------------------------------------------------ board 4


@board
def agent_whiteboard():
    b = Board(
        1150, 700,
        "Agents: from a plain LLM app to a basic agent",
        "The instructor drew this on one Excalidraw page, adding the tool after the plain-LLM sketch.",
    )

    # step 1
    b.group(20, 95, 1110, 240, "Step 1 · LLM alone = a Gen AI application", "yellow")
    i1 = b.card(60, 190, 100, 44, "I/p", [], "yellow")
    l1 = b.card(260, 175, 180, 74, "LLM", ["OpenAI, Gemini, Groq,", "any open model"], "grey")
    o1 = b.card(540, 190, 100, 44, "O/p", [], "yellow")
    b.arrow(i1.right(), l1.left())
    b.arrow(l1.right(), o1.left())
    mono(b, 340, 142, "GenAI app", 15, "purple", "700", anchor="middle")
    b.card(740, 135, 370, 78, "Works: 'write 200 words on AI'", ["the model answers from what it learned"], "green", size=12)
    b.card(740, 232, 370, 92, "Breaks: 'today's AI news?'", [
        "training stopped at a cut-off date,",
        "so the model has nothing current",
    ], "red", size=12)
    cross(b, 700, 278, 12)

    # step 2
    b.group(20, 360, 1110, 320, "Step 2 · give it a tool = a basic agent", "purple")
    i2 = b.card(60, 532, 110, 62, "I/p", ["today's AI news?"], "yellow", size=11)
    l2 = b.card(260, 520, 190, 86, "LLM", ["decides what to do"], "grey")
    o2 = b.card(590, 540, 100, 46, "O/p", [], "yellow")
    tool = b.card(250, 402, 210, 62, "Tool", ["API, Google search ..."], "purple")
    b.arrow(i2.right(), l2.left())
    b.arrow(l2.right(), o2.left(), label="3 answer from\nthe context", color="green", label_dy=-22)
    b.arrow(l2.top(0.25), tool.bottom(0.25), color="purple", label="1 I cannot\nanswer this", label_dx=-62)
    b.arrow(tool.bottom(0.75), l2.top(0.75), color="purple", label="2 context", label_dx=48)

    b.card(760, 400, 350, 130, "Autonomous decisions", [
        "the model itself decides:",
        "which query needs a tool,",
        "when to route, how to solve",
        "the task: that is an agent",
    ], "purple", size=12)
    b.card(760, 550, 350, 110, "'ReAct' (written in the corner)", [
        "the older pattern for wiring an",
        "LLM to tools by hand. create_agent",
        "builds that loop for you.",
    ], "grey", size=12)
    return b


# ------------------------------------------------------------------ board 5


@board
def agent_graph():
    b = Board(
        1100, 560,
        "The agent graph before and after adding a tool",
        "create_agent returns a compiled LangGraph graph; the notebook draws it when you evaluate the variable.",
    )

    b.group(20, 95, 500, 440, "tools=[]  (29:15)", "grey")
    st = b.pill(240, 135, "__start__", "grey", size=13, anchor="middle")
    m = b.card(185, 215, 140, 50, "model", [], "blue", title_size=15)
    en = b.pill(255, 350, "__end__", "purple", size=13, solid=True, anchor="middle")
    b.arrow(st.bottom(), m.top())
    b.arrow(m.bottom(), en.top())
    code(b, 45, 420, 450, 96, "", [
        "agent = create_agent(",
        "    model=\"gpt-5\",",
        "    tools=[],",
        "    system_prompt=\"You are a helpful assistant.\")",
    ], "grey")
    mono(b, 270, 400, "no tool node: input, LLM, output", 12, FAINT, anchor="middle")

    b.group(560, 95, 520, 440, "tools=[get_weather]  (31:45)", "blue")
    st2 = b.pill(800, 135, "__start__", "grey", size=13, anchor="middle")
    m2 = b.card(720, 215, 140, 50, "model", [], "blue", title_size=15)
    en2 = b.pill(640, 350, "__end__", "purple", size=13, solid=True, anchor="middle")
    tl = b.card(870, 335, 140, 50, "tools", [], "orange", title_size=15)
    b.arrow(st2.bottom(), m2.top())
    b.arrow(m2.bottom(0.2), (en2.cx, en2.y), via=[(m2.x + m2.w * 0.2, 300), (en2.cx, 300)],
            label="answer ready", dashed=True)
    b.arrow(m2.bottom(0.8), tl.top(0.3), via=[(m2.x + m2.w * 0.8, 300), (tl.x + tl.w * 0.3, 300)],
            label="tool call", dashed=True, color="orange", label_dx=44, label_dy=12)
    b.arrow(tl.top(0.75), m2.right(0.8), via=[(tl.x + tl.w * 0.75, m2.y + m2.h * 0.8)], color="orange",
            label="result", label_dx=26)
    code(b, 585, 420, 470, 96, "", [
        "agent = create_agent(",
        "    model=\"gpt-5\",",
        "    tools=[get_weather],",
        "    system_prompt=\"You are a helpful assistant.\")",
    ], "blue")
    mono(b, 820, 410, "dashed edges are chosen at run time", 12, FAINT, anchor="middle")
    return b


# ------------------------------------------------------------------ board 6


@board
def model_integration():
    b = Board(
        1200, 640,
        "Calling a model: three providers, two ways each (37:45 to 50:00)",
        "Explanatory board: the table is from the notebook; the lower flow shows why every model object behaves alike.",
    )
    rows = [
        ["Provider", "Key in .env", "Package", "init_chat_model(...)", "Provider class"],
        ["OpenAI", "OPENAI_API_KEY", "langchain-openai", "\"gpt-4.1\"", "ChatOpenAI(model=\"gpt-4.1\")"],
        ["Google Gemini", "GOOGLE_API_KEY", "langchain-google-genai", "\"google_genai:gemini-2.5-flash-lite\"", "ChatGoogleGenerativeAI(model=\"gemini-2.5-flash-lite\")"],
        ["Groq", "GROQ_API_KEY", "langchain-groq", "\"groq:qwen/qwen3-32b\"", "ChatGroq(model=\"qwen/qwen3-32b\")"],
    ]
    b.table(20, 100, [110, 135, 185, 310, 420], rows, "blue", size=12)

    b.group(20, 290, 1160, 330, "Why the two ways are equivalent", "green")
    c1 = b.card(50, 390, 280, 100, "init_chat_model(\"provider:model\")", [
        "reads the prefix, imports the",
        "right integration package",
    ], "blue", size=12)
    cl = [
        b.card(450, 335, 300, 60, "ChatOpenAI", [], "orange"),
        b.card(450, 415, 300, 60, "ChatGoogleGenerativeAI", [], "orange"),
        b.card(450, 495, 300, 60, "ChatGroq", [], "orange"),
    ]
    for c in cl:
        b.arrow(c1.right(), c.left(), color="blue")
    c3 = b.card(880, 360, 270, 150, "One shared interface", [
        ".invoke(...)", ".stream(...)", ".batch([...])", ".bind_tools([...])",
    ], "green", size=13)
    for c in cl:
        b.arrow(c.right(), c3.left(), color="green")
    mono(b, 600, 590, "You can also build the class yourself: the result is the same kind of object.", 12, FAINT, anchor="middle")
    return b


# ------------------------------------------------------------------ board 7


@board
def invoke_stream_batch():
    b = Board(
        1200, 640,
        "invoke, stream and batch (50:00 to 57:20)",
        "Explanatory board: the same model object offers three ways to ask.",
    )

    # invoke
    b.group(20, 95, 370, 520, "model.invoke(prompt)", "blue")
    p = b.card(50, 150, 310, 56, "one prompt", ["\"Write me a 200 words paragraph ...\""], "grey", size=11)
    w = b.card(50, 260, 310, 100, "wait ...", ["nothing is shown until the", "whole answer is finished"], "yellow", size=12)
    r = b.card(50, 415, 310, 120, "one AIMessage", ["use .content to read the text,", "metadata holds token counts"], "blue", size=12)
    b.arrow(p.bottom(), w.top())
    b.arrow(w.bottom(), r.top())
    mono(b, 205, 580, "simplest, good for scripts", 12, FAINT, anchor="middle")

    # stream
    b.group(415, 95, 370, 520, "model.stream(prompt)", "green")
    p2 = b.card(445, 150, 310, 56, "one prompt", ["same text"], "grey", size=11)
    b.arrow(p2.bottom(), (600, 250))
    xs = [445, 545, 645]
    ch = []
    for i, x in enumerate(xs):
        ch.append(b.card(x, 250, 90, 60, f"chunk {i + 1}", ["a few words"], "green", size=11, title_size=12))
    mono(b, 600, 340, "... arrive one after another", 12, "green", anchor="middle")
    r2 = b.card(445, 380, 310, 156, "your for loop prints each", [
        "for chunk in model.stream(...):",
        "    print(chunk.text, end=\"|\",",
        "     flush=True)",
        "the | marks where chunks end",
    ], "green", size=12, align="left")
    for c in ch:
        b.arrow(c.bottom(), (c.cx, 380), color="green", width=1.4)
    mono(b, 600, 580, "what chat apps use: first words appear fast", 12, FAINT, anchor="middle")

    # batch
    b.group(810, 95, 370, 520, "model.batch([p1, p2, p3])", "orange")
    ps = [b.card(840, 150 + i * 62, 310, 50, t, [], "grey", title_size=12) for i, t in enumerate(
        ["parrots: colourful feathers?", "how do airplanes fly?", "what is quantum computing?"])]
    box = b.card(900, 365, 190, 70, "run in parallel", ["max_concurrency=5"], "orange", size=12)
    b.arrow(ps[2].bottom(0.5), (box.cx, 365), color="orange", label="all three\nat once", label_dx=50)
    r3 = b.card(840, 470, 310, 66, "a list of 3 AIMessages", ["same order as the inputs"], "orange", size=12)
    b.arrow(box.bottom(), r3.top(), color="orange")
    mono(b, 995, 580, "independent prompts, one call", 12, FAINT, anchor="middle")
    return b


# ------------------------------------------------------------------ board 8


@board
def tool_loop():
    b = Board(
        1200, 740,
        "The tool execution loop, done by hand (1:04:00 to 1:06:00)",
        "Explanatory board: this is what create_agent automates. Your code is the middle man between model and tool.",
    )
    # lanes
    b.group(20, 95, 380, 450, "Your code", "blue")
    b.group(470, 95, 280, 450, "Model + tools bound", "purple")
    b.group(790, 95, 390, 450, "Tool: get_weather", "orange")

    c1 = code(b, 35, 135, 350, 96, "step 1", [
        "messages = [{\"role\": \"user\", \"content\":",
        "   \"What's the weather in Boston?\"}]",
        "ai_msg = model_with_tools.invoke(messages)",
        "messages.append(ai_msg)",
    ], "blue", size=11)
    m1 = b.card(485, 135, 250, 96, "decides to call a tool", [
        "get_weather(", "  location='Boston')"], "purple", size=12)
    b.arrow(c1.right(0.3), m1.left(0.3), label="invoke", color="blue", label_dy=-12)
    b.arrow(m1.left(0.75), c1.right(0.75), label="AIMessage\n(tool_calls)", color="purple", label_dy=14)

    c2 = code(b, 35, 275, 350, 110, "step 2", [
        "for tool_call in ai_msg.tool_calls:",
        "    tool_result = get_weather.invoke(",
        "   tool_call)",
        "    messages.append(tool_result)",
    ], "blue", size=11)
    t2 = b.card(805, 290, 360, 80, "runs the Python function", [
        "returns \"It's sunny in Boston\"", "wrapped as a ToolMessage"], "orange", size=12)
    b.arrow(c2.right(0.25), t2.left(0.25), label="tool_call (name, args, id)", color="blue", label_dy=-14)
    b.arrow(t2.left(0.8), c2.right(0.8), label="ToolMessage", color="orange", label_dy=14)

    c3 = code(b, 35, 430, 350, 96, "step 3", [
        "final_response = model_with_tools.invoke(",
        "    messages)",
        "print(final_response.text)",
    ], "blue", size=11)
    m3 = b.card(485, 430, 250, 96, "reads the tool result", [
        "and writes the answer:", "\"The weather in Boston", "is sunny.\""], "purple", size=12)
    b.arrow(c3.right(0.3), m3.left(0.3), label="invoke(messages)", color="blue", label_dy=-12)
    b.arrow(m3.left(0.75), c3.right(0.75), label="final AIMessage", color="purple", label_dy=14)

    b.group(20, 565, 1160, 150, "The messages list grows (what the notebook prints at 1:05:30)", "green")
    xs = [40, 330, 620, 910]
    texts = [
        ("start", ["[ user dict ]"]),
        ("after step 1", ["[ user, AIMessage(tool_calls) ]"]),
        ("after step 2", ["[ user, AIMessage,", "  ToolMessage ]"]),
        ("step 3 reply", ["a new AIMessage: the", "final natural-language text"]),
    ]
    cards = []
    for x, (t, ls) in zip(xs, texts):
        cards.append(b.card(x, 610, 260, 90, t, ls, "green", size=12))
    for a, c in zip(cards, cards[1:]):
        b.arrow(a.right(), c.left(), color="green")
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"01-{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
