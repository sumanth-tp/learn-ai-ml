"""Infographics for docs/projects/agentic-ai-complete-course/08-llm-evaluation.md.

Chapter 8 of the "Complete Agentic AI Course in 10 Hours" (LLM evaluation,
9:22:55 to 10:30:25). Five boards redraw the instructor's whiteboard pages, the
LangSmith documentation diagram and the LangSmith result screens; five are
explanatory boards that the video does not show. Run from the repo root:

    python3 scripts/infographics/agentic_course_08.py            # all boards
    python3 scripts/infographics/agentic_course_08.py four_metrics
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import FAINT, INK, MONO, PALETTE, Board, esc  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "agentic-course"
PREFIX = "08-"
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


def circle_num(b, cx, cy, n, color="orange", r=14):
    c = PALETTE[color]
    b.parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="2"/>'
    )
    b.parts.append(
        f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-family="{MONO}" font-size="14" '
        f'font-weight="700" fill="{c["text"]}">{n}</text>'
    )


def cell_pill(b, cx, cy, ok, text=None, w=70):
    """A green or red result chip, as the LangSmith result screens show them."""
    color = "green" if ok else "red"
    c = PALETTE[color]
    label = text if text is not None else ("1.00" if ok else "0.00")
    b.parts.append(
        f'<rect x="{cx - w / 2}" y="{cy - 12}" width="{w}" height="24" rx="6" fill="{c["fill"]}" '
        f'stroke="{c["stroke"]}" stroke-width="1.6"/>'
    )
    b.parts.append(
        f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-family="{MONO}" font-size="13" '
        f'font-weight="700" fill="{c["text"]}">{esc(label)}</text>'
    )


# ------------------------------------------------------------------ board 1


@board
def chatbot_eval_plan():
    """9:25:30 to 9:30:30: the first Excalidraw page, the plan for evaluating a chatbot."""
    b = Board(1200, 740, "Evaluating a chatbot: the plan on the board",
              "Redrawn from the instructor's Excalidraw page, 9:30:30")

    inp = b.card(40, 120, 130, 70, "Input", ["I/P"], "yellow")
    bot = b.card(250, 110, 210, 90, "Chatbot", ["any LLM inside"], "grey", title_size=16)
    out = b.card(540, 120, 130, 70, "Output", ["O/P"], "yellow")
    b.arrow(inp.right(), bot.left(), color="yellow")
    b.arrow(bot.right(), out.left(), color="yellow")

    g = b.group(730, 90, 440, 190, "Data = ground truth", "pink")
    dc = b.card(755, 150, 110, 60, "DATA", [], "pink")
    i = b.card(940, 125, 205, 52, "I/P", ["the question"], "pink", size=12)
    o = b.card(940, 195, 205, 52, "O/P", ["the expected answer"], "pink", size=12)
    b.arrow(dc.right(0.3), i.left(), color="pink")
    b.arrow(dc.right(0.7), o.left(), color="pink")
    b.arrow(out.right(), g.left(0.5), color="pink", both=True, label="compare")

    q = b.group(30, 290, 660, 250, "Three questions the chatbot raises", "purple")
    q1 = b.card(48, 335, 200, 185, "1  Which LLM?",
                ["OpenAI, Gemini,", "Groq open-source", "", "cost helps decide,", "accuracy for the", "use case decides more"],
                "purple")
    q2 = b.card(268, 335, 200, 185, "2  Ground truth",
                ["for each input, the", "output you expect", "", "the response is", "compared against it"],
                "purple")
    q3 = b.card(488, 335, 190, 185, "3  Metrics",
                ["who does the", "comparing?", "", "an LLM, given a", "prompt: LLM as", "a judge"],
                "purple")

    judge = b.card(30, 580, 420, 120, "LLM as a judge + prompt",
                   ["the prompt tells the judge how to", "grade the chatbot's output", "against the expected output"],
                   "orange", title_size=15)
    ls = b.card(490, 580, 200, 120, "LangSmith", ["tracks datasets,", "experiments", "and scores"], "red",
                title_size=16)
    b.arrow(q3.bottom(0.5), judge.top(0.85), color="purple")
    b.arrow(judge.right(), ls.left(), color="red")

    s = b.group(730, 310, 440, 390, "Step by step", "green")
    steps = [
        ("Gather data points", "input and expected output pairs"),
        ("LLM as a judge", "a prompt that grades each answer"),
        ("Evaluation metrics", "apply them to the outputs"),
        ("Compare LLM models", "keep the one that scores best"),
    ]
    prev = None
    for k, (t, d) in enumerate(steps):
        y = 355 + k * 83
        card = b.card(790, y, 360, 66, t, [d], "green", size=12)
        circle_num(b, 765, y + 33, k + 1, "green")
        if prev is not None:
            b.arrow(prev.bottom(0.5), card.top(0.5), color="green")
        prev = card
    return b


# ------------------------------------------------------------------ board 2


@board
def dataset_to_experiment():
    """Explanatory: how client.evaluate wires the dataset, the target and the evaluators."""
    b = Board(1200, 640, "client.evaluate connects dataset, target and evaluators",
              "One experiment = every example run through your app, then scored by every evaluator")

    ds = b.cylinder(30, 230, 180, 170, "Dataset", ["Chatbots Evaluation", "5 examples", "question + answer"], "purple")

    b.group(260, 110, 350, 400, "target = your AI system", "blue")
    t = b.card(285, 160, 300, 80, "ls_target(inputs)", ["gets the question dict", "returns the response dict"], "blue")
    a = b.card(285, 285, 300, 90, "my_app(question)", ["chat completion", "gpt-4o-mini, temperature 0", "one short sentence"], "blue")
    o = b.card(285, 415, 300, 70, "outputs", ["{'response': '...'}"], "blue")
    b.arrow(t.bottom(), a.top(), color="blue")
    b.arrow(a.bottom(), o.top(), color="blue")

    b.group(660, 110, 310, 400, "evaluators", "orange")
    ev1 = b.card(680, 160, 270, 140, "correctness", ["LLM as a judge", "reads inputs, outputs and", "reference_outputs", "returns True / False"], "orange")
    ev2 = b.card(680, 330, 270, 140, "concision", ["plain Python length check", "reads outputs and", "reference_outputs", "returns 1 / 0"], "orange")

    ex = b.card(1015, 230, 165, 170, "Experiment", ["openai-4o-mini-", "chatbot-<id>", "", "score per row", "average per metric"], "red",
                title_size=14)

    b.arrow(ds.right(), t.left(), via=[(235, 230 - 0), (235, t.cy)], color="purple", label="each\nquestion", label_dx=-2)
    b.arrow(o.right(), ev1.left(), via=[(630, o.cy), (630, ev1.cy)], color="blue", label="response", label_dx=4, label_dy=60)
    b.arrow(ds.bottom(), ev2.bottom(0.5), via=[(ds.cx, 560), (ev2.cx, 560)], color="purple", dashed=True,
            label="reference_outputs: the ground truth", label_at=0.5)
    b.arrow(ev1.right(), ex.left(0.3), color="orange")
    b.arrow(ev2.right(), ex.left(0.7), color="orange")
    b.text(600, 612, "client.evaluate(ls_target, data=dataset_name, evaluators=[correctness, concision], experiment_prefix=...)",
           12, FAINT)
    return b


# ------------------------------------------------------------------ board 3


@board
def judge_call():
    """Explanatory: what the correctness judge sees and returns."""
    b = Board(1200, 560, "Inside the first correctness evaluator",
              "A prompt is assembled, a judge model answers CORRECT or INCORRECT, Python turns that into a boolean")

    q = b.card(30, 110, 250, 70, "inputs['question']", ["what the user asked"], "blue")
    r = b.card(30, 215, 250, 70, "reference_outputs['answer']", ["the real answer"], "green")
    p = b.card(30, 320, 250, 70, "outputs['response']", ["what the chatbot said"], "orange")

    pr = b.card(335, 150, 270, 200, "user_content",
                ["You are grading the", "following question: ...", "Here is the real answer:", "...", "predicted answer: ...",
                 "Respond with CORRECT", "or INCORRECT. Grade:"], "yellow")
    for src, t in ((q, 0.2), (r, 0.5), (p, 0.8)):
        b.arrow(src.right(), pr.left(t), color="grey")

    sysc = b.card(335, 380, 270, 90, "system prompt", ["'You are an expert professor", "specialised in grading answers'"], "grey")
    judge = b.card(660, 190, 220, 120, "Judge LLM", ["openai_client (wrapped)", "gpt-4o-mini", "temperature 0"], "purple", title_size=15)
    b.arrow(pr.right(), judge.left(), color="yellow", label="user")
    b.arrow(sysc.right(), judge.bottom(0.5), via=[(770, 425)], color="grey", label="system", label_dx=-30)

    txt = b.card(930, 120, 230, 60, "'CORRECT'", ["or 'INCORRECT'"], "pink")
    b.arrow(judge.right(0.25), txt.left(), color="purple")
    d = b.diamond(1045, 270, 190, 90, "response ==\n'CORRECT' ?", "yellow")
    b.arrow(txt.bottom(), d.top(), color="pink")
    t = b.pill(960, 370, "True", "green", 14, solid=True)
    f = b.pill(1090, 370, "False", "red", 14, solid=True)
    b.arrow(d.bottom(0.2), (t.cx, t.y), color="green")
    b.arrow(d.bottom(0.8), (f.cx, f.y), color="red")
    b.text(1045, 450, "that boolean is the score\nLangSmith records", 13, FAINT)
    b.text(600, 530, "exact string match: any extra word or full stop from the judge would score False",
           12, "red", "700")
    return b


# ------------------------------------------------------------------ board 4


@board
def chatbot_results():
    """9:54:45 to 9:57:00: the LangSmith experiment screens for the two models."""
    b = Board(1200, 700, "Chatbot experiments in LangSmith",
              "Redrawn from the result screens, 9:54:45 to 9:57:00 (5 examples, two models)")

    b.text(40, 108, "Experiments table (dataset: Chatbots Evaluation)", 14, "blue", "700", anchor="start")
    b.table(40, 122, [330, 250, 150, 150, 200],
            [["Experiment", "Model", "Concision", "Correctness", "P50 latency"],
             ["openai-4o-mini-chatbot-ac3151d5", "gpt-4o-mini", "0.40", "0.60", "1.14 s"],
             ["openai-4-turbo-chatbot-ae4ea2ed", "gpt-4-turbo", "0.00", "1.00", "about 1.2 s (read from a small frame)"]],
            "blue", size=13)

    b.text(40, 290, "Per-example rows, experiment 1 (gpt-4o-mini)", 14, "green", "700", anchor="start")
    RH = 46
    b.table(40, 304, [180, 440, 110, 110],
            [["Input", "Reference output", "Concision", "Correct"],
             ["What is LangChain?", "A framework for building LLM applications", "", ""],
             ["What is Google?", "A technology company known for search", "", ""],
             ["What is OpenAI?", "A company that creates Large Language Models", "", ""],
             ["What is LangSmith?", "A platform for observing and evaluating LLM applications", "", ""],
             ["What is Mistral?", "A company that creates Large Language Models", "", ""]],
            "green", size=12, row_h=RH)
    vals = [(1, 1), (0, 1), (0, 1), (1, 0), (0, 0)]
    for k, (c1, c2) in enumerate(vals):
        y = 304 + RH * (k + 1) + RH / 2
        cell_pill(b, 40 + 180 + 440 + 55, y, bool(c1), w=64)
        cell_pill(b, 40 + 180 + 440 + 110 + 55, y, bool(c2), w=64)

    b.card(890, 300, 270, 120, "How the averages arise", ["concision: 2 of 5 = 0.40", "correctness: 3 of 5 = 0.60"],
           "yellow", size=12)
    b.card(890, 440, 270, 150, "gpt-4-turbo, same data", ["correctness 1.00,", "concision 0.00: all judged", "right, none under 2x", "the reference length"],
           "purple", size=12)
    return b


# ------------------------------------------------------------------ board 5


@board
def rag_eval_whiteboard():
    """9:57:30 to 10:02:30 (and 10:13:00): the second Excalidraw page, RAG evaluation."""
    b = Board(1200, 700, "RAG evaluation: three questions, three steps",
              "Redrawn from the instructor's Excalidraw page, 10:02:30")

    hd = b.card(40, 100, 330, 56, "RAG evaluation", [], "orange", title_size=18)
    ls = b.card(900, 100, 230, 56, "LangSmith  (done)", [], "red", title_size=15)

    q = b.group(30, 180, 760, 190, "What we have to figure out", "blue")
    b.card(50, 222, 330, 120, "1  Create test datasets?", ["question and expected", "answer pairs"], "blue")
    b.card(405, 222, 360, 120, "2  Run the RAG app", ["over those test datasets", "to get answers and documents"], "blue")
    b.text(795, 270, "", 12)

    m = b.card(810, 215, 350, 130, "3  Measure RAG performance", ["with different", "evaluation metrics"], "pink")
    b.arrow(ls.bottom(), m.top(0.5), color="red", dashed=True, label="tracked here")

    st = b.group(30, 410, 1130, 260, "Experiments: steps", "green")
    c1 = b.card(60, 460, 330, 170, "1  RAG", ["data ingestion", "retriever", "generation", "", "built first in this video"], "green")
    c2 = b.card(435, 460, 330, 170, "2  Test data", ["question  <->  answer", "", "the answer is the", "ground truth"], "green")
    c3 = b.card(810, 460, 330, 170, "3  Evaluation metrics", ["LLM as a judge", "", "four evaluators,", "one per question asked"], "green")
    b.arrow(c1.right(), c2.left(), color="green")
    b.arrow(c2.right(), c3.left(), color="green")
    b.text(225, 655, "done by 10:10", 12, "green", "700")
    b.text(600, 395, "", 12)
    return b


# ------------------------------------------------------------------ board 6


@board
def four_metrics():
    """9:59:00: the LangSmith documentation diagram the instructor pastes into the board."""
    b = Board(1200, 640, "The four RAG metrics on the pipeline",
              "Redrawn from the LangSmith documentation diagram shown at 9:59:00")

    qn = b.card(30, 290, 120, 60, "Question", [], "blue")
    sr = b.cylinder(210, 262, 130, 116, "Search", ["retriever"], "purple")
    dc = b.card(207, 440, 136, 60, "Documents", ["the corpus"], "grey")
    rd = b.card(410, 280, 150, 80, "Relevant", ["documents"], "teal", title_size=14)
    llm = b.card(630, 270, 160, 100, "LLM", ["context window", "= the documents"], "yellow", title_size=15)
    an = b.card(860, 285, 110, 70, "Answer", [], "orange", title_size=14)
    rf = b.card(1040, 285, 130, 70, "Reference", ["answer"], "blue", title_size=14)

    b.arrow(qn.right(), sr.left(), color="grey")
    b.arrow(dc.top(), sr.bottom(), color="grey")
    b.arrow(sr.right(), rd.left(), color="grey")
    b.arrow(rd.right(), llm.left(), color="grey")
    b.arrow(llm.right(), an.left(), color="grey")
    b.arrow(an.right(), rf.left(), color="blue", both=True)

    # answer relevance: question -> answer, over the top
    b.arrow(qn.top(0.25), an.top(0.3), via=[(qn.x + qn.w * 0.25, 125), (an.x + an.w * 0.3, 125)], color="orange")
    b.text(600, 110, "Answer relevance: does the answer address the question?", 14, "orange", "700")
    # retrieval relevance: question -> relevant documents
    b.arrow(qn.top(0.75), rd.top(0.5), via=[(qn.x + qn.w * 0.75, 195), (rd.cx, 195)], color="green")
    b.text(330, 182, "Retrieval relevance: are the documents relevant to the question?", 14, "green", "700")
    # groundedness: relevant documents -> answer, underneath
    b.arrow(rd.bottom(0.5), an.bottom(0.5), via=[(rd.cx, 470), (an.cx, 470)], color="red")
    b.text(685, 498, "Groundedness: is the answer grounded in the documents?", 14, "red", "700")
    # correctness
    b.text(1050, 245, "Correctness: does the answer\nmatch the ground truth?", 14, "blue", "700")

    b.pill(30, 560, "retrieval metric", "green", 13, solid=True)
    b.pill(190, 560, "generation metrics", "orange", 13, solid=True)
    b.text(430, 577, "retrieval relevance judges the search step; the other three judge what the LLM wrote",
           12, FAINT, anchor="start")
    return b


# ------------------------------------------------------------------ board 7


@board
def rag_pipeline():
    """Explanatory: the RAG app built before evaluating it."""
    b = Board(1200, 660, "The RAG app that gets evaluated",
              "Built once in the notebook, then wrapped in rag_bot so the evaluators see both answer and documents")

    b.group(30, 100, 1140, 190, "Ingestion and retrieval (built once)", "purple")
    u = b.card(50, 150, 190, 110, "3 blog posts", ["Lilian Weng:", "agents, prompt", "engineering,", "adversarial attacks"], "grey", size=11)
    l = b.card(280, 150, 170, 110, "WebBaseLoader", ["one load() per URL", "flattened to a list"], "purple", size=11)
    s = b.card(490, 150, 210, 110, "Text splitter", ["RecursiveCharacter", "from_tiktoken_encoder", "chunk_size=250", "chunk_overlap=0"], "purple", size=11)
    e = b.card(740, 150, 200, 110, "Vector store", ["InMemoryVectorStore", "OpenAIEmbeddings()"], "purple", size=11)
    r = b.card(980, 150, 170, 110, "retriever", ["as_retriever(k=6)", "returns 4 here, see", "the note in the page"], "orange", size=11)
    for x, y in [(u, l), (l, s), (s, e), (e, r)]:
        b.arrow(x.right(), y.left(), color="purple")

    b.group(30, 330, 1140, 290, "rag_bot(question)   decorated with @traceable", "blue")
    c1 = b.card(50, 390, 200, 100, "retriever.invoke", ["question in,", "relevant chunks out"], "orange", size=12)
    c2 = b.card(285, 390, 200, 100, "docs_string", ["' '.join of every", "doc.page_content"], "blue", size=12)
    c3 = b.card(520, 390, 260, 100, "system prompt", ["helpful assistant, use the", "documents, say if unknown,", "three sentences maximum"], "blue", size=12)
    c4 = b.card(815, 390, 160, 100, "llm.invoke", ["gpt-4o-mini via", "init_chat_model"], "yellow", size=12)
    c5 = b.card(1010, 390, 140, 100, "return", ["{'answer': ...,", "'documents': docs}"], "green", size=11)
    for x, y in [(c1, c2), (c2, c3), (c3, c4), (c4, c5)]:
        b.arrow(x.right(), y.left(), color="blue")
    b.arrow(r.bottom(), c1.top(), via=[(r.cx, 310), (c1.cx, 310)], color="orange", label="called from rag_bot")
    q = b.pill(60, 545, "user question", "blue", 12, solid=True)
    b.arrow((q.x + q.w / 2, q.y), c1.bottom(0.3), color="blue")
    b.arrow((q.x + q.w, q.y + 11), (c3.x + 60, c3.y + c3.h), via=[(380, q.y + 11)], color="blue", dashed=True)
    b.card(560, 545, 590, 55, "why documents are returned too", ["groundedness and retrieval relevance judge the docs, so they must leave the bot"],
           "green", size=11)
    return b


# ------------------------------------------------------------------ board 8


@board
def four_evaluators():
    """Explanatory: the four evaluators side by side."""
    b = Board(1200, 520, "The four evaluators side by side",
              "Each one is a function that receives named arguments and returns a boolean")
    b.table(40, 100, [190, 120, 240, 330, 110, 120],
            [["Evaluator", "Part judged", "Compares", "Reads", "Judge", "Key"],
             ["correctness", "generation", "answer vs ground truth", "inputs.question\nreference_outputs.answer\noutputs.answer", "gpt-4o-mini", "correct"],
             ["relevance", "generation", "answer vs question", "inputs.question\noutputs.answer", "gpt-4o", "relevant"],
             ["groundedness", "generation", "answer vs retrieved docs", "outputs.documents\noutputs.answer", "gpt-4o", "grounded"],
             ["retrieval_relevance", "retrieval", "retrieved docs vs question", "inputs.question\noutputs.documents", "gpt-4o", "relevant"]],
            "blue", size=13)
    b.card(40, 395, 540, 95, "Needs a reference answer?", ["only correctness; the other three", "work from the question, the documents", "and the answer alone"], "yellow", size=12,
           align="left")
    b.card(620, 395, 540, 95, "Same recipe four times", ["TypedDict grade (explanation, then boolean),", "a grading prompt, ChatOpenAI at temperature 0", "with structured output, and a thin wrapper"], "purple", size=12,
           align="left")
    return b


# ------------------------------------------------------------------ board 9


@board
def rag_results():
    """10:28:00 to 10:28:30: the RAG experiment in LangSmith."""
    b = Board(1200, 620, "RAG experiment: rag-doc-relevance",
              "Redrawn from the LangSmith result screens, 10:28:00 to 10:28:30 (3 examples, 4 evaluators)")

    cols = [400, 170, 190, 150, 200]
    b.table(40, 100, cols,
            [["Input", "Correctness", "Groundedness", "Relevance", "Retrieval relevance"],
             ["How does the ReAct agent use self-reflection?", "", "", "", ""],
             ["What are the types of biases that can arise with few-shot prompting?", "", "", "", ""],
             ["What are five types of adversarial attacks?", "", "", "", ""],
             ["Average", "1.00", "0.67", "1.00", "0.67"]],
            "teal", size=13, row_h=62)
    xs = [40 + 400 + 85, 40 + 570 + 95, 40 + 760 + 75, 40 + 910 + 100]
    grid = [(1, 0, 1, 0), (1, 1, 1, 1), (1, 1, 1, 1)]
    for r, row in enumerate(grid):
        y = 100 + 62 + r * 62 + 31
        for x, v in zip(xs, row):
            cell_pill(b, x, y, bool(v))

    b.card(40, 440, 360, 150, "Row 1 is the interesting one", ["correct and relevant,", "but not grounded and the", "retrieval was judged off-topic"],
           "red", size=12)
    b.card(430, 440, 360, 150, "Why the chart said 0.50", ["the bars were probably", "captured while results were", "still arriving; final is 2 of 3 = 0.67"],
           "yellow", size=12)
    b.card(820, 440, 340, 150, "Also on the page", ["latency, token count and", "cost per run, plus the full", "trace behind every row"],
           "grey", size=12)
    return b


# ------------------------------------------------------------------ board 10


@board
def reading_scores():
    """Explanatory: turning the four scores into a diagnosis."""
    b = Board(1200, 640, "Reading the four scores together",
              "A pattern across the columns tells you which part to fix first")
    rows = [
        ("retrieval relevance False", "red", "the search fetched the wrong chunks",
         "chunk size, number of chunks, embedding model, query rewriting"),
        ("groundedness False, retrieval True", "orange", "the model added facts beyond its context",
         "stricter prompt, 'say you do not know', lower temperature"),
        ("correct but not grounded", "yellow", "right answer, but from the model's memory, not your documents",
         "open the trace; a RAG app that ignores its documents is not proving RAG works"),
        ("correctness False, grounded True", "purple", "the documents lack the fact, or the reference answer is wrong",
         "check ingestion coverage and re-read the dataset's ground truth"),
        ("relevance False", "blue", "the answer drifts away from the question",
         "tighten the system prompt, keep answers on the question asked"),
    ]
    b.text(40, 106, "Pattern", 14, "grey", "700", anchor="start")
    b.text(380, 106, "What it usually means", 14, "grey", "700", anchor="start")
    b.text(800, 106, "What to try first", 14, "grey", "700", anchor="start")
    y = 122
    for pat, col, mean, fix in rows:
        c1 = b.card(30, y, 320, 80, pat, [], col, title_size=13)
        c2 = b.card(370, y, 400, 80, "", [mean], "grey", size=12)
        c3 = b.card(790, y, 380, 80, "", [fix], "grey", size=12)
        b.arrow(c1.right(), c2.left(), color=col)
        b.arrow(c2.right(), c3.left(), color=col)
        y += 96
    b.text(600, 622, "A small dataset only suggests a cause: confirm it in the trace before changing anything",
           12, FAINT)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"{PREFIX}{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
