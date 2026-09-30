"""Infographics for docs/projects/ai-security (Krish Naik's AI Security course).

Each function redraws one board, slide or app figure from the course as an
original image. Run from the repo root:

    python3 scripts/infographics/ai_security.py            # all boards
    python3 scripts/infographics/ai_security.py mosaic     # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "ai-security"
BOARDS = {}


def board(fn):
    BOARDS[fn.__name__] = fn
    return fn


# --------------------------------------------------------------------- Module 1


@board
def m1_why_guardrails():
    b = Board(1000, 430, "LLM Guardrails", "Why an LLM application needs a security layer (0:03 to 0:06)")
    top = b.card(380, 100, 240, 56, "LLM security", [], "red", title_size=18)
    items = [
        ("Prompt injection", ["The successor to SQL injection:", "instructions smuggled into", "the model's input"], "yellow"),
        ("Jailbreak", ["Talking the model out of", "its rules, like unlocking", "unlimited shots in a game"], "yellow"),
        ("Cost saving", ["Every off-topic answer burns", "tokens; refusing early", "saves money"], "green"),
    ]
    for i, (t, lines, col) in enumerate(items):
        box = b.card(60 + i * 300, 230, 280, 150, t, lines, col, title_size=15)
        b.arrow(top.bottom(), box.top(), color="red")
    b.text(500, 410, "threats the guardrail layer exists to stop", 12, "grey", italic=True)
    return b


@board
def m1_where_guardrail_sits():
    b = Board(1300, 560, "Where the guardrail sits", "LLMs power agents and RAG; a guard checks what goes in and what comes out (0:06 to 0:13)")
    llms = b.card(40, 110, 170, 56, "LLMs", [], "yellow", title_size=18)
    ag = b.card(270, 100, 190, 44, "Agentic AI", [], "pink")
    rag = b.card(270, 160, 190, 44, "RAG", [], "pink")
    b.arrow(llms.right(0.35), ag.left())
    b.arrow(llms.right(0.7), rag.left())

    b.group(40, 240, 540, 290, "RAG without a guard", "grey")
    u1 = b.person(110, 300, "yellow", label="user")
    l1 = b.card(260, 320, 140, 56, "LLM", ["answers"], "pink")
    db = b.cylinder(450, 300, 100, 110, "DB", ["documents"], "yellow")
    b.arrow((130, 340), l1.left(0.35), label="question")
    b.arrow(l1.left(0.75), (130, 362), label="answer")
    b.arrow(l1.right(0.3), (450, 330), label="retrieve", curve=True)
    b.arrow((450, 380), l1.right(0.8), label="context", curve=True)

    b.group(620, 240, 300, 290, "RAG with a guardrail", "red")
    u2 = b.person(770, 282, "pink", 0.7, label="user")
    gr = b.card(680, 370, 180, 50, "guardrails", ["check in and out"], "red")
    l2 = b.card(700, 455, 140, 50, "LLM", [], "green")
    b.arrow((770, 352), gr.top())
    b.arrow(gr.bottom(), l2.top())
    b.arrow(l2.left(), gr.left(0.8), via=[(665, l2.cy), (665, gr.y + gr.h * 0.8)], color="red", label="")

    b.group(960, 240, 310, 290, "Why check the output too", "orange")
    b.card(990, 285, 250, 60, "Chatbot on 50 GB", ["of enterprise data"], "orange")
    b.card(990, 370, 250, 60, "~500 MB is relevant", ["to this assistant"], "yellow")
    b.card(990, 455, 250, 60, "Answer only from it", ["nothing else leaks out"], "red")
    b.arrow((1115, 345), (1115, 370))
    b.arrow((1115, 430), (1115, 455))
    return b


@board
def m1_gateways_vs_guardrails():
    b = Board(1200, 520, "Gateways vs guardrails", "What a production LLM app needs, and which layer delivers it (0:13 to 0:16)")
    b.group(40, 100, 540, 380, "Reliability", "blue")
    r1 = b.card(70, 150, 220, 50, "Robust ✓", [], "blue", title_size=16)
    r2 = b.card(70, 220, 220, 50, "Fault tolerant ✓", [], "blue", title_size=16)
    r3 = b.card(70, 290, 220, 50, "Latency free", [], "blue", title_size=16, dashed=True)
    gw = b.card(360, 200, 190, 80, "LLM gateways", ["retries · fallback", "routing · caching"], "teal", title_size=16)
    b.arrow(r1.right(), gw.left(0.3))
    b.arrow(r2.right(), gw.left(0.6))
    b.text(310, 400, "a gateway keeps the app answering", 12, "blue", italic=True)
    b.text(310, 420, "when a provider fails or slows down", 12, "blue", italic=True)

    b.group(620, 100, 540, 380, "Security", "red")
    s1 = b.card(650, 150, 200, 56, "Secured ✓", [], "yellow", title_size=16)
    gr = b.card(650, 250, 200, 70, "Guardrails", ["the security layer"], "red", title_size=16)
    b.arrow(s1.bottom(), gr.top())
    rules = b.card(920, 150, 200, 50, "Rules", [], "red")
    reg = b.card(920, 230, 200, 50, "Regulatory", ["constraints"], "red")
    b.arrow(gr.right(0.3), rules.left())
    b.arrow(gr.right(0.7), reg.left())
    b.person(700, 370, "yellow", 0.8, label="user")
    b.arrow((730, 395), (820, 395), color="red", width=2.4)
    b.person(860, 370, "red", 0.8, label="turns malicious")
    b.card(950, 370, 180, 70, "Rules stop", ["the bad request"], "red")
    return b


@board
def m1_frameworks():
    b = Board(1150, 420, "Guardrail frameworks", "Four options discussed; the demo uses NeMo (0:16 to 0:20)")
    top = b.card(455, 100, 240, 56, "Guardrails", [], "red", title_size=18)
    fw = [
        ("NeMo Guardrails", ["NVIDIA · open source", "Colang rules", "used in this demo"], "green"),
        ("Guardrails AI", ["validators hub", "Python-first"], "yellow"),
        ("Llama Firewall", ["Meta", "prompt-attack and", "agent safety"], "yellow"),
        ("AWS Bedrock Guardrails", ["managed service", "topics · content · PII", "grounding"], "yellow"),
    ]
    for i, (t, lines, col) in enumerate(fw):
        box = b.card(40 + i * 275, 230, 245, 130, t, lines, col, title_size=14)
        b.arrow(top.bottom(), box.top(), color="yellow" if col == "yellow" else "green")
    b.pill(162, 375, "the demo", "green", solid=True, anchor="middle")
    return b


@board
def m1_dialog_rails():
    b = Board(1200, 560, "NeMo Guardrails Classroom · message flow",
              "Each experiment adds one branch; by Experiment 5 the intent check has five (0:24 to 0:33)")
    um = b.card(40, 250, 150, 60, "User", ["message"], "blue")
    ic = b.diamond(330, 280, 190, 120, "Intent check\n(LLM call 1)", "yellow")
    b.arrow(um.right(), ic.left())
    rows = [
        ("Refuse: off-topic", "off-topic", "pink", "Exp 2"),
        ("Refuse: jailbreak", "jailbreak", "pink", "Exp 3"),
        ("Refuse: sensitive topic", "sensitive", "pink", "Exp 4"),
        ("Scripted dialog", "dialog intent", "purple", "Exp 5"),
        ("LLM answer (LLM call 2)", "IT question", "green", "Exp 2"),
    ]
    br = b.card(1010, 250, 150, 60, "Bot", ["response"], "blue")
    for i, (t, label, col, exp) in enumerate(rows):
        y = 100 + i * 86
        lines = ["greeting · help · bye"] if col == "purple" else []
        box = b.card(610, y, 250, 60, t, lines, col)
        b.arrow(ic.right(), box.left(), label=label, label_at=0.62)
        b.arrow(box.right(), br.left(0.2 + i * 0.15), label="scripted reply" if col == "purple" else "")
        b.pill(870, y + 18, exp, "grey")
    b.text(600, 540, "Dialog rails don't block, they guide: matched intents get scripted, instant replies with no LLM call",
           12, "purple", italic=True)
    return b


@board
def m1_pii_urgency():
    b = Board(1300, 470, "Experiment 6 · custom Python actions",
              "Systematic input rails run on every message, before intent classification (0:57 to 0:58)")
    um = b.card(30, 190, 130, 60, "User", ["message"], "blue")
    pii = b.card(210, 170, 200, 100, "PII detector", ["Python action", "email · phone · SSN", "API keys · cards"], "yellow")
    urg = b.card(470, 170, 200, 100, "Urgency detector", ["Python action", "keyword scan for", "production emergencies"], "yellow")
    warn = b.card(470, 330, 200, 70, "Warn: urgent", ["then continue"], "orange")
    ic = b.diamond(820, 220, 180, 110, "Intent check\n(LLM call 1)", "yellow")
    llm = b.card(960, 90, 170, 60, "LLM answer", ["LLM call 2"], "green")
    bot = b.card(1150, 190, 120, 60, "Bot", ["response"], "blue")
    blk = b.card(210, 340, 200, 70, "Block: PII found", ["request stops here"], "red")
    b.arrow(um.right(), pii.left(), label="every msg")
    b.arrow(pii.right(), urg.left(), label="clean")
    b.arrow(pii.bottom(), blk.top(), label="PII found", color="red")
    b.arrow(urg.right(), ic.left(), label="normal")
    b.arrow(urg.bottom(), warn.top(), label="urgent", color="orange")
    b.arrow(warn.right(), (ic.cx, ic.y + ic.h), via=[(ic.cx, warn.cy)], label="continue")
    b.arrow((ic.cx + 40, ic.y + 20), llm.left(), label="allowed")
    b.arrow(ic.right(), bot.left(), label="rail fired")
    b.arrow(llm.right(), bot.top())
    b.arrow(blk.bottom(), bot.bottom(), via=[(blk.cx, 440), (1210, 440)], color="red", label="blocked reply")
    return b


@board
def m1_colang():
    b = Board(1200, 560, "NeMo Guardrails → rails → Colang", "Rails are rules and regulations, written in Colang (0:38 to 0:44)")
    nemo = b.card(40, 110, 220, 56, "NeMo Guardrails", ["by NVIDIA"], "green")
    rails = b.card(320, 110, 170, 56, "Rails", [], "red", title_size=18)
    rr = b.card(550, 110, 240, 56, "Rules & regulations", [], "pink")
    b.arrow(nemo.right(), rails.left())
    b.arrow(rails.right(), rr.left())
    co = b.card(850, 100, 300, 76, "Colang  ·  .co files", ["read by the guard LLM", "+ Python actions in .py"], "purple")
    b.arrow(rr.right(), co.left())

    b.group(40, 220, 440, 300, "Between two kinds of language", "purple")
    nl = b.card(70, 270, 170, 60, "Natural language", ["how users talk"], "yellow")
    pl = b.card(280, 270, 170, 60, "Programming", ["language"], "yellow")
    mid = b.card(160, 400, 200, 70, "Colang", ["readable like English,", "strict like code"], "purple")
    b.arrow(nl.bottom(), mid.top(0.3))
    b.arrow(pl.bottom(), mid.top(0.7))

    b.group(520, 220, 640, 300, "Three kinds of block", "red")
    du = b.card(550, 270, 280, 110, "define user ask off topic",
                ['"tell me a joke"', '"write me a poem"', '"recommend a movie"'], "pink", size=11, align="left")
    db = b.card(850, 270, 280, 110, "define bot refuse off topic",
                ['"I\'m an Enterprise IT', 'Assistant … ask me', 'anything technical!"'], "pink", size=11, align="left")
    df = b.card(700, 410, 280, 90, "define flow handle off topic",
                ["user ask off topic", "bot refuse off topic", "stop"], "red", size=11, align="left")
    b.arrow(du.bottom(0.5), df.top(0.3))
    b.arrow(db.bottom(0.5), df.top(0.7))
    b.pill(550, 386, "'off topic' is a variable: the intent's name", "red")
    return b


@board
def m1_intent_matching():
    b = Board(1250, 470, "How NeMo matches an intent", "The query becomes a vector; the closest example intents go to the guard LLM (0:46 to 0:50)")
    u = b.person(80, 150, "red", label="user")
    q = b.card(170, 150, 230, 76, "query", ['"what\'s the capital', 'of France?"'], "red")
    fe = b.card(460, 150, 190, 76, "FastEmbed", ["runs locally", "query → vector"], "yellow")
    v = b.card(700, 150, 220, 76, "[0.1, 0.3, 0.4 …]", ["query vector"], "grey")
    b.arrow((110, 190), q.left())
    b.arrow(q.right(), fe.left())
    b.arrow(fe.right(), v.left())
    b.group(700, 270, 520, 180, "Example vectors from the .co file", "purple")
    ex = [("ask off topic", "closest ✓", "green"), ("attempt jailbreak", "far", "grey"), ("express greeting", "far", "grey")]
    for i, (t, tag, col) in enumerate(ex):
        box = b.card(720 + i * 165, 310, 150, 80, t, ["[ … ]"], "purple", size=11)
        b.pill(box.cx, 402, tag, col, anchor="middle", solid=col == "green")
    b.arrow(v.bottom(0.3), (766, 270), label="")
    b.text(760, 252, "similarity", 11, "purple", "700", anchor="end")
    llm = b.card(960, 150, 240, 76, "Guard LLM decides", ["intent = ask off topic", "→ run that rail"], "red")
    b.arrow(v.right(), llm.left())
    b.card(170, 280, 480, 120, "Colang → rails", ["The examples are not a keyword list: a message that",
                                                 "matches none of them can still land on the right",
                                                 "intent if it is semantically close."], "green", size=12)
    return b


@board
def m1_observability():
    b = Board(1250, 520, "Security layer + observability layer", "Guard the calls, then see every call (0:51 to 0:55)")
    b.card(40, 100, 260, 60, "NeMo Guardrails", ["security layer"], "red")
    b.card(40, 180, 260, 60, "Pydantic Logfire", ["observability layer"], "yellow")
    b.group(340, 90, 420, 390, "LangChain ecosystem", "pink")
    lc = b.card(460, 140, 180, 50, "LangChain", [], "pink")
    kids = [("LangChain", "chains"), ("LangGraph", "agent graphs"), ("LangSmith", "observability")]
    for i, (t, sub) in enumerate(kids):
        box = b.card(360 + i * 132, 250, 124, 70, t, [sub], "pink", size=11)
        b.arrow(lc.bottom(), box.top())
    b.text(550, 380, "LangSmith = LangChain's", 12, "pink")
    b.text(550, 398, "tracing and evals", 12, "pink")
    b.group(800, 90, 420, 390, "Pydantic ecosystem", "yellow")
    py = b.card(920, 140, 180, 50, "Pydantic", [], "yellow")
    kids = [("Validation", "typed models"), ("Pydantic AI", "agents"), ("Logfire", "tracing")]
    for i, (t, sub) in enumerate(kids):
        box = b.card(820 + i * 132, 250, 124, 70, t, [sub], "yellow", size=11)
        b.arrow(py.bottom(), box.top())
    b.card(820, 360, 380, 90, "Validation underneath",
           ["LangChain · AutoGen · CrewAI · FastAPI", "all rely on Pydantic, e.g. checking", "that abc@gmail.com is an email"], "grey", size=11)
    return b


@board
def m1_three_rails():
    b = Board(1100, 420, "Three kinds of rails", "Where each check runs (0:55 to 0:57)")
    top = b.card(440, 100, 220, 56, "Rails", [], "yellow", title_size=18)
    kinds = [
        ("Input rails", ["check the message", "before the LLM sees it", "topic · jailbreak · intent"], "pink"),
        ("Output rails", ["check the reply", "before the user sees it", "e.g. regex for secrets"], "pink"),
        ("Custom rails", ["Python @action", "systematic: run on", "every message"], "pink"),
    ]
    for i, (t, lines, col) in enumerate(kinds):
        box = b.card(60 + i * 340, 220, 300, 120, t, lines, col, title_size=15)
        b.arrow(top.bottom(), box.top())
    b.pill(740, 360, "988196310 → PII  (a custom regex rail catches it)", "red")
    return b


@board
def m1_keys():
    b = Board(900, 280, "Keys the demo needs", "Bring your own key (0:58 to 1:03)")
    llm = b.card(60, 120, 180, 60, "LLM calls", ["chat + guard models"], "pink")
    g = b.card(420, 90, 400, 60, "groq_api_key", ["free at console.groq.com"], "pink")
    l = b.card(420, 170, 400, 60, "pydantic logfire api", ["token for tracing"], "yellow")
    b.arrow(llm.right(0.35), g.left())
    b.arrow(llm.right(0.7), l.left())
    return b


# --------------------------------------------------------------------- Module 2


@board
def m2_hiring():
    b = Board(1250, 520, "Evaluation = hiring a candidate", "Past scores are like benchmarks; the interview is your own evaluation (1:23 to 1:28)")
    b.person(110, 140, "grey", 1.2, label="Candidate")
    goo = b.card(250, 140, 250, 70, "Google", ["Role: GenAI engineer"], "blue")
    b.arrow((150, 170), goo.left())
    sc = b.card(40, 320, 300, 110, "Scores or metrics", ["10th: 82%", "12th: 85%", "CGPA: 9"], "yellow")
    iv = b.card(380, 320, 320, 110, "Interview", ["1. Test → score", "2. Reasoning abilities", "3. One-on-one interview"], "green")
    b.arrow((100, 250), sc.top(0.3))
    b.arrow((130, 250), iv.top(0.3))
    b.group(760, 110, 460, 360, "LLM evaluation", "purple")
    ev = b.card(890, 160, 200, 50, "LLM eval", [], "purple", title_size=16)
    bm = b.card(790, 300, 190, 90, "Benchmarks", ["generic tasks", "published at launch"], "yellow")
    ju = b.card(1000, 300, 200, 90, "Human / LLM as judge", ["your task,", "your data"], "green")
    b.arrow(ev.bottom(0.3), bm.top())
    b.arrow(ev.bottom(0.7), ju.top())
    b.arrow(sc.top(0.8), bm.left(0.2), color="grey", dashed=True, label="is like",
            via=[(sc.x + sc.w * 0.8, 290), (775, 290), (775, bm.y + bm.h * 0.2)])
    b.arrow(iv.bottom(), ju.bottom(), via=[(iv.cx, 480), (ju.cx, 480)], color="grey", dashed=True, label="is like")
    return b


@board
def m2_two_things():
    b = Board(1300, 670, "Two things you can evaluate", "The most important distinction in LLM evaluation: make it explicit (1:28 to 1:30)")
    b.group(30, 90, 610, 520, "A) Evaluate the model", "grey")
    b.text(335, 140, "Which model is better? · mostly off the shelf", 13, "grey", italic=True)
    b.card(55, 160, 560, 170, "Benchmarks (capability tests)",
           ["Knowledge: MMLU", "Maths: GSM8K, MATH", "Code: HumanEval, MBPP", "Reasoning: HellaSwag, ARC, GPQA",
            "Honesty: TruthfulQA · Instructions: IFEval"], "grey", size=12, align="left")
    b.card(55, 345, 560, 90, "Leaderboards",
           ["LMArena / Chatbot Arena (Elo, human votes)", "HF Open LLM Leaderboard · HELM · Artificial Analysis"],
           "grey", size=12, align="left")
    b.card(55, 450, 560, 135, "Reference-based metrics",
           ["n-gram: BLEU · ROUGE-N/L · METEOR · exact match / F1", "model-based: BERTScore · BLEURT",
            "perplexity: exp(avg negative log-likelihood)", "lower = better · open weights only"],
           "grey", size=12, align="left")
    b.group(660, 90, 610, 520, "B) Evaluate the application", "green")
    b.text(965, 140, "Is my RAG or agent doing its job?", 13, "green", italic=True)
    b.card(685, 160, 560, 70, "No public benchmark fits your data",
           ["custom development: your data, your tasks, your metrics"], "red")
    b.card(685, 245, 560, 70, "Golden dataset", ["your own labelled examples, specific to your task"], "green")
    b.card(685, 325, 560, 70, "Task-specific metrics", ["custom to your domain and use case"], "green")
    b.card(685, 405, 560, 70, "LLM-as-judge", ["an LLM scores other LLM outputs: usually the main approach"], "green")
    b.card(685, 490, 560, 90, "Frameworks · rest of this module",
           ["RAG triad · G-Eval · DeepEval · Ragas"], "purple", title_size=13)
    b.pill(335, 626, "a research point: 10% of your time", "grey", anchor="middle")
    b.pill(965, 626, "where 90% of the work is", "green", solid=True, anchor="middle")
    return b


@board
def m2_goldens_judge():
    b = Board(1250, 520, "Goldens + RAG output → judge → metrics", "Phase 1 runs the app; phase 2 judges it (1:33 to 2:04)")
    b.group(30, 90, 330, 250, "Golden · written by you", "yellow")
    b.card(50, 135, 290, 50, "Queries", ["what users will ask"], "yellow", size=11)
    b.card(50, 195, 290, 50, "Expected answer", ["the truth"], "yellow", size=11)
    b.card(50, 255, 290, 60, "Expected context", ["chunks, sources"], "yellow", size=11)
    rag = b.card(420, 170, 200, 70, "RAG pipeline", ["phase 1: run it"], "blue")
    b.arrow((340, 160), rag.left(0.3), label="query")
    b.group(680, 90, 300, 250, "Generated by the RAG pipeline", "blue")
    aa = b.card(700, 140, 260, 60, "Actual answer", [], "blue")
    rc = b.card(700, 230, 260, 60, "Retrieved contexts", [], "blue")
    b.arrow(rag.right(0.3), aa.left())
    b.arrow(rag.right(0.7), rc.left())
    j = b.card(420, 380, 360, 90, "Judge LLM · phase 2", ["a state-of-the-art model", "one call per metric"], "red")
    b.arrow((195, 340), j.left(), via=[(195, j.cy)])
    b.arrow((830, 340), j.top(0.85), via=[(830, 360), (j.x + j.w * 0.85, 360)])
    b.group(1010, 90, 220, 400, "Metrics", "green")
    for i, m in enumerate(["Answer relevance", "Context precision", "Context recall", "Faithfulness", "Answer correctness"]):
        b.card(1025, 130 + i * 70, 190, 54, m, [], "green", size=11, title_size=12)
    b.arrow(j.right(0.5), (1010, 290), via=[(995, j.cy), (995, 290)], label="scores")
    return b


def _claims_board(title, subtitle, left_title, left_lines, claims, score_text, value, threshold, note):
    b = Board(1300, 560, title, subtitle)
    b.card(30, 100, 420, 200, left_title, left_lines, "blue", size=11, align="left")
    b.card(30, 320, 420, 110, "Judge question", ["Can this claim be fully inferred", "from the retrieved context?", "yes / no"], "purple")
    b.group(480, 90, 790, 350, "Atomic claims", "grey")
    for i, (claim, verdict, where, ok) in enumerate(claims):
        y = 135 + i * 74
        col = "green" if ok else "red"
        b.card(500, y, 520, 58, claim, [where], col, size=11)
        b.pill(1040, y + 17, verdict, col, solid=True)
    b.card(30, 450, 420, 90, score_text, [note], "red" if value < threshold else "green")
    b.bar(520, 480, 560, value, threshold, "red" if value < threshold else "green", label=f"{value:.2f}")
    return b


@board
def m2_faithfulness():
    return _claims_board(
        "Metric 1 · Faithfulness (groundedness)", "Is every claim in the answer supported by the retrieved chunks? (2:04 to 2:12)",
        "Retrieved context (chunks)",
        ["Chunk 1: minimum balance is ₹10,000 for", "urban branches. Non-maintenance fee is", "₹350 + GST if balance falls below.", "",
         "Chunk 2: semi-urban minimum ₹5,000.", "Rural branch minimum ₹2,500."],
        [("Min balance is ₹10,000 for urban branches", "grounded", "chunk 1", True),
         ("Non-maintenance fee is ₹350 + GST", "grounded", "chunk 1", True),
         ("Online fund transfer has no extra charge", "hallucinated", "not in any chunk", False),
         ("Rural branch minimum is ₹2,500", "grounded", "chunk 2", True)],
        "Faithfulness = 3 ÷ 4 = 0.75", 0.75, 0.8, "below the 0.8 threshold: fail")


@board
def m2_answer_relevancy():
    b = Board(1250, 520, "Metric 2 · Answer relevancy", "Does the answer address the question? The judge works backwards (2:12 to 2:18)")
    b.card(30, 110, 260, 70, "input", ["the user question"], "grey")
    b.card(30, 200, 260, 70, "actual_output", ["the LLM response"], "grey")
    s1 = b.card(360, 180, 330, 100, "Step 1 · generate questions", ["the judge reads the answer and", "invents N questions it answers"], "green")
    s2 = b.card(760, 110, 440, 100, "Step 2 · similarity check", ["each generated question vs the input", "embedding cosine, not keywords"], "green")
    b.arrow((290, 235), s1.left())
    b.arrow((290, 145), s2.left(), via=[(725, 145)])
    b.arrow(s1.right(), s2.left(0.8), via=[(725, s1.cy), (725, s2.y + s2.h * 0.8)])
    sc = b.card(760, 240, 440, 80, "Score = average similarity", ["0.0 → 1.0 · 1.0 = fully on-topic"], "purple")
    b.arrow(s2.bottom(), sc.top())
    b.group(30, 350, 1170, 150, "What causes a low score", "orange")
    for i, (t, sub) in enumerate([("Off-topic answer", "answers a different question"),
                                  ("Padded response", "filler dilutes the on-topic signal"),
                                  ("Incomplete answer", "skips part of the question")]):
        b.card(60 + i * 380, 395, 350, 80, t, [sub], "red")
    return b


@board
def m2_rag_triad():
    b = Board(1000, 560, "The RAG triad", "Three pieces, three relationships: cover all three and every aspect is covered (2:18)")
    q = b.card(390, 100, 220, 70, "Query", [], "blue", title_size=18)
    c = b.card(700, 400, 220, 70, "Context", [], "yellow", title_size=18)
    r = b.card(80, 400, 220, 70, "Response", [], "teal", title_size=18)
    b.arrow(q.right(), c.top(), via=[(810, q.cy)], label="Context relevance\nis the context relevant\nto the query?", label_color="yellow")
    b.arrow(c.left(), r.right(), label="Groundedness\nis the response supported\nby the context?", label_color="yellow")
    b.arrow(r.top(), q.left(), via=[(190, q.cy)], label="Answer relevance\nis the answer relevant\nto the query?", label_color="teal")
    b.text(500, 330, "⟲", 60, "grey")
    return b


@board
def m2_context_precision():
    b = Board(1250, 600, "Metric 3 · Context precision", "Are the relevant chunks ranked above the noise? (2:18 to 2:24)")
    b.card(30, 100, 1190, 50, "Query: What is the minimum balance for my savings account?", [], "blue", title_size=14)
    rows = [("#1", "Min balance ₹10,000 urban branches", True, "P@1 = 1.00"),
            ("#2", "Non-maintenance fee ₹350 + taxes", True, "P@2 = 1.00"),
            ("#3", "KYC update required every 8 years", False, "P@3 = 0.67 (not counted)"),
            ("#4", "Semi-urban balance ₹5,000", True, "P@4 = 0.75"),
            ("#5", "Rural balance ₹2,500", True, "P@5 = 0.80")]
    for i, (rk, text, rel, pk) in enumerate(rows):
        y = 175 + i * 64
        col = "green" if rel else "red"
        b.pill(40, y + 12, rk, "grey", solid=True)
        b.card(100, y, 560, 50, text, [], col, title_size=13)
        b.pill(680, y + 12, "relevant" if rel else "noise", col, solid=True)
        b.text(800, y + 31, pk, 14, col, "700", anchor="start")
    b.card(30, 505, 640, 75, "Position penalty",
           ["Noise at rank k penalises every relevant chunk after it.", "The earlier the noise, the bigger the hit."], "orange", size=11)
    b.card(700, 480, 520, 100, "Precision = (1 + 1 + 0.75 + 0.80) ÷ 4 = 0.89",
           ["mean of P@k at the relevant ranks only", "threshold ≥ 0.7: pass"], "green", size=12)
    return b


@board
def m2_context_recall():
    b = Board(1300, 620, "Metric 4 · Context recall", "Did retrieval fetch everything the ideal answer needs? (2:24 to 2:30)")
    b.card(30, 100, 400, 80, "reference", ["the ground-truth answer: a proxy", "for what the retriever must cover"], "grey", size=11)
    b.card(30, 195, 400, 80, "retrieved_contexts", ["chunks fetched by the retriever,", "attributed against reference claims"], "grey", size=11)
    b.card(470, 100, 380, 80, "Step 1 · extract claims", ["judge splits the reference", "into atomic claims"], "green", size=11)
    b.card(890, 100, 380, 80, "Step 2 · attribution check", ["can each claim be attributed", "to a retrieved chunk? yes / no"], "green", size=11)
    b.arrow((430, 140), (470, 140))
    b.arrow((850, 140), (890, 140))
    claims = [("Urban branches require ₹10,000 minimum balance", "chunk 1 · yes", True),
              ("Non-maintenance fee ₹350 + taxes below the limit", "chunk 2 · yes", True),
              ("Rural branch minimum balance is ₹2,500", "no chunk · no", False),
              ("Semi-urban branch minimum balance is ₹5,000", "chunk 4 · yes", True)]
    for i, (t, v, ok) in enumerate(claims):
        y = 300 + i * 62
        col = "green" if ok else "red"
        b.card(470, y, 560, 50, t, [], col, title_size=12)
        b.pill(1050, y + 13, ("supported · " if ok else "missing · ") + v, col, solid=True)
    b.card(30, 300, 400, 110, "Recall = 3 ÷ 4 = 0.75", ["each claim weighted equally", "threshold ≥ 0.7"], "green")
    b.group(30, 430, 400, 170, "What causes a low score", "orange")
    b.card(45, 470, 370, 36, "Missing chunks: key facts never fetched", [], "red", title_size=11)
    b.card(45, 512, 370, 36, "k too small: fact sits at rank k+1", [], "red", title_size=11)
    b.card(45, 554, 370, 36, "Embedding gap: relevant doc ranked too low", [], "red", title_size=11)
    for i, (band, lo, hi, col) in enumerate([("Low", "0.0", "0.4", "red"), ("Medium", "0.4", "0.7", "yellow"), ("High", "0.7", "1.0", "green")]):
        b.card(470 + i * 270, 560, 250, 44, f"{band} · {lo}–{hi}", [], col, title_size=12)
    return b


@board
def m2_answer_correctness():
    b = Board(1300, 640, "Metric 5 · Answer correctness", "Factual F1 blended with semantic similarity (2:30 to 2:37)")
    for i, (t, sub) in enumerate([("user_input", "the question"), ("response", "LLM's answer"), ("reference", "ground truth")]):
        b.card(30 + i * 215, 100, 200, 60, t, [sub], "grey", size=11)
    b.group(30, 185, 640, 330, "Component 1 · factual F1", "green")
    claims = [("Minimum balance ₹10,000 for urban branches", "TP", "green"),
              ("Penalty fee is ₹400 per month", "FP", "red"),
              ("Internet banking is available at no cost", "FP", "red"),
              ("Interest rate is 3.5% per annum", "TP", "green"),
              ("Non-maintenance fee ₹350 + taxes (missed)", "FN", "orange"),
              ("Passbook issued free of charge (missed)", "FN", "orange")]
    for i, (t, tag, col) in enumerate(claims):
        y = 225 + i * 44
        b.card(45, y, 520, 36, t, [], col, title_size=11)
        b.pill(580, y + 7, tag, col, solid=True)
    b.card(30, 525, 640, 90, "F1 = TP ÷ (TP + ½ (FP + FN)) = 2 ÷ 4 = 0.50", ["catches wrong values and hallucinations"], "green", size=11)
    b.group(700, 185, 570, 170, "Component 2 · semantic similarity", "blue")
    b.card(720, 225, 530, 50, "embedding cosine(response, reference)", [], "blue", title_size=12)
    b.bar(720, 300, 420, 0.72, None, "blue", label="0.72")
    b.card(700, 370, 570, 110, "Blend", ["0.75 × 0.50 + 0.25 × 0.72 = 0.555 ≈ 0.55", "weights w₁ = 0.75, w₂ = 0.25 (defaults)", "move weight to the factual side when facts matter"], "purple", size=11)
    b.card(700, 495, 570, 120, "Requires embeddings", ["unlike the other Ragas metrics, this one", "needs an embedding model; similar-domain", "texts score high even when the facts are wrong"], "yellow", size=11)
    return b


# --------------------------------------------------------------------- Module 3


@board
def m3_mosaic():
    b = Board(1400, 1250, "MOSAIC — Multi-Agent Clinical Trial Intelligence Engine")

    # Row 1: sources → ingestion → storage
    b.group(20, 90, 360, 250, "Data Sources", "green")
    ct = b.card(40, 140, 320, 74, "ClinicalTrials.gov API v2", ["400K+ studies · FREE · no auth"], "blue")
    pm = b.card(40, 240, 320, 74, "PubMed eUtils API", ["Research papers · FREE · no auth"], "blue")

    b.group(420, 90, 420, 250, "Ingestion Layer", "orange")
    c1 = b.card(440, 132, 380, 58, "clinical_trials_client.py",
                ["requests · asyncio.to_thread · retry · pagination"], "orange", size=11)
    c2 = b.card(440, 200, 380, 58, "pubmed_client.py",
                ["esearch + efetch · XML parse · batch · rate limit"], "orange", size=11)
    dp = b.card(440, 268, 380, 58, "document_parser.py",
                ["Raw JSON → ParsedStudy · ParsedPaper (Pydantic)"], "orange", size=11)

    b.group(880, 90, 500, 250, "GCP Storage Layer", "green")
    gcs = b.card(900, 132, 460, 86, "Google Cloud Storage",
                 ["raw/studies/ · raw/papers/", "processed/studies/ · processed/papers/"], "green")
    sql = b.card(900, 230, 460, 96, "Cloud SQL · PostgreSQL + pgvector",
                 ["studies · chunks · signals · hitl_reviews", "episodes · procedures · sponsor_knowledge"],
                 "green")

    b.arrow(ct.right(), c1.left())
    b.arrow(pm.right(), c2.left())
    b.arrow(c1.right(0.5), gcs.left(0.5), via=[(860, c1.cy), (860, gcs.cy)], label="save raw")

    # Row 2: processing → memory
    b.group(20, 380, 620, 270, "Processing Layer", "purple")
    ch = b.card(40, 424, 180, 206, "chunker.py",
                ["500 words/chunk", "50-word overlap", "Field-labelled", "TextChunk output"], "purple",
                size=11)
    em = b.card(240, 424, 180, 206, "embedder.py",
                ["OpenAI embed", "3-small model", "1536 dims", "batch size 50"], "purple", size=11)
    vs = b.card(440, 424, 180, 206, "vector_store.py",
                ["asyncpg pool", "Cosine <=> op", "Filter by source", "144 chunks stored"], "purple", size=11)
    b.arrow(dp.bottom(), (dp.cx, 380), label="")
    b.arrow(ch.right(), em.left())
    b.arrow(em.right(), vs.left())

    b.group(680, 380, 700, 270, "Memory Layer — LangMem", "red")
    ep = b.card(700, 424, 210, 206, "Episodic Memory",
                ["Past sessions saved", "Semantic search", "\"What did I find", "before?\"", "Never forgets"],
                "red", size=11)
    pr = b.card(925, 424, 210, 206, "Procedural Memory",
                ["How to reason", "Updated by HITL", "Rejection → new rule", "All future sessions"], "red",
                size=11)
    se = b.card(1150, 424, 210, 206, "Semantic Memory",
                ["Sponsor knowledge", "Credibility score", "Broken promises count", "Avg delay days"], "red",
                size=11)
    b.arrow(vs.right(0.35), ep.left(0.35), label="embeddings")
    b.arrow(vs.right(0.7), ep.left(0.7), label="store chunks")

    # Row 3: agent graph
    b.group(20, 690, 1360, 270, "Agent Graph — LangGraph + GPT-4o (parallel execution)", "blue")
    sup = b.card(470, 736, 460, 64, "Supervisor Agent",
                 ["Routes tasks · Activates specialists · Compiles final brief"], "dark", size=11)
    agents = [
        ("Broken Promises", ["Outcome switching", "detection", "signal: broken_promise", "HITL threshold 0.60"]),
        ("Missing Results", ["Completed trials,", "no results posted", "signal: missing_results", "HITL threshold 0.65"]),
        ("Track Record", ["Sponsor credibility", "scores over time", "signal: low_credibility", "HITL threshold 0.70"]),
        ("Pattern Finder", ["Cross-study patterns", "invisible to humans", "signal: cross_study", "HITL threshold 0.65"]),
        ("Side Effect Checker", ["Filing vs papers", "safety gap check", "signal: safety_gap", "HITL threshold 0.55"]),
        ("Timeline Analyst", ["Study vs own schedule", "silent delay detection", "signal: timeline_delay",
                              "HITL threshold 0.60"]),
    ]
    boxes = []
    for i, (name, lines) in enumerate(agents):
        x = 40 + i * 222
        box = b.card(x, 830, 206, 112, name, lines, "red", size=10, title_size=12)
        boxes.append(box)
        b.arrow(sup.bottom(), box.top(), color="grey", width=1.3)
    b.arrow(pr.bottom(), (pr.cx, 690), both=True, label="reads · updates")

    # Row 4: HITL, API, observability
    b.group(20, 1000, 440, 230, "Human-in-the-Loop Gate", "orange")
    rq = b.card(40, 1040, 400, 82, "Review Queue",
                ["Low-confidence signals wait for a human", "Approve → signals · Reject → memory update",
                 "Edit → fix summary"], "orange", size=10)
    ll = b.card(40, 1134, 400, 80, "The Learning Loop",
                ["Rejection reason → ProceduralStore update", "New rule added to the agent",
                 "Loaded in EVERY future session"], "orange", size=10)
    b.arrow(boxes[0].bottom(), rq.top(0.2), label="low confidence")
    b.arrow(rq.bottom(0.9), ll.top(0.9))
    b.arrow(ll.left(), (30, 900), via=[(28, ll.cy), (28, 900)], color="red", dashed=True)
    b.text(60, 975, "feedback loop", 11, "red", "700", anchor="start")

    b.group(480, 1000, 500, 230, "FastAPI + Cloud Run", "teal")
    b.card(500, 1040, 270, 172, "API routes",
           ["POST /api/v1/analyze", "GET /api/v1/signals", "GET /api/v1/review/queue",
            "PATCH /api/v1/review/{id}", "GET /api/v1/memory/episodes", "GET /api/v1/sponsors/{name}",
            "GET /api/v1/health"], "teal", size=10, align="left")
    b.card(785, 1040, 175, 172, "Google Cloud Run",
           ["Serverless", "Scales to zero", "AMD64 Docker", "2 GB RAM · 2 vCPU", "~$35–55 / month"], "teal",
           size=10)
    b.arrow(boxes[3].bottom(), (boxes[3].cx, 1000), label="signals")

    b.group(1000, 1000, 380, 230, "Observability + External", "purple")
    b.card(1020, 1040, 340, 40, "LangSmith", ["traces · tokens · evals · prompts"], "purple", size=10)
    b.card(1020, 1088, 340, 40, "OpenAI API", ["GPT-4o reasoning · text-embedding-3-small"], "purple", size=10)
    b.card(1020, 1136, 340, 40, "GCP Secret Manager", ["OpenAI key · LangSmith key · DB creds"], "purple",
           size=10)
    b.card(1020, 1184, 340, 36, "GCP Cloud Logging", ["activation logs · latency · health"], "purple", size=10)
    return b



@board
def m3_lineage():
    b = Board(1300, 520, "The agent memory lineage", "Thirteen techniques, each invented to fix the one before it (2:55)")
    b.group(20, 90, 620, 400, "Short-term · lives in RAM, resets per session", "blue")
    short = [("1", "Conversation buffer", "keep every message"), ("2", "Sliding window", "keep the last k turns"),
             ("3", "Summary", "compress old turns"), ("4", "Summary buffer", "recent verbatim + summary"),
             ("5", "Token buffer", "hard token budget")]
    for i, (n, t, sub) in enumerate(short):
        y = 135 + i * 70
        b.pill(40, y + 16, n, "blue", solid=True)
        b.card(80, y, 540, 56, t, [sub], "blue", size=11)
    b.group(660, 90, 620, 400, "Long-term · lives in a database", "purple")
    long = [("6", "Vector store", "retrieve by meaning"), ("7", "Entity", "one record per thing"),
            ("8", "Episodic", "sessions as episodes"), ("9", "Semantic", "distilled facts"),
            ("10", "Procedural", "learned rules"), ("11", "Self-reflection", "own post-mortems")]
    for i, (n, t, sub) in enumerate(long):
        col, row = i % 2, i // 2
        x, y = 680 + col * 300, 135 + row * 90
        b.pill(x, y + 22, n, "purple", solid=True)
        b.card(x + 44, y, 240, 70, t, [sub], "purple", size=11)
    b.card(680, 410, 280, 60, "12 · Memory routing", ["pick the right store"], "teal", size=11)
    b.card(980, 410, 280, 60, "13 · Forgetting and decay", ["let unused memories fade"], "red", size=11)
    return b


@board
def m3_survey():
    b = Board(1250, 460, "Memory in the Age of AI Agents", "The survey's three questions (NUS, 2026) · shown at 2:57")
    root = b.card(475, 95, 300, 56, "Agent memory", [], "dark", title_size=16)
    cols = [("Forms", "what carries memory?", ["Token-level: flat, planar,", "hierarchical", "Parametric", "Latent"], "blue"),
            ("Functions", "why do agents need it?", ["Factual: user, environment", "Experiential: case, strategy,", "skill, hybrid", "Working: single / multi-turn"], "green"),
            ("Dynamics", "how does it operate?", ["Formation: summarise, distil,", "structure, internalise", "Evolution", "Retrieval"], "orange")]
    for i, (t, q, lines, col) in enumerate(cols):
        x = 40 + i * 400
        h = b.card(x, 200, 370, 60, t, [q], col, title_size=16)
        b.arrow(root.bottom(), h.top())
        b.card(x, 280, 370, 150, "", lines, col, size=12, align="left")
    return b


@board
def m3_buffer():
    b = Board(1300, 560, "1 · Conversation buffer memory", "Store every message and re-send the whole list on every call (3:00 to 3:12)")
    turns = [("Turn 1", "[System] + [User 1]", 149), ("Turn 2", "[System] + [User 1] + [AI 1] + [User 2]", 238),
             ("Turn 3", "… + [AI 2] + [User 3]", 334), ("Turn N", "[System] + ALL previous turns + [User N]", 976)]
    b.group(20, 90, 640, 300, "What goes to the API", "blue")
    for i, (t, sent, tok) in enumerate(turns):
        y = 135 + i * 62
        b.pill(40, y + 12, t, "blue", solid=True)
        b.card(130, y, 510, 48, sent, [], "blue", title_size=12)
    b.group(690, 90, 590, 300, "Prompt tokens per turn (notebook)", "orange")
    vals = [149, 238, 334, 423, 513, 605, 700, 791, 882, 976]
    for i, v in enumerate(vals):
        x = 720 + i * 54
        h = v / 976 * 210
        b.parts.append(f'<rect x="{x}" y="{360 - h:.1f}" width="38" height="{h:.1f}" rx="4" fill="#e8590c" fill-opacity="{0.35 + 0.06 * i:.2f}"/>')
        b.text(x + 19, 376, str(i + 1), 11, "grey")
        if i in (0, 9):
            b.text(x + 19, 352 - h, str(v), 11, "orange", "700")
    b.text(985, 150, "6.6× growth in 10 turns", 13, "orange", "700")
    b.table(20, 410, [320, 320, 320, 300], [
        ["Strength", "", "Weakness", ""],
        ["Perfect recall in a session", "Zero implementation effort", "Token cost grows every turn", "Hard context ceiling"],
        ["Deterministic, easy to debug", "No loss or distortion", "No persistence across sessions", "No prioritising of facts"],
    ], "blue", size=11)
    return b


@board
def m3_sliding_window():
    b = Board(1300, 600, "2 · Sliding window memory (k = 4)", "Keep only the last k messages; older ones fall off the edge (3:12 to 3:30)")
    phases = [("Phase 1 · filling the window", ["msg1", "msg2", "—", "—"], "blue", "Hi, I'm Alice → reply"),
              ("Phase 2 · full: first eviction", ["msg2", "msg3", "msg4", "msg5"], "yellow", "msg5 'What's my name?'\n→ evict msg1, append msg5"),
              ("Phase 3 · the fact has slid out", ["msg4", "msg5", "msg6", "msg7"], "red", "'What language do I like?'\n→ 'I don't have that information.'")]
    for i, (t, slots, col, note) in enumerate(phases):
        y = 100 + i * 130
        b.group(20, y, 800, 115, t, col)
        for j, sl in enumerate(slots):
            b.card(50 + j * 120, y + 42, 105, 50, sl, [], "grey" if sl == "—" else col, title_size=13)
        b.text(545, y + 64, note, 11, col, "700", anchor="start")
    b.pill(545, 100 + 2 * 130 + 86, "'I like Python' (msg3) is gone", "red", solid=True)
    b.group(850, 100, 430, 370, "FinCoach, window = 3 turns", "orange")
    ex = [("T1", "salary ₹1,20,000", "grey"), ("T2", "expenses ₹60,000", "grey"), ("T3", "FD ₹50,000", "grey"),
          ("T4", "What about SIPs? → T1 evicted", "red"), ("T5", "'Could you remind me of your salary?'", "red")]
    for i, (t, txt, col) in enumerate(ex):
        y = 145 + i * 62
        b.pill(870, y + 12, t, "orange", solid=True)
        b.card(920, y, 340, 48, txt, [], col, title_size=11)
    b.card(20, 500, 1260, 80, "Cost is constant, not compounding · but context is lost",
           ["Production: almost always present, almost never alone. Pair it with a long-term layer (vector store, graph) that catches what slides out."],
           "purple", size=12)
    return b


@board
def m3_summary():
    b = Board(1300, 560, "3 · Summary memory", "Old messages are replaced by a short summary written by a second LLM (3:37 to 3:56)")
    u = b.card(30, 130, 150, 60, "User", [], "blue")
    buf = b.card(230, 110, 220, 100, "Message buffer", ["msg1 … msg4", "then: buffer full,", "trigger reached"], "blue")
    chat = b.card(230, 260, 220, 70, "Chat LLM", ["summary + messages"], "pink")
    z = b.card(540, 110, 260, 100, "Summariser LLM", ["summarise(old summary", "+ msg1..msg4)", "one call per cycle"], "red")
    st = b.card(540, 260, 260, 70, "Summary store", ['"Alice, likes Python, …"'], "green")
    b.arrow(u.right(), buf.left())
    b.arrow(buf.bottom(), chat.top())
    b.arrow(buf.right(), z.left(), label="compress", color="red")
    b.arrow(z.bottom(), st.top())
    b.arrow(st.left(), chat.right(), label="next turns start here")
    b.text(340, 360, "then: clear the buffer", 11, "grey", italic=True)
    b.group(850, 90, 420, 300, "Progressive (hierarchical) summarisation", "purple")
    lv = [("Level 0", "raw turns, verbatim", "most tokens"), ("Level 1", "rolling summary", "one paragraph per N turns"),
          ("Level 2", "session summary", "one paragraph per session"), ("Level 3", "user profile", "key facts, fewest tokens")]
    for i, (l, t, sub) in enumerate(lv):
        box = b.card(870 + i * 10, 130 + i * 62, 380 - i * 20, 50, f"{l} · {t}", [sub], "purple", size=11)
    b.card(30, 410, 610, 120, "Lossy compression is the fundamental risk",
           ["'User is allergic to equity instruments' survives", "the first summary and vanishes by the third.",
            "Low-frequency, high-importance facts go first."], "red", size=12)
    b.card(660, 410, 610, 120, "Mitigation",
           ["A domain-specific summarisation prompt that names", "what must survive: salary, risk profile,", "goals, constraints, decisions."], "green", size=12)
    return b


@board
def m3_summary_buffer():
    b = Board(1300, 470, "4 · Summary buffer memory", "Recent turns stay verbatim; older turns are summarised and carried forward (3:56 to 4:04)")
    um = b.card(30, 190, 150, 60, "User message", [], "purple")
    buf = b.card(230, 180, 200, 80, "Buffer", ["last K messages", "verbatim"], "blue")
    d = b.diamond(560, 220, 180, 110, "Buffer over\nthreshold?", "yellow")
    ev = b.card(470, 330, 180, 60, "Evict oldest", ["pop earliest messages"], "pink", size=11)
    z = b.card(700, 330, 220, 60, "Summariser LLM", ["old summary + evicted"], "green", size=11)
    st = b.card(970, 330, 200, 60, "Summary store", ["running summary"], "green", size=11)
    pa = b.card(760, 170, 220, 100, "Prompt assembly", ["[summary]", "+ [buffer]", "+ [new msg]"], "grey")
    llm = b.card(1030, 180, 110, 80, "LLM", [], "pink")
    r = b.card(1160, 180, 120, 80, "Response", ["back to buffer"], "purple", size=11)
    b.arrow(um.right(), buf.left())
    b.arrow(buf.right(), d.left())
    b.arrow(d.bottom(), ev.top(0.5), label="yes", color="red", via=[(560, 300), (ev.cx, 300)])
    b.arrow(d.right(), pa.left(), label="no", color="green")
    b.arrow(ev.right(), z.left())
    b.arrow(z.right(), st.left())
    b.arrow(st.top(), pa.bottom(0.8), via=[(st.cx, 300), (pa.x + pa.w * 0.8, 300)])
    b.arrow(pa.right(), llm.left())
    b.arrow(llm.right(), r.left())
    b.arrow(r.top(), buf.top(), via=[(r.cx, 140), (buf.cx, 140)], dashed=True, color="purple", label="appended back")
    b.text(650, 440, "The engineering knob is the transition threshold: where messages leave the buffer for the summary", 12, "grey", italic=True)
    return b


@board
def m3_token_buffer():
    b = Board(1300, 500, "5 · Token buffer memory", "Trim the conversation to a strict token budget; drop the oldest first (4:20 to 4:24)")
    nm = b.card(30, 160, 150, 60, "New message", [], "purple")
    tk = b.card(220, 150, 200, 80, "Tokenizer", ["tiktoken", "tokens per message"], "blue")
    tc = b.card(460, 150, 200, 80, "Token counter", ["total = Σ tokens"], "blue")
    d = b.diamond(800, 190, 190, 110, "total over\nmax_token_limit?", "yellow")
    ev = b.card(700, 300, 200, 60, "Evict oldest message", ["then re-count"], "red", size=11)
    th = b.card(960, 150, 150, 80, "Trimmed history", ["fits the budget"], "green", size=11)
    llm = b.card(1140, 150, 130, 80, "LLM", ["reply appended"], "pink", size=11)
    b.arrow(nm.right(), tk.left())
    b.arrow(tk.right(), tc.left())
    b.arrow(tc.right(), d.left())
    b.arrow(d.bottom(), ev.top(), label="yes", color="red")
    b.arrow(ev.left(), tc.bottom(), via=[(tc.cx, ev.cy)], dashed=True, color="red", label="re-count")
    b.arrow(d.right(), th.left(), label="no", color="green")
    b.arrow(th.right(), llm.left())
    b.card(30, 390, 600, 90, "input tokens = system + min(history, max buffer)",
           ["exact budget · no summariser calls · no latency spikes", "LangChain: ConversationTokenBufferMemory"], "green", size=11)
    b.card(660, 390, 610, 90, "vs sliding window: tokens, not turns",
           ["evicts one message at a time, so a user/assistant", "pair can be split; the budget is exact, not approximate"], "orange", size=11)
    return b


@board
def m3_vector_store():
    b = Board(1300, 560, "6 · Vector store memory", "Everything changes here: memory moves from RAM to a database (4:24 to 4:41)")
    b.table(30, 100, [300, 330], [
        ["Techniques 1–5", "Technique 6 onwards"],
        ["Memory lives in RAM", "Memory lives in a database"],
        ["Resets when the session ends", "Survives across sessions"],
        ["Retrieved by position (recency)", "Retrieved by semantic similarity"],
        ["One user, one buffer", "Multi-tenant: one store, many users"],
        ["A Python list", "A vector database (ChromaDB)"],
    ], "purple", size=12)
    q = b.card(700, 100, 560, 50, "User: 'Should I rebalance my portfolio?'", [], "blue", title_size=13)
    e = b.card(700, 170, 260, 60, "Embed the query", ["text-embedding-3-small"], "blue", size=11)
    db = b.cylinder(1000, 165, 260, 110, "ChromaDB", ["every past turn,", "every session"], "purple")
    r = b.card(700, 300, 560, 100, "Retrieved (top k = 3)", ["'I'm conservative with investments' · 3 months ago",
                                                             "'I have ₹2L in equity funds' · 2 sessions ago",
                                                             "'Buying a house in 2 years' · last session"], "green", size=11)
    a = b.card(700, 420, 560, 60, "FinCoach: personalised, cross-session advice", [], "pink", title_size=12)
    b.arrow(q.bottom(0.2), e.top())
    b.arrow(e.right(), (1000, 200), label="ANN search")
    b.arrow(db.bottom(), r.top(0.8))
    b.arrow(r.bottom(), a.top())
    b.card(30, 420, 630, 110, "The stale fact problem",
           ["Old and new values are both stored and both come back.", "Salary, job, health: facts that change need",
            "time awareness (temporal graph, e.g. Graphiti)."], "red", size=12)
    return b


@board
def m3_entity():
    b = Board(1300, 560, "7 · Entity memory", "Extract named entities and keep one structured record per entity (4:41 to 4:56)")
    b.group(20, 90, 420, 250, "Each conversation turn", "purple")
    um = b.card(40, 135, 380, 50, "User message", ["'I changed jobs to TCS'"], "purple", size=11)
    ex = b.card(40, 205, 380, 60, "Entity extractor", ["LLM call: return only stated facts as JSON"], "orange", size=11)
    b.arrow(um.bottom(), ex.top())
    store = b.card(480, 100, 380, 230, "Entity store (key-value)",
                   ['"chiru_001": {', '  "name": "Chiru",', '  "monthly_salary_inr": 120000,', '  "employer": "TCS",',
                    '  "risk_profile": "averse"', "}", "", "in-place update: new value replaces old"], "yellow",
                   size=11, align="left")
    b.arrow(ex.right(), store.left(0.55))
    b.group(900, 90, 380, 250, "Response generation", "teal")
    lk = b.card(920, 135, 340, 50, "Look up mentioned entities", [], "teal", title_size=12)
    bp = b.card(920, 200, 340, 50, "Build prompt", ["system + entity context + recent"], "teal", size=11)
    ll = b.card(920, 265, 340, 50, "LLM → response", [], "pink", title_size=12)
    b.arrow(store.right(0.3), lk.left())
    b.arrow(lk.bottom(), bp.top())
    b.arrow(bp.bottom(), ll.top())
    b.table(20, 370, [260, 330, 330], [
        ["", "Vector store (6)", "Entity memory (7)"],
        ["Storage unit", "raw message text", "structured key-value facts"],
        ["Retrieval", "semantic similarity", "direct key lookup"],
        ["Update model", "append-only", "in place: old value replaced"],
        ["Stale facts", "old and new both retrieved", "only the current value"],
    ], "yellow", size=11)
    b.card(980, 370, 300, 160, "NER in one line", ["'Barack Obama, 44th President", "of the USA, born in Honolulu'",
                                                   "→ PERSON · NUMBER · GPE", "", "Tesla the car vs Nikola Tesla"], "grey", size=11)
    return b


@board
def m3_hot_background():
    b = Board(1150, 380, "Memory updates: hot path vs background", "From the LangMem docs shown in the session (4:50 to 4:52)")
    b.group(20, 90, 540, 260, "In the hot path", "red")
    seq = ["User message", "Update memory", "Respond to user"]
    prev = None
    for i, t in enumerate(seq):
        box = b.card(45, 135 + i * 65, 490, 48, t, [], "red" if t == "Update memory" else "grey", title_size=13)
        if prev:
            b.arrow(prev.bottom(), box.top())
        prev = box
    b.text(290, 342, "'Call me Alex' → the reply already says 'Hi Alex' · costs latency", 11, "red", italic=True)
    b.group(590, 90, 540, 260, "In the background", "blue")
    a1 = b.card(615, 135, 230, 48, "User message", [], "grey", title_size=13)
    a2 = b.card(615, 200, 230, 48, "Respond to user", [], "grey", title_size=13)
    a3 = b.card(875, 265, 230, 48, "Update memory", ["separate process"], "blue", size=11)
    b.arrow(a1.bottom(), a2.top())
    b.arrow(a2.right(), a3.top(), via=[(990, a2.cy)], dashed=True, label="30 minutes later")
    b.text(860, 342, "no latency; the reply still says 'Chirantan' for a while", 11, "blue", italic=True)
    return b


@board
def m3_episodic():
    b = Board(1300, 470, "8 · Episodic memory", "Group turns into episodes; store each as a package; retrieve by time and topic (4:56 to 5:15)")
    cs = b.card(30, 110, 200, 70, "Conversation stream", ["incoming turns"], "purple")
    bd = b.diamond(380, 145, 210, 120, "Boundary?\ntopic shift · idle\nexplicit goodbye", "orange", size=11)
    pk = b.card(560, 100, 260, 90, "Episode packager", ["summary + title", "participants + topics", "metadata"], "blue", size=11)
    st = b.cylinder(880, 90, 380, 120, "Episode store", ["indexed by time and topic", "grows across sessions"], "purple")
    b.arrow(cs.right(), bd.left())
    b.arrow(bd.top(), cs.top(), via=[(380, 80), (cs.cx, 80)], dashed=True, color="red", label="continuing: keep listening")
    b.arrow(bd.right(), pk.left(), label="done")
    b.arrow(pk.right(), (880, 150))
    b.group(20, 250, 1260, 200, "At query time", "teal")
    nq = b.card(40, 300, 280, 70, "New query", ["'What did we decide", "about X last week?'"], "blue", size=11)
    re = b.card(360, 300, 260, 70, "Retrieval engine", ["time filter + semantic match"], "orange", size=11)
    ci = b.card(660, 300, 280, 70, "Context injection", ["episode summaries,", "not full transcripts"], "green", size=11)
    ll = b.card(980, 300, 280, 70, "LLM → response", ["grounded in the past"], "pink", size=11)
    b.arrow(nq.right(), re.left())
    b.arrow((1070, 210), re.top(0.8), via=[(1070, 240), (re.x + re.w * 0.8, 240)])
    b.arrow(re.right(), ci.left())
    b.arrow(ci.right(), ll.left())
    b.text(650, 425, "Immutable: episodes are an audit trail · generated at session end, asynchronously", 12, "teal", italic=True)
    return b


@board
def m3_semantic():
    b = Board(1300, 520, "9 · Semantic memory", "Distil standalone facts; deduplicate, resolve conflicts, persist (5:15 to 5:20)")
    cv = b.card(30, 140, 180, 70, "Conversation", ["or episodes"], "purple")
    fe = b.card(250, 140, 200, 70, "Fact extractor", ["standalone facts only"], "orange", size=11)
    dc = b.diamond(580, 175, 190, 110, "Dedup /\nconflict check", "yellow")
    outs = [("New fact → insert", "green"), ("Duplicate → merge, bump confidence", "teal"), ("Contradiction → prefer recent", "red")]
    kb = b.cylinder(1060, 110, 210, 140, "Knowledge base", ["facts + confidence", "+ timestamps"], "purple")
    for i, (t, col) in enumerate(outs):
        box = b.card(730, 95 + i * 62, 290, 48, t, [], col, title_size=11)
        b.arrow(dc.right(), box.left(), color=col)
        b.arrow(box.right(), (1060, 180), color=col)
    b.arrow(cv.right(), fe.left())
    b.arrow(fe.right(), dc.left())
    b.table(30, 310, [180, 520, 250, 290], [
        ["Type", "What it stores", "When was it true?", "Like"],
        ["Episodic", "'On June 12, Chiru was anxious and chose debt funds'", "on a specific date", "a diary entry"],
        ["Semantic", "'Chiru panics in volatility and needs reassurance'", "always", "an encyclopedia entry"],
    ], "purple", size=12)
    b.text(650, 470, "Episodic memory records what happened. Semantic memory keeps what it means.", 13, "purple", "700")
    return b


@board
def m3_procedural():
    b = Board(1300, 560, "10 · Procedural memory", "Learn reusable procedures from successful runs; reuse them on new tasks (5:20 to 5:25)")
    b.group(20, 90, 1260, 170, "Learning · from completed task to stored skill", "orange")
    te = b.card(40, 135, 220, 80, "Task execution", ["step-by-step trace"], "purple", size=11)
    ok = b.diamond(370, 175, 150, 100, "Succeeded?", "yellow")
    pe = b.card(500, 135, 220, 80, "Procedure extractor", ["LLM: trace →", "parameterised recipe"], "green", size=11)
    wt = b.card(760, 125, 230, 100, "Workflow template", ["1. {search query}", "2. filter by {date}", "3. summarise {topic}"], "yellow", size=11, align="left")
    sl = b.cylinder(1030, 120, 230, 110, "Skill library", ["indexed by task type"], "purple")
    b.arrow(te.right(), ok.left())
    b.arrow(ok.right(), pe.left(), label="yes", color="green")
    b.arrow(pe.right(), wt.left())
    b.arrow(wt.right(), (1030, 175))
    b.text(370, 250, "no → discard / log", 11, "red", "700")
    b.group(20, 280, 1260, 130, "Execution · reuse a stored skill", "teal")
    nt = b.card(40, 320, 300, 60, "New task", ["'Summarise last quarter's reports'"], "blue", size=11)
    rt = b.card(380, 320, 260, 60, "Retrieval", ["closest stored procedure"], "teal", size=11)
    ad = b.card(680, 320, 260, 60, "Adaptation", ["fill in the parameters"], "teal", size=11)
    ex = b.card(980, 320, 280, 60, "Execute procedure", ["skip re-planning"], "green", size=11)
    b.arrow(nt.right(), rt.left())
    b.arrow(rt.right(), ad.left())
    b.arrow(ad.right(), ex.left())
    b.table(20, 430, [220, 520, 520], [
        ["", "Semantic (9)", "Procedural (10)"],
        ["Stores", "facts about the user: 'Chiru is risk averse'", "rules for the agent: 'Always quantify the worst case'"],
        ["Lives in", "a user-facts block", "the system prompt, read as directives"],
    ], "teal", size=11)
    return b


@board
def m3_self_reflection():
    b = Board(1300, 500, "11 · Self-reflection memory", "After each task the agent reviews itself and stores the lesson (5:25 to 5:33)")
    b.group(20, 90, 380, 300, "Task attempt", "green")
    tp = b.card(40, 135, 340, 50, "Task prompt", [], "green", title_size=13)
    rr = b.card(40, 205, 340, 60, "Retrieve past reflections", ["'how did we handle this before?'"], "green", size=11)
    ae = b.card(40, 285, 340, 60, "Agent executes task", ["reflections in the system prompt"], "green", size=11)
    b.arrow(tp.bottom(), rr.top())
    b.arrow(rr.bottom(), ae.top())
    b.group(430, 90, 380, 300, "Outcome evaluation", "red")
    op = b.card(450, 135, 340, 50, "Output", [], "red", title_size=13)
    cm = b.card(450, 205, 340, 50, "Compare vs expected", [], "red", title_size=13)
    sd = b.diamond(620, 320, 170, 90, "Success?", "red")
    b.arrow(ae.right(), op.left(), via=[(415, ae.cy), (415, op.cy)])
    b.arrow(op.bottom(), cm.top())
    b.arrow(cm.bottom(), sd.top())
    b.group(840, 90, 440, 300, "Reflection loop", "orange")
    f = b.card(860, 135, 190, 70, "fail / partial", ["what went wrong?", "root cause?"], "red", size=11)
    s_ = b.card(1070, 135, 190, 70, "success", ["what worked?", "key strategy?"], "green", size=11)
    gi = b.card(860, 225, 400, 50, "Extract concise insight · 1–3 sentences", [], "orange", title_size=12)
    rs = b.card(860, 295, 400, 70, "Reflection store", ["(task_type, outcome, insight, timestamp)"], "yellow", size=11)
    b.arrow(sd.right(), f.left(), via=[(830, sd.cy), (830, f.cy)])
    b.arrow(f.bottom(), gi.top(0.2))
    b.arrow(s_.bottom(), gi.top(0.8))
    b.arrow(gi.bottom(), rs.top())
    b.arrow(rs.bottom(), rr.bottom(0.2), via=[(rs.cx, 420), (rr.x + rr.w * 0.2, 420)], dashed=True, color="purple",
            label="next task of the same type retrieves the lesson")
    b.text(650, 470, "Like a doctor's debrief after a hard case · Reflexion (2023): verbal reinforcement, no weight updates", 12, "grey", italic=True)
    return b


@board
def m3_routing():
    b = Board(1300, 560, "12 · Memory routing", "One classifier decides which store each message reads or writes (5:33 to 5:40)")
    ag = b.card(30, 220, 170, 70, "Agent", ["reads + writes"], "purple")
    rt = b.card(260, 205, 220, 100, "Memory router", ["classifier tags", "memory type", "routes read / write"], "teal")
    reg = b.card(260, 350, 220, 90, "Store registry", ["capabilities, schemas,", "access patterns"], "red", size=11)
    b.arrow(ag.right(), rt.left())
    b.arrow(reg.top(), rt.bottom(), dashed=True, color="purple")
    stores = [("Entity store", "structured current facts", "yellow"), ("Episodic store", "past sessions", "blue"),
              ("Semantic store", "behaviour patterns", "purple"), ("Procedural store", "hard constraints, rules", "orange"),
              ("Vector store", "general knowledge", "green")]
    for i, (t, sub, col) in enumerate(stores):
        box = b.card(560, 95 + i * 72, 240, 58, t, [sub], col, size=11)
        b.arrow(rt.right(), box.left(), color=col, width=1.4)
    b.table(840, 95, [260, 180], [
        ["Message", "Routed to"],
        ["'What is my current salary?'", "entity (read)"],
        ["'What did we decide last April?'", "episodic"],
        ["'How does an SIP work?'", "vector store"],
        ["'I just changed jobs to TCS'", "entity update + vector write"],
        ["'Never recommend equity again'", "procedural (constraint)"],
        ["'I'm worried about volatility'", "semantic"],
    ], "teal", size=11)
    b.card(30, 470, 1240, 70, "Without routing: 950 tokens on every turn",
           ["entity 150 + vector 200 + episodic 250 + semantic 200 + reflection 150, whatever the question"], "red", size=12)
    return b


@board
def m3_forgetting():
    b = Board(1300, 600, "13 · Forgetting and decay", "Memories weaken over time; reads reinforce them; weak ones are pruned (5:40 to 5:50)")
    de = b.card(30, 120, 200, 80, "Decay engine", ["R(t) = e^(−t/S)", "runs periodically"], "red")
    mems = [("Memory A", 0.92, "green"), ("Memory B", 0.45, "yellow"), ("Memory C", 0.08, "red")]
    b.group(280, 90, 520, 230, "Memory store", "purple")
    for i, (t, v, col) in enumerate(mems):
        y = 135 + i * 55
        b.text(300, y + 13, t, 13, col, "700", anchor="start")
        b.bar(410, y, 280, v, None, col, label=f"{v:.2f}")
    tx = 410 + 280 * 0.10
    b.parts.append(f'<line x1="{tx}" y1="125" x2="{tx}" y2="290" stroke="#e03131" stroke-width="2" stroke-dasharray="4 4"/>')
    b.text(tx, 305, "threshold 0.10", 11, "red", "700")
    b.arrow(de.right(), (280, 160), color="red")
    b.text(255, 182, "− strength", 11, "red", "700")
    rt = b.card(30, 230, 200, 70, "Retrieval", ["boost on access", "0.5 → 0.8"], "green")
    b.arrow(rt.right(), (280, 250), color="green")
    b.text(255, 280, "+ strength", 11, "green", "700")
    pe = b.card(850, 130, 200, 70, "Pruning engine", ["below threshold"], "orange")
    ar = b.cylinder(1090, 100, 180, 90, "Archive", ["soft delete"], "purple")
    dl = b.card(1090, 210, 180, 50, "Deleted", ["hard delete"], "pink", size=11)
    sp = b.card(850, 240, 200, 70, "Storage pressure", ["raises the threshold", "when full"], "pink", size=11)
    b.arrow((800, 250), pe.left(), label="C: 0.08")
    b.arrow(pe.right(0.3), (1090, 145), label="soft")
    b.arrow(pe.right(0.8), dl.left(), label="hard")
    b.arrow(sp.top(), pe.bottom(), dashed=True, color="purple")
    b.table(30, 340, [300, 330, 300, 310], [
        ["Strategy", "Rule", "Pro", "Con"],
        ["1 · TTL", "fixed expiry per memory", "simple, bounded storage", "critical facts age out"],
        ["2 · LRU", "evict least recently used", "keeps what's used (OS caches)", "rare-but-critical pruned"],
        ["3 · Importance-weighted", "f(recency, access, relevance, category)", "context-aware", "needs tuning"],
        ["4 · Budget-constrained", "hard cap; importance eviction", "bounded whatever happens", "budget chosen up front"],
    ], "red", size=11)
    b.text(650, 580, "Every read is a vote to keep; silence is a vote to forget · a 24-hour half-life halves an unused memory each day", 12, "grey", italic=True)
    return b


# --------------------------------------------------------------------- Module 4


@board
def m4_prototype_production():
    b = Board(1200, 420, "Your agent works on localhost. Now what?", "The opening slide of the AgentOps module (5:53)")
    b.card(30, 100, 540, 200, "🔬 Prototype reality", ["Jupyter notebook, hardcoded API keys", "One test query, one happy path",
                                                      "\"It works on my machine\"", "No concurrency, no failure modes"], "blue", size=13, bullets=True, align="left")
    b.card(630, 100, 540, 200, "🏭 Production reality", ["10,000 concurrent users", "Non-deterministic outputs at scale",
                                                        "Cascading agent failures", "No rollback plan"], "red", size=13, bullets=True, align="left")
    b.arrow((570, 200), (630, 200), width=3)
    b.card(30, 320, 1140, 70, "⚠ Traditional MLOps was designed for models that predict. Agents act. That changes everything.", [], "yellow", title_size=14)
    return b


@board
def m4_six_pillars():
    b = Board(1200, 480, "What is AgentOps?", "The operational discipline for deploying, scaling, observing and governing autonomous AI agents (7:20)")
    pillars = [("🚀 Deployment & orchestration", "GitHub Actions → EKS"), ("⚖ Scaling & reliability", "HPA · fallbacks · rollback"),
               ("⚙ Agentic CI/CD", "tests + golden dataset in CI"), ("🔗 A2A multi-agent coordination", "agent card + task endpoint"),
               ("👁 Observability & tracing", "Logfire + Langfuse"), ("🛡 Governance & guardrails", "Bedrock rails · HITL · audit trail")]
    for i, (t, sub) in enumerate(pillars):
        col, row = i % 2, i // 2
        b.card(40 + col * 570, 110 + row * 115, 550, 95, t, [sub], "purple", title_size=16, size=12)
    return b


@board
def m4_system_overview():
    b = Board(1400, 560, "arXiv Paper Curator · seven phases", "From raw infrastructure to a LangGraph agent (repository overview, 5:55)")
    b.group(20, 90, 420, 200, "Ingestion · Airflow, Mon–Fri 6 AM UTC", "orange")
    ax = b.card(40, 135, 120, 60, "arXiv API", ["cs.AI"], "orange", size=11)
    dl = b.card(180, 135, 120, 60, "Docling", ["parse PDFs"], "orange", size=11)
    pg = b.cylinder(320, 125, 100, 100, "Neon", ["metadata"], "green", size=11)
    b.arrow(ax.right(), dl.left())
    b.arrow(dl.right(), (320, 170))
    b.group(470, 90, 440, 200, "Indexing · phase 4", "purple")
    ch = b.card(490, 135, 130, 70, "Chunker", ["600w / 100w", "section-based"], "purple", size=11)
    ji = b.card(640, 135, 110, 70, "Jina", ["1024-dim"], "purple", size=11)
    os_ = b.cylinder(770, 125, 120, 100, "OpenSearch", ["BM25 + k-NN"], "blue", size=11)
    b.arrow((420, 170), ch.left())
    b.arrow(ch.right(), ji.left())
    b.arrow(ji.right(), (770, 170))
    b.group(940, 90, 440, 200, "Search · phases 3–4", "blue")
    b.card(960, 135, 190, 60, "BM25 query", ["text ×3, title ×2"], "blue", size=11)
    b.card(1170, 135, 190, 60, "k-NN query", ["cosine"], "blue", size=11)
    b.card(1060, 215, 200, 55, "RRF · k = 60", [], "teal", title_size=13)
    b.group(20, 320, 690, 220, "Serving · phases 5–6", "green")
    q = b.card(40, 365, 140, 60, "Query", [], "grey")
    rc = b.diamond(270, 395, 150, 90, "Redis\ncache?", "yellow")
    ans = b.card(540, 365, 150, 60, "Answer", ["+ sources"], "green")
    rag = b.card(360, 455, 200, 60, "RAG pipeline", ["gpt-4o-mini"], "pink", size=11)
    b.arrow(q.right(), rc.left())
    b.arrow(rc.right(), ans.left(), label="hit")
    b.arrow(rc.bottom(), rag.left(), via=[(270, rag.cy)], label="miss")
    b.arrow(rag.right(), ans.bottom(), via=[(ans.cx, rag.cy)])
    b.group(740, 320, 640, 220, "Agentic RAG · phase 7 + AgentOps", "red")
    for i, t in enumerate(["Guardrail", "Retrieve", "Grade", "Rewrite", "Generate", "Output check"]):
        b.card(760 + (i % 3) * 205, 365 + (i // 3) * 75, 190, 55, t, [], "red", title_size=13)
    b.text(1060, 530, "+ Langfuse traces · MCP server · Telegram · EKS", 12, "red", "700")
    return b


@board
def m4_phase1_infra():
    b = Board(1250, 480, "Phase 1 · infrastructure", "Four local containers; everything else in cloud free tiers (5:56 and 6:02)")
    b.group(20, 90, 600, 360, "Docker Compose · rag-network (bridge)", "blue")
    api = b.card(40, 140, 260, 70, "rag-api", ["FastAPI :8000"], "blue")
    af = b.card(340, 140, 260, 70, "rag-airflow", ["Airflow 2.10.3 :8080"], "orange")
    os_ = b.card(190, 270, 260, 70, "rag-opensearch", ["OpenSearch 2.19.5 :9200"], "purple")
    da = b.card(190, 370, 260, 60, "rag-dashboards", ["Dashboards :5601"], "purple", size=11)
    b.arrow(api.bottom(), os_.top(0.3), label="healthy")
    b.arrow(af.bottom(), os_.top(0.7), label="healthy")
    b.arrow(da.top(), os_.bottom())
    b.group(660, 90, 570, 360, "Cloud-managed services", "green")
    for i, (t, sub, col) in enumerate([("Neon", "serverless PostgreSQL 17", "green"), ("Upstash", "serverless Redis cache", "red"),
                                       ("Langfuse Cloud", "tracing", "teal"), ("OpenAI / Bedrock + Jina", "LLM + embeddings", "pink")]):
        box = b.card(690, 140 + i * 72, 510, 58, t, [sub], col, size=11)
    b.arrow(api.bottom(0.9), (690, 175), via=[(api.x + api.w * 0.9, 224), (640, 224), (640, 175)], label="SQLAlchemy",
            label_at=0.85)
    b.arrow(af.right(0.3), (690, 161), label="psycopg2")
    b.text(310, 470, "networks · health checks · volumes · port maps", 11, "blue", italic=True)
    return b


@board
def m4_langfuse_trace():
    b = Board(1250, 440, "One agentic request in Langfuse", "Trace-level view of 'What is vector policy?' (6:11 to 6:14)")
    t = b.card(30, 110, 330, 90, "Trace: agentic_rag_request", ["'What is vector policy?'", "≈ 25 s end to end"], "dark", size=12)
    lg = b.card(420, 120, 180, 70, "LangGraph", [], "purple", title_size=15)
    b.arrow(t.right(), lg.left())
    steps = [("guardrail", "Bedrock: passed all checks", "red"), ("retrieve", "decides to call the tool", "blue"),
             ("tools · retrieve_papers", "OpenSearch hybrid search", "blue"), ("grade_documents", "LLM scores each chunk", "yellow"),
             ("generate_answer", "gpt-4o-mini / Llama 3.1 70B", "green"), ("output_guardrail", "grounding + relevance", "red")]
    for i, (n, sub, col) in enumerate(steps):
        y = 90 + i * 55
        box = b.card(700, y, 380, 44, n, [sub], col, size=10, title_size=12)
        b.arrow(lg.right(), box.left(), color="grey", width=1.2)
    b.text(1160, 250, "each span:", 12, "grey", "700")
    b.text(1160, 270, "input · output", 12, "grey")
    b.text(1160, 290, "latency · tokens", 12, "grey")
    b.card(30, 250, 570, 150, "Human feedback on the trace", ["POST /api/v1/feedback", '{"trace_id": "…", "score": 0.9,', ' "comment": "helpful and accurate"}',
                                                            "internal annotators only · never end users"], "teal", size=11, align="left")
    return b


@board
def m4_bedrock_guardrails():
    b = Board(1300, 560, "AWS Bedrock Guardrails · four policies", "scripts/create_bedrock_guardrail.py, walked through at 6:19 to 6:21")
    q = b.card(30, 240, 150, 70, "User query", [], "blue")
    b.group(220, 90, 520, 440, "Input check (ApplyGuardrail, INPUT)", "red")
    pols = [("Topic denial · DENY", ["not CS / AI / ML research", "e.g. 'How do I cook pasta?'"], "red"),
            ("Content filters", ["HATE high · INSULTS medium · SEXUAL high", "VIOLENCE medium · MISCONDUCT high", "PROMPT_ATTACK high (input only)"], "red"),
            ("PII · ANONYMIZE", ["EMAIL · PHONE · NAME · ADDRESS", "→ redacted, request continues"], "yellow"),
            ("PII · BLOCK", ["credit/debit card · AWS keys"], "red")]
    y = 135
    for t, lines, col in pols:
        box = b.card(240, y, 480, None, t, lines, col, size=11)
        y += box.h + 14
    b.arrow(q.right(), (220, 275))
    rag = b.card(780, 240, 200, 70, "Retrieve · grade", ["generate"], "green")
    b.arrow((740, 275), rag.left(), label="allowed")
    b.group(1010, 90, 270, 440, "Output check", "purple")
    b.card(1030, 135, 230, 100, "Contextual grounding", ["GROUNDING ≥ 0.7", "answer supported", "by the chunks"], "purple", size=11)
    b.card(1030, 250, 230, 100, "Relevance", ["RELEVANCE ≥ 0.7", "answer addresses", "the question"], "purple", size=11)
    b.card(1030, 380, 230, 120, "Blocked messages", ["input: 'only CS, AI and", "ML research papers'", "output: 'not grounded", "in the papers'"], "grey", size=10)
    b.arrow(rag.right(), (1030, 250))
    return b


@board
def m4_dag():
    b = Board(1300, 300, "Airflow DAG · arxiv_paper_ingestion", "Monday to Friday at 6 AM UTC · graph view at 6:31")
    steps = [("setup_environment", ["check services"]), ("fetch_daily_papers", ["arXiv API → Docling", "→ PostgreSQL"]),
             ("index_papers_hybrid", ["chunk → Jina embed", "→ OpenSearch"]), ("generate_daily_report", ["counts, failures"]),
             ("cleanup_temp_files", ["delete PDFs > 30 days"])]
    prev = None
    for i, (t, lines) in enumerate(steps):
        box = b.card(30 + i * 252, 120, 222, 100, t, lines, "orange" if i in (1, 2) else "grey", size=11, title_size=12)
        if prev:
            b.arrow(prev.right(), box.left())
        prev = box
    b.text(650, 270, "retries: 2 · retry delay 30 min · max_active_runs 1 · catchup off", 12, "orange", italic=True)
    return b


@board
def m4_hybrid_search():
    b = Board(1300, 500, "Chunking and hybrid search", "Phases 3 and 4, from the workflow documents shown at 6:40 to 6:45")
    b.group(20, 90, 600, 380, "From paper to vectors", "purple")
    p = b.card(40, 135, 200, 60, "Parsed paper", ["Docling"], "purple")
    d = b.diamond(380, 165, 170, 90, "Sections\nfound?", "yellow")
    sb = b.card(250, 250, 170, 60, "Section-based", ["preferred"], "green", size=11)
    wb = b.card(440, 250, 160, 60, "Word fallback", ["600 w / 100 w"], "orange", size=11)
    em = b.card(40, 360, 270, 80, "Jina embeddings", ["batches of 100 · 1024-dim", "task = retrieval.passage"], "blue", size=11)
    ix = b.cylinder(340, 340, 260, 110, "arxiv-papers-chunks", ["text + vector + metadata"], "blue", size=11)
    b.arrow(p.right(), d.left())
    b.arrow(d.bottom(), sb.top(), label="yes")
    b.arrow(d.right(), wb.top(), via=[(wb.cx, d.cy)], label="no")
    b.arrow(sb.left(), em.top(0.5), via=[(175, sb.cy)])
    b.arrow(em.right(), (340, 395))
    b.group(650, 90, 630, 380, "Hybrid query", "blue")
    q = b.card(670, 165, 160, 60, "Query", [], "grey")
    bm = b.card(870, 130, 390, 58, "BM25 · keyword", ["chunk_text ×3 · title ×2 · abstract ×1"], "blue", size=11)
    kn = b.card(870, 205, 390, 58, "k-NN · dense", ["Jina query embedding · cosine"], "purple", size=11)
    rrf = b.card(870, 290, 390, 70, "RRF(d) = Σ 1 / (60 + rank)", ["ranks only: no score scaling needed"], "teal", size=11)
    tk = b.card(870, 390, 390, 58, "Top-k chunks + PDF URLs", [], "green", title_size=13)
    b.arrow(q.right(0.3), bm.left())
    b.arrow(q.right(0.7), kn.left())
    b.arrow(bm.bottom(0.15), rrf.top(0.15))
    b.arrow(kn.bottom(), rrf.top())
    b.arrow(rrf.bottom(), tk.top())
    return b


@board
def m4_rag_cache():
    b = Board(1250, 470, "RAG with an exact-match Redis cache", "Phases 5 and 6 (6:44 to 6:49)")
    c = b.card(30, 100, 150, 60, "Client", [], "grey")
    k = b.card(230, 90, 320, 80, "Cache key", ["SHA-256 of query, model, top_k,", "use_hybrid, categories → 16 hex"], "yellow", size=11)
    r = b.diamond(680, 130, 170, 100, "Upstash\nRedis?", "red")
    hit = b.card(880, 90, 340, 80, "Cached answer", ["200–300 ms"], "green", title_size=15)
    b.arrow(c.right(), k.left())
    b.arrow(k.right(), r.left())
    b.arrow(r.right(), hit.left(), label="hit", color="green")
    steps = [("Embed query", "Jina"), ("Hybrid search", "top-k + PDF URLs"), ("Prompt builder", "context assembly"),
             ("LLM", "20–30 s"), ("Store in Redis", "TTL 6 h")]
    prev = None
    for i, (t, sub) in enumerate(steps):
        box = b.card(30 + i * 240, 280, 220, 70, t, [sub], "pink" if t == "LLM" else "blue", size=11)
        if prev:
            b.arrow(prev.right(), box.left())
        prev = box
    b.arrow(r.bottom(), (140, 280), via=[(680, 240), (140, 240)], label="miss", color="red")
    b.arrow(prev.top(), hit.bottom(), via=[(prev.cx, 220), (hit.cx, 220)])
    b.card(30, 380, 1190, 70, "Exact match: change one character and it misses",
           ["semantic caching (similar questions) needs a Redis extension or a gateway such as Bifrost; not implemented here"], "orange", size=11)
    return b


@board
def m4_langgraph():
    b = Board(1300, 480, "Phase 7 · the LangGraph agent", "The graph shown in Langfuse and the repository (6:49 to 6:51)")
    s = b.card(30, 200, 90, 50, "start", [], "grey")
    g = b.diamond(230, 225, 170, 100, "guardrail\nBedrock", "red")
    o = b.card(150, 360, 170, 60, "out_of_scope", ["polite refusal"], "red", size=11)
    r = b.card(360, 195, 150, 60, "retrieve", ["tool call?"], "blue", size=11)
    t = b.card(560, 195, 170, 60, "tool_retrieve", ["OpenSearch"], "blue", size=11)
    gd = b.diamond(850, 225, 170, 100, "grade\ndocuments", "yellow")
    rw = b.card(560, 330, 180, 70, "rewrite_query", ["temperature 0.3", "max 2 attempts"], "orange", size=11)
    ga = b.card(990, 120, 150, 60, "generate", ["answer"], "green", size=11)
    og = b.card(990, 230, 150, 60, "output", ["guardrail"], "red", size=11)
    e = b.card(1180, 230, 90, 60, "end", [], "grey")
    b.arrow(s.right(), g.left())
    b.arrow(g.right(), r.left(), label="continue", color="green")
    b.arrow(g.bottom(), o.top(), label="out_of_scope", color="red")
    b.arrow(o.bottom(), e.bottom(), via=[(o.cx, 450), (1225, 450)])
    b.arrow(r.right(), t.left(), label="tools")
    b.arrow(t.right(), gd.left())
    b.arrow(gd.top(), ga.left(), via=[(850, ga.cy)], label="relevant", color="green")
    b.arrow(gd.bottom(), rw.right(), via=[(850, rw.cy)], label="not relevant", color="orange")
    b.arrow(rw.left(), r.bottom(), via=[(435, rw.cy)])
    b.arrow(ga.bottom(), og.top())
    b.arrow(og.right(), e.left())
    return b


@board
def m4_mcp():
    b = Board(1250, 470, "The whole app as an MCP server", "FastMCP mounted on the same FastAPI app at /mcp (6:53 to 7:02)")
    b.group(20, 90, 640, 350, "FastAPI app :8000", "blue")
    rest = b.card(40, 135, 280, 60, "/api/v1/* routes", ["REST"], "blue", size=11)
    mcp = b.card(360, 135, 280, 60, "/mcp · FastMCP", ["streamable HTTP, stateless"], "teal", size=11)
    svc = b.card(40, 240, 600, 70, "Shared services", ["agentic RAG · OpenSearch · Postgres · Langfuse"], "grey", size=11)
    b.arrow(rest.bottom(), svc.top(0.25))
    b.arrow(mcp.bottom(), svc.top(0.75))
    b.card(40, 340, 600, 80, "Six tools", ["ask_question · search_papers · get_paper_details", "list_recent_papers · submit_feedback · get_index_stats"], "teal", size=11)
    ins = b.card(720, 110, 500, 70, "MCP Inspector", ["npx @modelcontextprotocol/inspector · http://localhost:8000/mcp"], "purple", size=11)
    cl = b.card(720, 210, 500, 70, "Claude via mcp-remote", ["'Explain vector policy optimisation using the arXiv RAG tool'"], "purple", size=11)
    tg = b.card(720, 310, 500, 70, "Telegram bot", ["same agentic service · local only"], "orange", size=11)
    b.arrow(ins.left(), mcp.right(0.3), via=[(690, ins.cy), (690, 153)])
    b.arrow(cl.left(), mcp.right(0.7), via=[(690, cl.cy), (690, 177)])
    b.arrow(tg.left(), svc.right(), via=[(690, tg.cy), (690, svc.cy)])
    return b


@board
def m4_eks():
    b = Board(1300, 560, "Amazon EKS deployment", "Two m5.xlarge nodes, production namespace (7:08 to 7:18)")
    b.group(20, 90, 900, 400, "AWS · us-east-1", "orange")
    b.card(40, 130, 250, 70, "EKS control plane", ["managed by AWS · ~$73/month"], "grey", size=11)
    b.group(40, 220, 420, 210, "Node 1 · m5.xlarge · 4 vCPU / 16 GB", "blue")
    b.card(60, 270, 380, 60, "rag-api pod", ["requests 6 GiB · limit 8 GiB"], "blue", size=11)
    b.card(60, 350, 380, 60, "opensearch pod", ["StatefulSet + PVC"], "purple", size=11)
    b.group(480, 220, 420, 210, "Node 2 · m5.xlarge · 4 vCPU / 16 GB", "blue")
    b.card(500, 270, 380, 60, "rag-api pod", ["HPA: 2 → 6 replicas"], "blue", size=11)
    b.card(500, 350, 180, 60, "airflow pod", ["1 replica"], "orange", size=11)
    b.card(700, 350, 180, 60, "dashboards", [":5601"], "purple", size=11)
    b.card(310, 130, 280, 70, "Load balancers", ["API :80 · Airflow :8080 · :5601"], "green", size=11)
    b.card(610, 130, 290, 70, "ECR", ["agentic-rag/api · agentic-rag/airflow"], "yellow", size=11)
    b.group(950, 90, 330, 450, "Outside the cluster", "green")
    for i, (t, sub) in enumerate([("Neon", "PostgreSQL"), ("Upstash", "Redis cache"), ("Langfuse", "traces"),
                                  ("Jina", "embeddings"), ("Bedrock / OpenAI", "LLM + guardrails"), ("Grafana Cloud", "monitoring")]):
        b.card(970, 130 + i * 65, 290, 52, t, [sub], "green", size=11)
    b.text(470, 462, "IRSA: the API's service account assumes a Bedrock IAM role · no static AWS keys in pods", 11, "orange", "700")
    return b


@board
def m4_cicd():
    b = Board(1300, 360, "CI/CD with GitHub Actions", "Push → checks → images → EKS (7:18 to 7:19)")
    steps = [("git push", ["agentops / deployment"], "grey"), ("CI", ["ruff · mypy · pytest", "golden-dataset gate"], "blue"),
             ("Build + push", ["rag-api image", "rag-airflow image → ECR"], "yellow"),
             ("Deploy", ["namespace · Secret", "OpenSearch · DAGs", "API + HPA · Airflow"], "orange"),
             ("Rollout status", ["2/2 API replicas ready"], "green")]
    prev = None
    for i, (t, lines, col) in enumerate(steps):
        box = b.card(30 + i * 252, 110, 225, 130, t, lines, col, size=11, title_size=15)
        if prev:
            b.arrow(prev.right(), box.left())
        prev = box
    b.card(30, 265, 1240, 70, "The golden-dataset gate mocks every service",
           ["it checks the pipeline's structure for five questions, not answer quality · a full run takes 16–18 minutes"], "red", size=11)
    return b


@board
def m4_load_test():
    b = Board(1300, 520, "Load testing with Locust", "Three runs against /api/v1/ask-agentic; the HPA adds pods under load (7:21 to 7:40)")
    runs = [("10 users", 2, 2, "1%", "92 requests; two pods cope; CPU well under target", "green"),
            ("20 users", 2, 4, "5% → 1%", "CPU 63% → 89%; new pods start Pending, failures fall once they're up", "yellow"),
            ("50 users", 2, 6, "40% → 22%", "HPA hits its 6-pod max; some pods stay Pending; memory ~33 GB of 32", "red")]
    for i, (u, p0, p1, fail, note, col) in enumerate(runs):
        y = 100 + i * 120
        b.card(30, y, 150, 90, u, [], col, title_size=18)
        for k in range(6):
            filled = k < p1
            fill = "#1c7ed6" if k < p0 else ("#74c0fc" if filled else "#e9ecef")
            b.parts.append(f'<rect x="{210 + k * 46}" y="{y + 25}" width="38" height="40" rx="6" fill="{fill}" stroke="#adb5bd"/>')
        b.text(348, y + 85, f"pods: {p0} → {p1}", 11, "blue", "700")
        b.card(510, y, 160, 90, "failures", [fail], col, size=16)
        b.card(690, y, 580, 90, "", [note], col, size=12)
    b.card(30, 460, 1240, 50, "Bottlenecks: Bedrock ~20 concurrent · Jina ~100 · SQLAlchemy on Neon ~40–50 → for 10,000 users, self-host and redesign",
           [], "purple", title_size=12)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
