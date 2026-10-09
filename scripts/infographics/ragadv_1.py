"""Infographics for docs/genai/rag-advanced.

Run from the repo root:

    python3 scripts/infographics/ragadv_1.py            # all boards
    python3 scripts/infographics/ragadv_1.py graphrag-and-knowledge-graphs-pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "rag-adv"
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


def hbar(b, x, y, w, h, frac, color, label, value):
    rect(b, x, y, w, h, "#f1f3f5", "#ced4da", 1, 3)
    if frac > 0:
        rect(b, x, y, max(2, w * frac), h, PALETTE[color]["stroke"], PALETTE[color]["stroke"], 1, 3)
    raw_text(b, x - 10, y + h / 2 + 4, label, 12, INK, "end")
    raw_text(b, x + w + 10, y + h / 2 + 4, value, 12, PALETTE[color]["text"], "start", "700")


@board("graphrag-and-knowledge-graphs-pipeline")
def graphrag_pipeline():
    b = Board(1240, 720, "GraphRAG: build the graph once, ask two kinds of question", "Counts are from chapter code blocks 2, 4, 5 and 6 (24 documents)")
    b.group(20, 90, 1200, 190, "Index time: every step before the first question", "blue")
    d = b.card(40, 135, 200, 110, "24 documents", ["one sentence each,", "in four business areas"], "grey", size=12)
    e = b.card(280, 135, 220, 110, "extract", ["entities and relations", "an LLM call per chunk;", "a regex stands in here"], "orange", size=12)
    g = b.card(540, 135, 220, 110, "entity graph", ["22 entities", "28 relations", "each edge keeps its document"], "teal", size=12)
    c = b.card(800, 135, 200, 110, "communities", ["Louvain, seed 0:", "5 groups, sizes", "6, 5, 4, 4, 3"], "purple", size=12)
    r = b.card(1040, 135, 160, 110, "reports", ["one summary", "per community"], "pink", size=12)
    for a, z in ((d, e), (e, g), (g, c), (c, r)):
        b.arrow(a.right(), z.left())

    b.group(20, 310, 590, 380, "Local question", "green")
    q1 = b.card(40, 355, 550, 60, "Which regulator oversees the owner of Voltro?", [], "green", size=12)
    s1 = b.card(40, 440, 260, 90, "find the seed entity", ["Voltro is in the graph"], "green", size=12)
    s2 = b.card(330, 440, 260, 90, "follow the relations", ["owns (in), then", "regulates (in)"], "green", size=12)
    s3 = b.card(40, 570, 550, 90, "answer: Nordic Safety Board", ["supported by documents 2 and 3", "path following: 8 of 8 questions exact"], "green", size=12)
    b.arrow(q1.bottom(0.25), s1.top())
    b.arrow(s1.right(), s2.left())
    b.arrow(s2.bottom(), s3.top(0.75))
    b.arrow((320, 284), (320, 308), label="graph", color="green", label_dx=36)

    b.group(630, 310, 590, 380, "Global question", "purple")
    q2 = b.card(650, 355, 550, 60, "What are the main themes across these documents?", [], "purple", size=12)
    t1 = b.card(650, 440, 260, 90, "map", ["read each of the 5 reports,", "write a partial answer"], "purple", size=12)
    t2 = b.card(940, 440, 260, 90, "reduce", ["merge the partial answers", "into one"], "purple", size=12)
    t3 = b.card(650, 570, 550, 90, "touches 5 of 5 communities", ["vector top 5 touches 2 of 5", "cost: about 104 vector queries"], "purple", size=12)
    b.arrow(q2.bottom(0.25), t1.top())
    b.arrow(t1.right(), t2.left())
    b.arrow(t2.bottom(), t3.top(0.75))
    b.arrow((910, 284), (910, 308), label="reports", color="purple", label_dx=40)
    return b


@board("graphrag-and-knowledge-graphs-vector-vs-graph")
def graphrag_vs_vector():
    b = Board(1240, 700, "What the graph buys and what it costs", "Numbers printed by code blocks 3, 6 and 7")
    b.group(20, 90, 600, 290, "Local, multi-hop: questions fully supported (of 8)", "blue")
    ks = [2, 3, 4, 5, 6, 7, 8]
    vals = [0, 1, 2, 3, 4, 4, 6]
    for i, (k, v) in enumerate(zip(ks, vals)):
        hbar(b, 150, 128 + i * 28, 380, 20, v / 8, "blue", f"vector k={k}", f"{v}")
    hbar(b, 150, 128 + 7 * 28 + 4, 380, 20, 1.0, "green", "path following", "8")
    raw_text(b, 320, 374, "path following needs the question mapped to a relation path first", 11, FAINT)

    b.group(640, 90, 580, 290, "Global: communities touched (of 5)", "purple")
    hbar(b, 800, 140, 280, 24, 2 / 5, "blue", "vector top 3", "2")
    hbar(b, 800, 185, 280, 24, 2 / 5, "blue", "vector top 5", "2")
    hbar(b, 800, 230, 280, 24, 3 / 5, "blue", "vector top 8", "3")
    hbar(b, 800, 275, 280, 24, 1.0, "purple", "all 5 reports", "5")
    raw_text(b, 930, 350, "a themes question has no passage to match", 11, FAINT)

    b.group(20, 410, 1200, 270, "Cost in units, on synthetic prices (1.0 in, 4.0 out per million tokens; 1,000,000-token corpus)", "orange")
    rows = [["", "index", "one local query", "one global query"],
            ["vector RAG", "0.00 generation", "0.0056", "0.0056 (no passage to match)"],
            ["GraphRAG", "9.33 (7.33 + 2.00)", "0.0071", "0.5821"],
            ["ratio", "1,666 vector queries", "1.3 x", "104 x"]]
    b.table(50, 455, [200, 300, 300, 350], rows, "orange", size=14, row_h=44)
    raw_text(b, 620, 660, "Parameters: 600-token chunks, 1 gleaning pass, 400 communities, all editable in block 7", 12, FAINT)
    return b


@board("graphrag-and-knowledge-graphs-worked-example")
def graphrag_worked():
    b = Board(1240, 640, "Worked example: follow two relations from Voltro", "Three of the 24 sentences; the same walk as chapter code block 4")
    b.group(20, 90, 1200, 230, "The graph the three sentences make (each edge remembers its document)", "teal")
    n4 = b.card(50, 150, 200, 90, "Nordic Safety Board", ["regulator"], "pink", size=12)
    n3 = b.card(370, 150, 200, 90, "Helion Group", ["owner"], "orange", size=12)
    n2 = b.card(690, 150, 200, 90, "Voltro", ["seed entity"], "green", size=12)
    n1 = b.card(1010, 150, 200, 90, "Kestrel Motors", ["not needed"], "grey", size=12)
    b.arrow(n4.right(), n3.left(), label="regulates\ndoc 3", color="pink")
    b.arrow(n3.right(), n2.left(), label="owns\ndoc 2", color="orange")
    b.arrow(n2.right(), n1.left(), label="supplies\ndoc 1", color="grey", dashed=True)
    raw_text(b, 620, 290, "dashed edge: never touched, so document 1 is never read", 12, FAINT)

    b.group(20, 350, 1200, 270, "The walk, backwards along the arrows", "green")
    s1 = b.card(50, 395, 260, 100, "1. seed", ["start at Voltro", "named in the question"], "green", size=12)
    s2 = b.card(360, 395, 260, 100, "2. hop 1: owns (in)", ["who points at Voltro", "with owns?", "Helion Group, doc 2"], "orange", size=12)
    s3 = b.card(670, 395, 260, 100, "3. hop 2: regulates (in)", ["who points at Helion", "with regulates?", "Nordic Safety Board, doc 3"], "pink", size=12)
    s4 = b.card(980, 395, 200, 100, "answer", ["Nordic Safety Board", "evidence: docs 2, 3"], "teal", size=12)
    for a, z in ((s1, s2), (s2, s3), (s3, s4)):
        b.arrow(a.right(), z.left())
    raw_text(b, 620, 560, "vector top 2 for this question: doc 2 (rank 1) and a wrong one; doc 3 ranks 3rd", 12, FAINT)
    raw_text(b, 620, 585, "path following reads 2 documents and gets both", 12, "#2b8a3e", "middle", "700")
    return b


@board("contextual-retrieval-and-reranking-idea")
def contextual_idea():
    b = Board(1240, 700, "Two repairs for retrieval: restore each chunk's context, then rescore the shortlist", "Numbers from chapter code blocks 1, 5 and 9")
    b.group(20, 90, 1200, 250, "Repair 1, at index time: a chunk should say where it came from", "orange")
    c1 = b.card(40, 135, 330, 110, "chunk as cut", ["Revenue grew by 12% over the", "previous quarter."], "grey", size=12)
    c2 = b.card(450, 135, 330, 110, "add context (once per chunk)", ["a model, or here a template, writes:", "Brightwell Energy, Q3 2025 report."], "orange", size=12)
    c3 = b.card(860, 135, 330, 110, "chunk as indexed", ["Brightwell Energy, Q3 2025", "report. Revenue grew by 12%", "over the previous quarter."], "green", size=12)
    b.arrow(c1.right(), c2.left())
    b.arrow(c2.right(), c3.left())
    raw_text(b, 620, 285, "plain chunks: hit@1 0.042 (BM25), 0.050 (dense)", 12, FAINT)
    raw_text(b, 620, 310, "with context: hit@1 1.000 (BM25), 0.742 (dense)", 13, "#2b8a3e", "middle", "700")

    b.group(20, 370, 1200, 310, "Repair 2, at query time: a slower, sharper model rescores a short list", "purple")
    s1 = b.card(40, 420, 210, 100, "first stage", ["BM25 + dense,", "fused (RRF)", "top 100"], "blue", size=12)
    s2 = b.card(310, 420, 210, 100, "shortlist", ["top 20 sent on", "its recall is the", "ceiling for what follows"], "teal", size=12)
    s3 = b.card(580, 420, 250, 100, "cross-encoder", ["reads query and", "document together", "7.7 ms per pair"], "purple", size=12)
    s4 = b.card(890, 420, 300, 100, "top 5 for the answer", ["order can change,", "membership cannot", "(shortlist is a ceiling)"], "pink", size=12)
    for a, z in ((s1, s2), (s2, s3), (s3, s4)):
        b.arrow(a.right(), z.left())
    raw_text(b, 620, 575, "hybrid nDCG@10 0.691  |  general reranker 0.682  |  reranker fine-tuned on 3,346 labelled pairs 0.705", 12, "#5f3dc4", "middle", "700")
    raw_text(b, 620, 605, "a reranker trained on someone else's data can hurt; measure before you ship", 12, FAINT)
    return b


@board("contextual-retrieval-and-reranking-worked-example")
def contextual_worked():
    b = Board(1240, 640, "Worked example: why a bare chunk cannot be found", "Counting query words that appear in each chunk; the chapter code measures the same effect on 120 chunks")
    b.group(20, 90, 1200, 130, "Query", "blue")
    b.card(60, 130, 1120, 60, "How fast did revenue grow at Brightwell Energy in Q3 2025?", [], "blue", size=13)
    b.group(20, 245, 590, 370, "Plain chunks: 24 chunks that all say much the same", "grey")
    rows = [["chunk text", "shared words"],
            ["Revenue grew by 12% ...", "1 (revenue)"],
            ["Revenue grew by 31% ...", "1 (revenue)"],
            ["Revenue grew by 7% ...", "1 (revenue)"],
            ["... 21 more, same pattern", "1 each"]]
    b.table(45, 295, [330, 220], rows, "grey", size=12, row_h=34)
    raw_text(b, 315, 520, "24 chunks tie, so the right one is a coin toss", 13, "#343a40", "middle", "700")
    raw_text(b, 315, 548, "hit@1 = 1 in 24 = 0.042", 13, "#c92a2a", "middle", "700")
    b.group(630, 245, 590, 370, "Chunks with context prefixed", "green")
    rows2 = [["chunk text", "shared words"],
            ["Brightwell Energy, Q3 2025. Revenue ...", "5  (right one)"],
            ["Brightwell Energy, Q2 2025. Revenue ...", "4"],
            ["Eskarn Telecom, Q3 2025. Revenue ...", "3"],
            ["Aldermoor Foods, Q1 2025. Revenue ...", "2"]]
    b.table(655, 295, [380, 180], rows2, "green", size=12, row_h=34)
    raw_text(b, 925, 520, "the right chunk wins with no ties", 13, "#2b8a3e", "middle", "700")
    raw_text(b, 925, 548, "BM25 hit@1 = 1.000 (block 1)", 13, "#2b8a3e", "middle", "700")
    return b


@board("text-to-sql-and-structured-rag-pipeline")
def sql_pipeline():
    b = Board(1240, 720, "Text-to-SQL: link, generate, guard, grade", "Numbers from chapter code blocks 2 to 6")
    b.group(20, 90, 1200, 260, "One question, left to right", "blue")
    q = b.card(40, 140, 170, 110, "question", ["How many customers", "live in Lyon?"], "grey", size=12)
    l = b.card(250, 140, 190, 110, "1. link schema", ["31 columns in,", "about 12 shown", "recall 0.979"], "teal", size=12)
    m = b.card(480, 140, 190, 110, "2. model writes SQL", ["greedy decoding,", "SQL only"], "purple", size=12)
    g = b.card(710, 140, 190, 110, "3. guard", ["one SELECT, read-only", "file, authorizer, timer"], "red", size=12)
    d = b.card(940, 140, 240, 110, "4. database", ["rows come back", "compared with the gold", "rows on two databases"], "green", size=12)
    for a, z in ((q, l), (l, m), (m, g), (g, d)):
        b.arrow(a.right(), z.left())
    b.arrow((805, 250), (575, 300), via=[(805, 300)], label="rejected: error text goes back", color="orange", dashed=True)
    b.arrow((575, 300), (575, 252), color="orange", dashed=True)

    b.group(20, 380, 590, 310, "Three layers in the guard", "red")
    b.card(40, 425, 550, 70, "statement check", ["one SELECT or WITH, no second statement"], "red", size=12)
    b.card(40, 510, 550, 70, "read-only file + authorizer", ["writes fail; ATTACH ran through the read-only file alone"], "red", size=12)
    b.card(40, 595, 550, 70, "progress handler", ["aborts the endless recursive query after 0.5 s"], "red", size=12)

    b.group(630, 380, 590, 310, "Why grade on two databases", "green")
    b.card(650, 425, 550, 70, "database 1", ["gold and generated query return the same rows"], "green", size=12)
    b.card(650, 510, 550, 70, "database 2: seven shipped orders changed", ["a lucky wrong query stops matching"], "green", size=12)
    raw_text(b, 925, 625, "robust = matches on both", 13, "#2b8a3e", "middle", "700")
    return b


@board("text-to-sql-and-structured-rag-worked-example")
def sql_worked():
    b = Board(1240, 600, "Worked example: how many customers live in Lyon?", "40 customers, cities cycling Lyon, Porto, Graz, Leeds, Turin by id")
    c1 = b.card(30, 110, 260, 130, "1. link", ["customers.city", "\"city where the", "customer lives\""], "teal", size=12)
    c2 = b.card(340, 110, 260, 130, "2. prompt", ["customers(customer_id,", "name, city, signup_date)", "Question: ..."], "blue", size=12)
    c3 = b.card(650, 110, 270, 130, "3. query", ["SELECT COUNT(*)", "FROM customers", "WHERE city = 'Lyon'"], "purple", size=12)
    c4 = b.card(970, 110, 240, 130, "4. guard", ["starts with SELECT", "one statement", "reads only: allowed"], "red", size=12)
    for a, z in ((c1, c2), (c2, c3), (c3, c4)):
        b.arrow(a.right(), z.left())
    b.group(30, 290, 1180, 280, "Run and grade", "green")
    b.card(60, 340, 340, 170, "ids that are Lyon", ["5, 10, 15, 20,", "25, 30, 35, 40", "8 customers"], "green", size=13)
    b.card(450, 340, 340, 170, "database returns", ["[(8,)]"], "green", size=16)
    b.card(840, 340, 340, 170, "gold query returns", ["[(8,)]", "execution match", "text need not match"], "green", size=13)
    b.arrow((400, 425), (450, 425))
    b.arrow((790, 425), (840, 425))
    raw_text(b, 620, 545, "COUNT(DISTINCT customer_id) would also return 8: grade the rows, not the text", 12, FAINT)
    return b


@board("long-context-vs-rag-decision")
def lc_decision():
    b = Board(1240, 680, "Long context or RAG: three questions, in order", "Cost figures are from chapter code block 1 (Claude Sonnet 5.5 list prices, 7 October 2026)")
    d1 = b.diamond(250, 190, 330, 150, "does the corpus fit\nthe window?", "yellow", size=13)
    r0 = b.card(520, 140, 290, 100, "RAG only", ["1,200,000 tokens does not", "fit in 1,000,000"], "green", size=12)
    d2 = b.diamond(250, 420, 330, 150, "is it asked more often\nthan every 5 minutes?", "yellow", size=13)
    b.arrow(d1.right(), r0.left(), label="no", color="green")
    b.arrow(d1.bottom(), d2.top(), label="yes", color="blue")
    r1 = b.card(520, 370, 290, 100, "cache stays warm", ["500,000 tokens: 0.1032 per", "question, 9.2 x RAG"], "blue", size=12)
    r2 = b.card(520, 500, 290, 100, "cache goes cold", ["long context costs more", "with a cache: 1.25 vs 1.00 an hour"], "red", size=12)
    b.arrow(d2.right(), r1.left(), label="yes", color="blue")
    b.arrow(d2.bottom(), r2.left(), via=[(250, 550)], label="no", color="red")
    d3 = b.diamond(1020, 420, 330, 150, "does every question\nneed the whole corpus?", "yellow", size=13)
    b.arrow(r1.right(), d3.left())
    s1 = b.card(920, 140, 290, 100, "long context", ["summaries, comparisons", "across the corpus"], "purple", size=12)
    s2 = b.card(920, 540, 290, 100, "RAG first, long context", ["as the fallback", "0.2118 per question at 80% by RAG"], "teal", size=12)
    b.arrow(d3.top(), s1.bottom(), label="yes", color="purple")
    b.arrow(d3.bottom(), s2.top(), label="no", color="teal")
    return b


@board("long-context-vs-rag-worked-example")
def lc_worked():
    b = Board(1240, 600, "Worked example: one question about a 500,000-token knowledge base", "100 tokens in, 300 tokens out, 4,000 tokens of retrieved chunks for RAG")
    rows = [
        ("long context, no cache", 1.0032, "orange", "1.0032"),
        ("long context, cache hit", 0.1032, "purple", "0.1032"),
        ("RAG", 0.0112, "green", "0.0112"),
    ]
    b.group(20, 90, 1200, 280, "Dollars per question", "blue")
    for i, (label, value, color, text) in enumerate(rows):
        y = 140 + i * 68
        hbar(b, 280, y, 700, 40, max(value / 1.0032, 0.004), color, label, text)
    raw_text(b, 620, 345, "bar length is to scale: RAG is too short to see at this width", 12, FAINT)
    b.group(20, 400, 1200, 180, "Where the numbers come from", "grey")
    b.card(40, 445, 380, 110, "no cache", ["500,100 x 2.00 / 1,000,000", "= 1.0002 + 0.003 output"], "orange", size=12)
    b.card(430, 445, 380, 110, "cache hit", ["500,000 x 0.20 / 1,000,000 = 0.10", "+ 0.0002 + 0.003 output"], "purple", size=12)
    b.card(820, 445, 380, 110, "RAG", ["4,100 x 2.00 / 1,000,000 = 0.0082", "+ 0.003 output"], "green", size=12)
    return b


@board("contextual-retrieval-and-reranking-results")
def contextual_results():
    b = Board(1240, 640, "What each repair bought on this data", "Numbers printed by chapter code blocks 1, 2, 3, 5, 6 and 9")
    b.group(20, 90, 600, 250, "Reranking on SciFact: nDCG@10 over 300 test claims", "purple")
    for i, (label, value, color) in enumerate([("hybrid, no reranker", 0.691, "blue"), ("general reranker, top 20", 0.682, "red"), ("fine-tuned reranker, top 20", 0.702, "green")]):
        hbar(b, 250, 140 + i * 50, 300, 28, value / 0.8, color, label, f"{value:.3f}")
    raw_text(b, 320, 310, "bars start at zero; the differences are small, which is the point", 11, FAINT)
    b.group(640, 90, 580, 250, "Context on chunks", "orange")
    for i, (label, value, color, text) in enumerate([("synthetic, BM25 hit@1: bare", 0.042, "grey", "0.042"), ("synthetic, BM25 hit@1: context", 1.0, "green", "1.000")]):
        hbar(b, 910, 140 + i * 50, 200, 28, value, color, label, text)
    for i, (label, value, color) in enumerate([("real, nDCG@10: bare chunk", 0.670, "grey"), ("real: title prefix", 0.687, "green"), ("real: late chunking", 0.659, "yellow")]):
        hbar(b, 910, 245 + i * 30, 200, 20, value, color, label, f"{value:.3f}")
    b.group(20, 370, 1200, 250, "What a deeper shortlist buys (general reranker)", "teal")
    rows = [["depth", "5", "10", "20", "30", "50"],
            ["nDCG@10", "0.691", "0.684", "0.682", "0.675", "0.684"],
            ["relevant in shortlist", "0.754", "0.817", "0.879", "0.898", "0.937"]]
    b.table(80, 420, [300, 150, 150, 150, 150, 150], rows, "teal", size=13, row_h=44)
    raw_text(b, 620, 590, "a deeper shortlist raised the ceiling; this reranker could not use it", 13, "#0b7285", "middle", "700")
    return b


@board("text-to-sql-and-structured-rag-results")
def sql_results():
    b = Board(1240, 640, "What reached the prompt, and what the model did with it", "Numbers printed by chapter code blocks 2, 5 and 6")
    b.group(20, 90, 600, 330, "Schema linking: mean recall of needed columns", "teal")
    rows = [("top-5, name only", 0.897, "9/12"), ("top-5, with description", 0.761, "6/12"), ("top-8, with description", 0.931, "10/12"), ("top-5 + key columns", 0.854, "9/12"), ("top-8 + key columns", 0.979, "11/12")]
    for i, (label, value, found) in enumerate(rows):
        hbar(b, 260, 135 + i * 50, 250, 26, value, "teal", label, f"{value:.3f}  {found}")
    raw_text(b, 320, 395, "found = questions where nothing needed was lost", 11, FAINT)
    b.group(640, 90, 580, 330, "SmolLM2-1.7B, 11 answerable questions", "purple")
    for i, (label, ran, right, color) in enumerate([("no schema", 3, 2, "grey"), ("linked schema", 7, 5, "blue"), ("full schema", 9, 7, "green")]):
        hbar(b, 800, 140 + i * 70, 300, 24, ran / 11, color, f"{label}: runs", f"{ran}")
        hbar(b, 800, 170 + i * 70, 300, 24, right / 11, "purple", "returns right rows", f"{right}")
    raw_text(b, 930, 395, "delete request blocked in every condition", 12, "#2b8a3e", "middle", "700")
    b.group(20, 450, 1200, 170, "One retry with the error message, linked and full schema", "orange")
    b.card(60, 495, 520, 90, "linked schema", ["4 rejected, 0 fixed by one retry"], "orange", size=13)
    b.card(640, 495, 520, 90, "full schema", ["2 rejected, 0 fixed by one retry"], "orange", size=13)
    return b


@board("long-context-vs-rag-needle-and-cost")
def lc_needle():
    b = Board(1240, 660, "What the small model found, and what each prompt costs the machine", "SmolLM2-360M-Instruct on CPU, 32-bit floats; numbers from chapter code blocks 3, 4 and 5")
    b.group(20, 90, 640, 340, "Needle recovered out of 6, whole prompt (8 look-alike sentences)", "blue")
    rows = [["tokens", "start", "middle", "end", "RAG, top 3"],
            ["512", "6", "5", "6", "3"],
            ["2,048", "6", "4", "6", "5"],
            ["4,096", "6", "4", "6", "6"]]
    b.table(45, 140, [130, 100, 110, 100, 150], rows, "blue", size=14, row_h=44)
    raw_text(b, 340, 360, "middle is the weak spot; no collapse with length", 13, "#1864ab", "middle", "700")
    raw_text(b, 340, 392, "same needle, question reworded: 4 of 6 becomes 2 of 6", 13, "#c92a2a", "middle", "700")
    b.group(680, 90, 540, 340, "Cost of reading the prompt", "teal")
    rows2 = [["tokens", "prefill s", "KV cache MB"],
             ["512", "0.89", "21.0"],
             ["1,024", "1.36", "41.9"],
             ["2,048", "2.59", "83.9"],
             ["4,096", "4.71", "167.8"]]
    b.table(705, 140, [150, 170, 190], rows2, "teal", size=14, row_h=44)
    raw_text(b, 950, 400, "1,000,000 tokens would need 41.0 GB of KV cache", 12, "#0b7285", "middle", "700")
    b.group(20, 460, 1200, 170, "Reading it", "grey")
    b.card(45, 505, 370, 100, "whole prompt", ["reads 512 to 4,096 tokens", "finds the needle 16 to 17 times in 18"], "blue", size=12)
    b.card(435, 505, 370, 100, "RAG", ["reads about 215 tokens", "3 of 6 at 512, 6 of 6 at 4,096"], "green", size=12)
    b.card(825, 505, 370, 100, "so", ["accuracy was not the gap here", "tokens and time were"], "orange", size=12)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or v.endswith(name))
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
