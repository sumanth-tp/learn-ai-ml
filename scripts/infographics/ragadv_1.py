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
    b = Board(1240, 720, "GraphRAG: build the graph once, ask two kinds of question", "Counts are from chapter code blocks 1 and 2 (24 documents)")
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
    b = Board(1240, 700, "What the graph buys and what it costs", "Numbers printed by code blocks 1, 2 and 3")
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
    raw_text(b, 620, 660, "Parameters: 600-token chunks, 1 gleaning pass, 400 communities, all editable in block 3", 12, FAINT)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or v.endswith(name))
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
