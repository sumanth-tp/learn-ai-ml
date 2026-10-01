"""Infographics for docs/projects/enterprise-rag/02-session-2.md.

Each function redraws one whiteboard, poster, vendor figure or README diagram
shown in Session 2 (video jOgqWdck7BU) as an original board-style image. The
content, labels, grouping and flow follow the session; the drawing is ours.
Run from the repo root:

    python3 scripts/infographics/enterprise_rag_s2.py              # all boards
    python3 scripts/infographics/enterprise_rag_s2.py s2_gateway   # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import FAINT, INK, MONO, PALETTE, SANS, Board, Box, esc  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "enterprise-rag"
BOARDS = {}


def board(fn):
    BOARDS[fn.__name__] = fn
    return fn


# ------------------------------------------------------------------ local helpers


def raw(b: Board, svg: str):
    b.parts.append(svg)


def stroke(color: str) -> str:
    return PALETTE[color]["stroke"] if color in PALETTE else color


def circled(b: Board, cx, cy, text: str, color: str = "red", r: float = 16, size: int = 15):
    c = PALETTE[color]
    raw(b, f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="2"/>')
    b.text(cx, cy + size * 0.36, text, size, color, "700")
    return Box(cx - r, cy - r, 2 * r, 2 * r)


def mark(b: Board, x, y, ok: bool = True, size: int = 18):
    """A tick or a cross, drawn as paths so no glyph fallback is needed."""
    col = PALETTE["green" if ok else "red"]["stroke"]
    s = size
    if ok:
        d = f"M{x - s * 0.4},{y} L{x - s * 0.1},{y + s * 0.32} L{x + s * 0.45},{y - s * 0.38}"
    else:
        d = (f"M{x - s * 0.35},{y - s * 0.35} L{x + s * 0.35},{y + s * 0.35} "
             f"M{x + s * 0.35},{y - s * 0.35} L{x - s * 0.35},{y + s * 0.35}")
    raw(b, f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{max(2.4, s / 6):.1f}" '
           f'stroke-linecap="round" stroke-linejoin="round"/>')


def underline(b: Board, x1, x2, y, color: str = "yellow", double: bool = False):
    col = stroke(color)
    raw(b, f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{col}" stroke-width="2" stroke-linecap="round"/>')
    if double:
        raw(b, f'<line x1="{x1 + 6}" y1="{y + 5}" x2="{x2 - 6}" y2="{y + 5}" stroke="{col}" '
               f'stroke-width="2" stroke-linecap="round"/>')


def graph_icon(b: Board, x, y, color: str = "yellow", s: float = 1.0):
    """A small LangGraph-style node graph: top node, branch, looped middle node, bottom node."""
    c = PALETTE[color]
    col, fill = c["stroke"], c["fill"]
    pts = {"a": (x + 30 * s, y + 14 * s), "b": (x + 44 * s, y + 62 * s), "c": (x + 78 * s, y + 102 * s),
           "l1": (x + 4 * s, y + 52 * s), "l2": (x + 14 * s, y + 100 * s)}
    edges = [("a", "b"), ("b", "c"), ("a", "l1"), ("b", "l2")]
    for p, q in edges:
        (x1, y1), (x2, y2) = pts[p], pts[q]
        raw(b, f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{2 * s}"/>')
    for k in ("a", "b", "c"):
        cx, cy = pts[k]
        raw(b, f'<circle cx="{cx}" cy="{cy}" r="{11 * s}" fill="{fill}" stroke="{col}" stroke-width="{2.2 * s}"/>')
    ax, ay = pts["a"]
    raw(b, f'<path d="M{ax + 16 * s},{ay + 4 * s} C{ax + 70 * s},{ay + 10 * s} {ax + 72 * s},{ay + 60 * s} '
           f'{ax + 34 * s},{ay + 56 * s}" fill="none" stroke="{col}" stroke-width="{2 * s}" '
           f'marker-end="url(#{b._marker(col)})"/>')
    return Box(x, y, 110 * s, 116 * s)


def chunk_box(b: Board, x, y, w, h, color: str = "grey", label: str = "", size: int = 11, fill=None):
    c = PALETTE[color]
    raw(b, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill or c["fill"]}" '
           f'stroke="{c["stroke"]}" stroke-width="1.5"/>')
    for i in range(3):
        ly = y + 9 + i * 7
        if ly < y + h - 5 and not label:
            raw(b, f'<line x1="{x + 8}" y1="{ly}" x2="{x + w - 8 - i * 9}" y2="{ly}" stroke="{c["stroke"]}" '
                   f'stroke-opacity="0.45" stroke-width="2"/>')
    if label:
        b.text(x + w / 2, y + h / 2 + size * 0.36, label, size, color, "700")
    return Box(x, y, w, h)


def doc_stack(b: Board, x, y, color: str = "grey", n: int = 3, w: float = 54, h: float = 66,
              label: str = ""):
    c = PALETTE[color]
    for i in range(n - 1, -1, -1):
        dx, dy = i * 7, -i * 7
        raw(b, f'<path d="M{x + dx},{y + dy} h{w - 14} l14,14 v{h - 14} h{-w} z" fill="{c["fill"]}" '
               f'stroke="{c["stroke"]}" stroke-width="1.5"/>')
    for k in range(4):
        ly = y + 22 + k * 10
        raw(b, f'<line x1="{x + 8}" y1="{ly}" x2="{x + w - 10}" y2="{ly}" stroke="{c["stroke"]}" '
               f'stroke-opacity="0.45" stroke-width="2"/>')
    if label:
        b.text(x + w / 2 + (n - 1) * 3.5, y + h + 18, label, 11, color, "700")
    return Box(x, y - (n - 1) * 7, w + (n - 1) * 7, h + (n - 1) * 7)


def brain(b: Board, cx, cy, color: str = "pink", r: float = 28, label: str = "LLM"):
    """A simple 'brain' blob: two lobes and a few folds."""
    c = PALETTE[color]
    col, fill = c["stroke"], c["fill"]
    raw(b, f'<path d="M{cx},{cy - r} C{cx - r * 1.35},{cy - r * 1.2} {cx - r * 1.5},{cy + r * 0.9} {cx},{cy + r * 0.85} '
           f'C{cx + r * 1.5},{cy + r * 0.9} {cx + r * 1.35},{cy - r * 1.2} {cx},{cy - r} z" fill="{fill}" '
           f'stroke="{col}" stroke-width="2"/>')
    raw(b, f'<path d="M{cx},{cy - r} v{r * 1.85} M{cx - r * 0.75},{cy - r * 0.35} q{r * 0.35},{r * 0.2} {r * 0.2},{r * 0.55} '
           f'M{cx + r * 0.75},{cy - r * 0.35} q{-r * 0.35},{r * 0.2} {-r * 0.2},{r * 0.55}" fill="none" '
           f'stroke="{col}" stroke-width="1.6" stroke-opacity="0.7"/>')
    if label:
        b.text(cx, cy + r + 18, label, 12, color, "700")
    return Box(cx - r * 1.2, cy - r, r * 2.4, r * 1.9)


def db_icon(b: Board, x, y, w=46, h=52, color="purple"):
    c = PALETTE[color]
    ry = 7
    raw(b, f'<path d="M{x},{y + ry} a{w / 2},{ry} 0 0 0 {w},0 v{h - 2 * ry} a{w / 2},{ry} 0 0 1 {-w},0 z" '
           f'fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="1.6"/>'
           f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="{c["fill"]}" '
           f'stroke="{c["stroke"]}" stroke-width="1.6"/>')
    return Box(x, y, w, h)


def panel(b: Board, x, y, w, h, title: str, color: str = "blue", size: int = 14, lines=None,
          body_size: int = 11, dashed: bool = False, align: str = "center"):
    """A card whose title sits at the top, leaving room below for drawings."""
    box = b.card(x, y, w, h, "", [], color, dashed=dashed)
    tx = x + w / 2 if align == "center" else x + 12
    anchor = "middle" if align == "center" else "start"
    b.text(tx, y + 12 + size, title, size, color if color != "dark" else "#ffffff", "700", anchor=anchor)
    yy = y + 12 + size + 8
    for ln in lines or []:
        yy += body_size * 1.45
        b.text(tx, yy, ln, body_size, INK if color != "dark" else "#e9ecef", anchor=anchor)
    return box


def verdict(b: Board, x, y, text: str, ok: bool | None, anchor: str = "start", size: int = 11):
    color = "grey" if ok is None else ("green" if ok else "red")
    return b.pill(x, y, text, color, size=size, solid=ok is not None, anchor=anchor)


def section_label(b: Board, x, y, text: str, color: str = "grey"):
    b.text(x, y, text.upper(), 11, FAINT if color == "grey" else color, "700", anchor="start", family=SANS)


def bracket(b: Board, x, y, h, color="red", side="left"):
    """A tall curly bracket that fans a heading into several items."""
    col = stroke(color)
    k = 10 if side == "left" else -10
    raw(b, f'<path d="M{x + k},{y} q{-k},0 {-k},{12} v{h / 2 - 22} q0,10 {-k * 0.8},10 q{k * 0.8},0 {k * 0.8},10 '
           f'v{h / 2 - 22} q0,12 {k},12" fill="none" stroke="{col}" stroke-width="2"/>')


# ------------------------------------------------------------ 1 · Recap boards


@board
def s2_recap_advanced_rag():
    b = Board(1240, 700, "Advanced RAG · what we already built",
              "Recap whiteboard, page 1 (0:17 to 0:20) and the tech-stack page 3 (0:23 to 0:24)",
              title_color="yellow")

    # Page 1 -----------------------------------------------------------------
    b.group(20, 92, 1200, 400, "page 1 · Advance Rag", "yellow", label_pos="bottom")
    b.text(620, 150, "Advanced RAG", 30, "yellow", "700")
    underline(b, 505, 735, 162, "yellow")

    # centre: L-shaped arrow from the title into the agent graph
    b.arrow((560, 172), (650, 300), via=[(560, 300)], color="yellow", width=2.2)
    graph_icon(b, 662, 236, "yellow", 1.05)
    b.text(720, 384, "LangGraph agent graph", 13, "yellow", "700")
    b.text(720, 402, "nodes · branches · loops", 11, FAINT)

    # left: production system fanning into tracing and Logfire
    ps = b.card(60, 196, 250, 60, "Prod system", ["it runs as a production system"], "red", size=11)
    tr = b.card(110, 300, 250, 58, "Trace the app", ["every request is traced"], "red", size=11)
    lf = b.card(110, 388, 250, 64, "Logfire !!", ["Pydantic Logfire observes", "the execution"], "pink", size=11)
    b.arrow(ps.left(0.7), tr.left(), via=[(44, ps.y + 42), (44, tr.cy)], color="red")
    b.arrow(ps.left(0.7), lf.left(), via=[(44, ps.y + 42), (44, lf.cy)], color="red")
    b.arrow((505, 150), ps.top(0.6), via=[(185, 150)], color="red", dashed=True, width=1.4)

    # right: retrieval, then reranking with FlashRank
    rt = b.card(930, 196, 250, 58, "Retrieval", ["Qdrant vector search"], "red", size=11)
    rr = b.card(930, 290, 250, 58, "Reranking", ["re-score the candidates"], "red", size=11)
    fr = b.card(930, 384, 250, 64, "FlashRank", ["the reranking library", "used in the project"], "red", size=11)
    b.arrow(rt.right(0.5), rr.right(0.5), via=[(1200, rt.cy), (1200, rr.cy)], color="red")
    b.arrow(rr.bottom(), fr.top(), color="red")
    underline(b, 985, 1125, 456, "red", double=True)
    b.arrow((810, 290), rt.left(0.25), color="yellow", dashed=True, width=1.4, label="graph nodes")

    # Page 3 -----------------------------------------------------------------
    b.group(20, 512, 1200, 168, "page 3 · Tech stack :-", "red")
    circled(b, 110, 600, "1", "red")
    ts1 = b.card(140, 566, 300, 68, "LangGraph", ["the agent graph framework"], "yellow", size=11)
    graph_icon(b, 470, 546, "red", 0.9)
    circled(b, 690, 600, "2", "red")
    vdb = b.card(720, 566, 200, 68, "Vector DB", ["stores the chunks"], "purple", size=11)
    qd = b.cylinder(990, 552, 190, 96, "Qdrant", ["qdrant db"], "red", size=11)
    b.arrow(vdb.right(), qd.left(0.55), color="red")
    b.arrow(ts1.right(), (470, 600), color="red", width=1.4)
    return b


@board
def s2_recap_scalability():
    b = Board(1200, 640, "Scalability, K8s data and the goal",
              "Recap whiteboard, page 2 (0:20 to 0:23)", title_color="yellow")

    # Scalability fork
    b.group(20, 92, 400, 400, "Scalability ✓", "yellow")
    top = (220, 150)
    b.text(220, 146, "two axes", 12, FAINT)
    us = b.card(40, 200, 170, 150, "User", ["10", "↓", "100,000", "", "1 million users,", "same latency"],
                "yellow", size=12)
    da = b.card(230, 200, 170, 150, "Data", ["10 MB", "↓", "10 GB", "", "big data needs", "ETL / ELT"],
                "yellow", size=12)
    b.arrow(top, us.top(), color="yellow")
    b.arrow(top, da.top(), color="yellow")
    b.text(220, 388, "grow users and data", 12, "yellow", "700")
    b.text(220, 406, "without losing speed", 12, "yellow", "700")

    # K8s → data → true / noisy
    b.group(450, 92, 480, 400, "K8s ! · the Kubernetes chatbot", "pink")
    k8 = b.card(520, 136, 200, 60, "K8s !", ["corpus scraped from the web"], "pink", size=11)
    d2 = b.card(560, 270, 150, 64, "Data", ["scraped pages"], "pink", size=11)
    b.arrow(k8.left(0.7), d2.left(), via=[(480, k8.y + 42), (480, d2.cy)], color="pink", curve=False)
    tr = b.card(770, 204, 140, 64, "True !", ["clean, useful"], "green", size=11)
    no = b.card(770, 330, 140, 64, "Noisy !", ["menus, junk", "duplicates"], "red", size=11)
    b.arrow(d2.right(0.4), tr.left(), color="pink")
    b.arrow(d2.right(0.6), no.left(), color="pink")
    b.text(900, 300, "?", 22, "pink", "700")
    b.text(900, 430, "?", 22, "pink", "700")

    # ETL / ELT
    b.group(960, 92, 220, 400, "", "yellow")
    b.text(1070, 150, "ETL, ELT", 22, "yellow", "700")
    underline(b, 1000, 1140, 162, "yellow", double=True)
    b.card(980, 200, 180, 170, "when Data grows", ["Extract", "Transform", "Load", "", "(or load first,", "transform later)",
           "", "the Data axis"], "yellow", size=11)

    # Goal line
    goal = b.card(120, 530, 960, 80, "Accurate sys, high noisy env ! ✓",
                  ["the target: an accurate system in a highly noisy environment"], "yellow", size=13,
                  title_size=20)
    b.arrow(d2.bottom(), (d2.cx, goal.y), color="pink", label="both kinds reach the index")
    return b


@board
def s2_recap_implementation():
    b = Board(1100, 520, "Recap → implementation",
              "Whiteboard page 11 (0:47 to 0:48): the project in one line before the code", title_color="yellow")

    # Left: the recap, and the data split written under it
    b.text(200, 128, "Recap", 30, "yellow", "700")
    underline(b, 130, 270, 140, "yellow")
    tr = b.card(60, 190, 150, 60, "True", ["useful K8s data"], "green", size=11)
    five = b.card(250, 190, 130, 60, "5 %", ["true data"], "green", title_size=20, size=11)
    nine = b.card(250, 300, 130, 60, "95 %", ["noisy data"], "red", title_size=20, size=11)
    b.arrow(tr.right(), five.left(), color="green")
    b.arrow(tr.bottom(), nine.left(), via=[(tr.cx, nine.cy)], color="red", curve=False)
    b.card(40, 400, 360, 90, "Goal", ["pick the right information even in a", "highly noisy, inaccurate environment"],
           "yellow", size=12)

    # Right: what the implementation adds
    b.text(760, 128, "implementation", 30, "yellow", "700")
    underline(b, 630, 890, 140, "yellow")
    b.arrow((420, 120), (610, 120), color="yellow")
    agent = b.card(560, 180, 400, 70, "Advanced RAG agent", ["a chatbot built for the Kubernetes platform"],
                   "blue", size=12)
    props = b.card(560, 280, 400, 70, "Fault tolerant · robust · secure · reliable",
                   ["what the project must also be"], "purple", size=12, title_size=13)
    b.arrow(agent.bottom(), props.top())
    for i, (t, sub, col) in enumerate([("Guardrails", "NeMo · section 2", "pink"),
                                       ("Gateways", "Portkey · section 3", "teal"),
                                       ("Vector DB", "Qdrant", "orange")]):
        c = b.card(540 + i * 150, 400, 135, 70, t, [sub], col, size=11)
        b.arrow(props.bottom(), c.top())
    return b


def mini_waterfall(b: Board, x, y, w):
    spans = [(0.0, 1.0, "request", "pink"), (0.05, 0.25, "guard", "teal"), (0.28, 0.55, "retrieve", "purple"),
             (0.55, 0.7, "rerank", "orange"), (0.7, 0.98, "respond", "blue")]
    for i, (s, e, name, col) in enumerate(spans):
        yy = y + i * 17
        raw(b, f'<rect x="{x + w * s:.1f}" y="{yy}" width="{w * (e - s):.1f}" height="12" rx="3" '
               f'fill="{PALETTE[col]["stroke"]}" fill-opacity="0.75"/>')
        b.text(x - 6, yy + 10, name, 10, FAINT, anchor="end")


@board
def s2_recap_rerank_observability():
    b = Board(1300, 900, "Retrieval is math, reranking is meaning",
              "Recap whiteboard, pages 4 and 5 (0:28 to 0:31)", title_color="red")

    # Page 4: vector DB retrieval
    b.group(20, 92, 610, 470, "page 4 · RAG → Technical query", "red")
    q1 = b.card(50, 140, 190, 50, "[ query ]", [], "grey", size=12)
    vdb = b.cylinder(290, 128, 130, 78, "Vdb", ["Qdrant"], "purple", size=11)
    b.arrow(q1.right(), vdb.left(0.55), color="red")
    raw(b, '<ellipse cx="530" cy="168" rx="82" ry="44" fill="#fff5f5" stroke="#e03131" stroke-width="2" '
           'stroke-dasharray="6 4"/>')
    b.text(530, 164, "15 chunks", 16, "red", "700")
    b.text(530, 184, "retrieved", 11, FAINT)
    b.arrow(vdb.right(0.55), (448, 168), color="red")

    b.card(50, 236, 250, 92, "Cosine Similarity", ["\"math\": how close two", "vectors are", "→ Relevant?"],
           "red", size=11)
    qx = b.card(60, 380, 150, 46, "[ query ]", [], "grey", size=12)
    chunks = [chunk_box(b, 400, 350 + i * 50, 170, 36, "red") for i in range(3)]
    for i, ch in enumerate(chunks):
        b.arrow(qx.right(), ch.left(), color="red", width=1.5)
    b.text(300, 360, "cos θ", 13, "red", "700")
    b.text(325, 520, "one vector each · compared by angle", 12, FAINT)
    b.text(325, 540, "close vectors ≠ an answer to the question", 12, "red", "700")

    # Page 5: FlashRank cross-encoder
    b.group(670, 92, 610, 470, "page 5 · flash Rank", "yellow")
    fr = b.card(700, 136, 250, 64, "FlashRank", ["cross encoder", "attention"], "yellow", size=12)
    circled(b, 1030, 168, "15", "red", r=22)
    b.text(1062, 173, "retrieved", 12, "red", "700", anchor="start")
    b.arrow((600, 140), (700, 150), color="red", label="15 in", width=2)
    q2 = b.card(700, 330, 150, 46, "[ query ]", [], "grey", size=12)
    pairs = [chunk_box(b, 950, 290 + i * 50, 150, 36, "yellow") for i in range(3)]
    for ch in pairs:
        b.arrow(q2.right(), ch.left(), color="yellow", width=1.5)
    b.text(900, 280, "read together", 11, "yellow", "700")
    b.text(1180, 300, "Semantics", 16, "yellow", "700")
    underline(b, 1125, 1240, 310, "yellow", double=True)
    circled(b, 1180, 380, "5", "yellow", r=26, size=22)
    b.text(1180, 430, "top 5 kept !!", 12, "yellow", "700")
    b.text(975, 470, "query + chunk pass through attention together,", 12, FAINT)
    b.text(975, 490, "so each chunk is scored on meaning", 12, FAINT)
    b.arrow(fr.bottom(0.3), (q2.cx, q2.y), color="yellow", width=1.4)

    # Math vs meaning strip
    b.card(160, 590, 400, 70, "math", ["cosine similarity · fast · rough"], "red", size=12, title_size=20)
    b.card(740, 590, 400, 70, "meaning !!", ["cross-encoder attention · slower · precise"], "yellow", size=12,
           title_size=20)
    b.text(650, 632, "vs", 20, INK, "700")

    # Observability
    b.group(20, 690, 1260, 190, "", "pink")
    b.text(170, 790, "Observability", 24, "pink", "700")
    b.arrow((420, 785), (330, 785), color="pink", width=3)
    panel(b, 440, 716, 230, 144, "Span", "pink", lines=["one timed operation,", "e.g. one Qdrant search"])
    raw(b, f'<rect x="475" y="806" width="160" height="14" rx="4" fill="{PALETTE["purple"]["stroke"]}" fill-opacity="0.75"/>')
    b.text(555, 842, "start → end, with attributes", 10, FAINT)
    panel(b, 690, 716, 230, 144, "Trace", "pink", lines=["every span of one", "request, grouped"])
    for i, (s0, e0) in enumerate([(0, 1), (0.1, 0.4), (0.45, 0.8), (0.8, 0.95)]):
        raw(b, f'<rect x="{715 + 180 * s0:.0f}" y="{800 + i * 13}" width="{180 * (e0 - s0):.0f}" height="9" rx="3" '
               f'fill="{PALETTE["pink"]["stroke"]}" fill-opacity="{0.9 - i * 0.15:.2f}"/>')
    panel(b, 940, 716, 320, 144, "Waterfall !!", "pink")
    mini_waterfall(b, 1010, 752, 230)
    return b


@board
def s2_langgraph_graph():
    b = Board(1100, 700, "The LangGraph the app renders",
              "GET /graph at 0:50 (again at 2:01), with the recap's numbers beside each node", title_color="purple")
    b.group(300, 92, 500, 588, "compiled graph", "purple")
    st = b.pill(550, 130, "__start__", "grey", size=13, anchor="middle")
    pl = b.card(450, 214, 200, 56, "planner", [], "purple", size=14, title_size=16)
    rt = b.card(340, 360, 200, 56, "retriever", [], "purple", size=14, title_size=16)
    rp = b.card(470, 500, 200, 56, "responder", [], "purple", size=14, title_size=16)
    en = b.pill(570, 628, "__end__", "grey", size=13, anchor="middle")
    b.arrow(st.bottom(), pl.top(), color="purple")
    b.arrow(pl.bottom(0.3), rt.top(0.6), dashed=True, color="purple", label="needs evidence")
    b.arrow(pl.bottom(0.75), rp.top(0.7), dashed=True, color="purple", label="answerable\nfrom history")
    b.arrow(rt.bottom(0.6), rp.top(0.2), color="purple")
    b.arrow(rp.bottom(), en.top(), color="purple")

    # annotations
    q = b.card(30, 180, 230, 90, "Technical question", ["or a follow-up", "+ conversation history"], "blue",
               size=11)
    b.arrow(q.right(), pl.left(), color="blue", dashed=True, width=1.4)
    ra = b.card(30, 330, 250, 116, "retriever does", ["Qdrant search:", "15 candidates", "→ rerank: top 5"],
           "orange", size=12, dashed=True)
    pa = b.card(830, 180, 250, 100, "planner decides", ["retrieve, or answer", "from the stored", "conversation"],
           "purple", size=12, dashed=True)
    rpa = b.card(830, 470, 250, 116, "responder returns", ["the answer + a trace", "+ the stored", "conversation"],
           "green", size=12, dashed=True)
    for a, n in ((ra.right(), rt.left()), (pa.left(), pl.right()), (rpa.left(), rp.right())):
        raw(b, f'<line x1="{a[0]}" y1="{a[1]}" x2="{n[0]}" y2="{n[1]}" stroke="{FAINT}" stroke-width="1.3" '
               f'stroke-dasharray="2 4"/>')
    b.text(550, 668, "dotted edges are conditional, as in the rendered graph", 11, FAINT)
    return b

# ------------------------------------------------------- Aside · model strategy


def org_data_oval(b: Board, cx, cy, rx, ry):
    c = PALETTE["pink"]
    raw(b, f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{c["fill"]}" stroke="{c["stroke"]}" '
           f'stroke-width="2.4"/>')
    dots = [(-0.55, -0.25), (-0.2, -0.45), (0.2, -0.35), (0.55, -0.15), (-0.45, 0.25), (-0.05, 0.1),
            (0.35, 0.3), (0.1, 0.55), (-0.3, 0.6)]
    for i, (dx, dy) in enumerate(dots):
        x, y = cx + dx * rx, cy + dy * ry
        raw(b, f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{9 if i % 2 else 12}" fill="#ffffff" stroke="{c["stroke"]}" '
               f'stroke-width="1.6"/>')
        if i % 3 == 0:
            mark(b, x + 20, y + 4, True, 14)
    mark(b, cx - 18, cy - ry + 24, False, 16)
    mark(b, cx + 4, cy - ry + 24, False, 16)


def server_icon(b: Board, x, y, w=90, h=110, color="yellow"):
    c = PALETTE[color]
    raw(b, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{c["fill"]}" stroke="{c["stroke"]}" '
           f'stroke-width="2"/><line x1="{x}" y1="{y + h * 0.72}" x2="{x + w}" y2="{y + h * 0.72}" '
           f'stroke="{c["stroke"]}" stroke-width="2"/>')
    for i in range(2):
        for j in range(2):
            raw(b, f'<rect x="{x + 16 + j * 34}" y="{y + 14 + i * 30}" width="22" height="18" rx="3" fill="#ffffff" '
                   f'stroke="{c["stroke"]}" stroke-width="1.5"/>')
    return Box(x, y, w, h)


@board
def s2_api_vs_llm_engineering():
    b = Board(1300, 720, "API models vs LLM engineering",
              "Aside whiteboard, pages 12 and 13 (1:04 to 1:08)", title_color="pink")

    b.group(20, 92, 620, 608, "use an LLM through an API", "red")
    top = b.card(50, 138, 260, 64, "LLM ✓", ["for AI agentic engineering"], "yellow", size=11, title_size=18)
    uc = b.card(50, 236, 260, 64, "use case", ["built for generic use cases"], "red", size=11)
    se = b.card(50, 330, 260, 64, "secure", ["your data goes to them"], "red", size=11)
    for box in (uc, se):
        mark(b, box.x + box.w - 44, box.y + 22, False, 16)
        mark(b, box.x + box.w - 22, box.y + 22, False, 16)
    b.arrow(top.left(0.8), uc.left(), via=[(36, top.y + 51), (36, uc.cy)], color="red")
    b.arrow(top.left(0.8), se.left(), via=[(36, top.y + 51), (36, se.cy)], color="red")

    org_data_oval(b, 470, 480, 150, 110)
    b.text(470, 616, "your organisation's data", 13, "pink", "700")
    api = b.card(40, 450, 170, 64, "api", ["third-party provider"], "red", size=11, title_size=18)
    b.arrow((320, 480), api.right(), color="red", label="data leaves", width=2)
    b.text(330, 660, "fast to start · but the provider decides and your data leaves", 12, "red", "700")

    b.group(660, 92, 620, 608, "LLM engineering", "yellow")
    os_ = b.card(690, 138, 300, 70, "Open source model", ["trained on general Data"], "blue", size=11)
    sv = server_icon(b, 1080, 132, 110, 130, "yellow")
    b.text(1200, 200, "your", 12, "yellow", "700", anchor="start")
    b.text(1200, 216, "own", 12, "yellow", "700", anchor="start")
    b.text(1200, 232, "model", 12, "yellow", "700", anchor="start")
    b.arrow(os_.right(), (sv.x, sv.y + 40), color="yellow")
    od = b.cylinder(1060, 340, 150, 80, "own Data", ["goes into it"], "pink", size=11)
    b.arrow(od.top(), sv.bottom(0.5), color="pink")
    ft = b.card(690, 470, 260, 64, "finetuning", ["train it on your data"], "pink", size=12, title_size=16)
    pa = b.card(690, 590, 260, 70, "Private assistants", ["stays inside the organisation"], "pink", size=11,
                title_size=16)
    b.arrow((sv.x, sv.y + 100), ft.top(0.7), via=[(1010, sv.y + 100), (1010, 440), (ft.x + 182, 440)],
            color="yellow")
    b.arrow(ft.bottom(), pa.top(), color="pink")
    co = b.card(1030, 500, 220, 80, "costly", ["GPUs, data work, people"], "red", size=11, title_size=20)
    underline(b, 1060, 1220, 590, "yellow", double=True)
    b.arrow(sv.right(0.95), (1240, co.y), via=[(1240, sv.y + 124)], color="yellow", dashed=True)
    return b


@board
def s2_knowledge_distillation():
    b = Board(1200, 600, "Knowledge Distillation", "Aside whiteboard, page 14 (1:08 to 1:11)", title_color="pink")

    b.person(110, 130, "yellow", 1.5, "teacher")
    t = b.card(200, 120, 330, 110, "Teacher model", ["trained on a lot of data"], "yellow", size=12,
               title_size=18)
    b.pill(220, 132, "huge", "yellow", solid=True)
    pr = b.card(250, 300, 230, 70, "Prediction", ["the teacher's answers"], "yellow", size=11, title_size=16)
    b.arrow(t.bottom(0.4), pr.top(0.4), color="yellow")

    s = b.card(720, 120, 300, 110, "Student model", ["learns from the predictions"], "green", size=12,
               title_size=18)
    raw(b, f'<rect x="1040" y="140" width="46" height="40" rx="4" fill="#ebfbee" stroke="{PALETTE["green"]["stroke"]}" '
           f'stroke-width="2"/>')
    b.text(1063, 200, "small", 12, "green", "700")
    b.person(1120, 150, "green", 0.9, "student")
    b.arrow(pr.right(), s.bottom(0.3), via=[(620, pr.cy), (620, 270), (s.x + 90, 270)], color="yellow", width=2.2,
            label="train to match")

    b.group(20, 410, 1160, 170, "the host's analogy", "grey")
    b.card(60, 456, 480, 100, "teacher took", ["15 days  (or 1 month)", "to learn it from scratch"], "yellow", size=13,
           title_size=16)
    b.card(660, 456, 480, 100, "student picks it up in", ["1 day  (or 16 hrs)", "by learning from the teacher"], "green",
           size=13, title_size=16)
    b.arrow((545, 506), (655, 506), color="grey", width=2.4)
    return b


# --------------------------------------------------------- 2 · Guardrails (NeMo)


@board
def s2_llm_security():
    b = Board(1100, 470, "LLM security", "Whiteboard page 5, repeated at the top of page 6 (0:35 to 0:36)",
              title_color="red")
    root = b.card(400, 100, 300, 64, "LLM security", ["two controls around the model"], "red", size=12,
                  title_size=18)
    g1 = b.card(80, 240, 420, 180, "guardrails", ["what the assistant will talk about", "", "NeMo Guardrails",
                                                   "rules on the conversation", "→ section 2"], "green", size=12,
                title_size=18)
    g2 = b.card(600, 240, 420, 180, "gateways", ["how each model call is made", "", "Portkey",
                                                 "routing, fallback, keys, logs", "→ section 3"], "yellow", size=12,
                title_size=18)
    b.arrow(root.bottom(0.4), g1.top(), color="red", width=2.2)
    b.arrow(root.bottom(0.6), g2.top(), color="red", width=2.2)
    return b


@board
def s2_why_evaluate():
    b = Board(1300, 700, "Why a RAG system needs evals",
              "Whiteboard pages 6 and 7 (0:36 to 0:38)", title_color="pink")
    b.pill(420, 84, "guardrails", "green", size=12)
    b.pill(560, 84, "gateways", "yellow", size=12)
    b.pill(690, 84, "→ and now: evaluation", "pink", size=12)

    b.group(20, 120, 600, 560, "ML, DL", "blue")
    b.text(320, 176, "Train, Test, Val", 18, "blue", "700")
    x0, y0, w = 60, 196, 520
    for (s0, e0, name, col) in ((0, 0.7, "Train", "blue"), (0.7, 0.85, "Test", "red"), (0.85, 1, "Val", "teal")):
        raw(b, f'<rect x="{x0 + w * s0:.0f}" y="{y0}" width="{w * (e0 - s0):.0f}" height="40" '
               f'fill="{PALETTE[col]["fill"]}" stroke="{PALETTE[col]["stroke"]}" stroke-width="1.6"/>')
        b.text(x0 + w * (s0 + e0) / 2, y0 + 25, name, 13, col, "700")
    raw(b, f'<ellipse cx="{x0 + w * 0.775:.0f}" cy="{y0 + 20}" rx="52" ry="30" fill="none" '
           f'stroke="{PALETTE["red"]["stroke"]}" stroke-width="2.4"/>')
    b.text(320, 270, "a held-out Test split, kept aside", 12, FAINT)
    acc = b.card(60, 300, 520, 90, "Acc, Pre, Recall !!", ["deterministic maths over known labels"], "blue",
                 size=12, title_size=20)
    b.arrow((x0 + w * 0.775, y0 + 50), (x0 + w * 0.775, acc.y), color="red")
    b.card(60, 420, 250, 90, "Math →", ["same input,", "same label, every time"], "blue", size=11,
           title_size=16)
    b.card(330, 420, 250, 90, "DL", ["deep learning uses", "the same recipe"], "blue", size=11, title_size=16)
    b.text(320, 560, "you can score a classifier because", 13, "blue", "700")
    b.text(320, 580, "its output is a fixed label", 13, "blue", "700")

    b.group(660, 120, 620, 560, "Rag →", "pink")
    s1 = b.card(700, 170, 260, 54, "Data ingestion", [], "pink", size=12)
    s2 = b.card(700, 248, 260, 54, "Data Retrieval", [], "pink", size=12)
    s3 = b.card(700, 326, 260, 60, "LLM evaluation", ["the missing stage"], "red", size=11)
    underline(b, 720, 940, 394, "pink", double=True)
    b.arrow(s1.bottom(), s2.top(), color="pink")
    b.arrow(s2.bottom(), s3.top(), color="pink")
    cb = b.card(1000, 170, 250, 70, "Chatbot", ["the most random thing"], "yellow", size=11)
    rd = b.card(1000, 270, 250, 70, "Random", ["new wording every run"], "yellow", size=11)
    mt = b.card(1000, 370, 250, 70, "Metric", ["needs its own"], "yellow", size=11)
    b.arrow(cb.bottom(), rd.top(), color="yellow")
    b.arrow(rd.bottom(), mt.top(), color="yellow")
    ev = b.card(780, 480, 400, 150, "Evals", ["metrics built for generated text:", "", "", ""], "pink", size=12,
                title_size=22)
    b.pill(800, 568, "hallucination", "red", size=11)
    b.pill(935, 568, "answer relevancy", "pink", size=11)
    b.pill(1000, 598, "context recall", "purple", size=11, anchor="middle")
    b.arrow(s3.bottom(0.5), (s3.cx, ev.y), color="pink")
    b.arrow(mt.bottom(), (mt.cx, ev.y), color="yellow")
    return b


def code_card(b: Board, x, y, w, lines, title="", color="dark", size=11):
    h = 24 + len(lines) * size * 1.5 + (22 if title else 0)
    raw(b, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="#1f2937" stroke="#1f2937"/>')
    yy = y + 20
    if title:
        b.text(x + 14, yy, title, 11, "#9ca3af", "700", anchor="start")
        yy += 22
    palette = {"define": "#f783ac", "user": "#74c0fc", "bot": "#8ce99a", "flow": "#ffd43b"}
    for ln in lines:
        indent = len(ln) - len(ln.lstrip(" "))
        ln = "\u00a0" * indent + ln.lstrip(" ")
        col = "#e9ecef"
        bare = ln.lstrip("\u00a0")
        if bare.startswith('"'):
            col = "#ffe8cc"
        for k, v in palette.items():
            if bare.startswith(k):
                col = v
        b.text(x + 14, yy, ln, size, col, anchor="start")
        yy += size * 1.5
    return Box(x, y, w, h)


@board
def s2_colang_define():
    b = Board(1300, 720, "Guardrails → NeMo → Colang",
              "Whiteboard pages 8 and 9 (0:38 to 0:41)", title_color="green")
    gr = b.card(40, 100, 220, 60, "guardrails", [], "green", size=12, title_size=18)
    ng = b.card(340, 100, 260, 60, "nemo guard", ["NeMo Guardrails"], "green", size=11)
    b.pill(620, 116, "by NVIDIA", "green", size=12, solid=True)
    b.arrow(gr.right(), ng.left(), color="green")

    rr = b.card(40, 210, 380, 66, "Rules & Regulation", ["what may be said, and how to answer"], "green",
                size=11, title_size=18)
    b.arrow(gr.bottom(0.3), rr.top(0.18), color="green")
    co = b.card(520, 210, 200, 66, "Colang", ["the rule language"], "green", size=11, title_size=18)
    underline(b, 540, 700, 284, "green", double=True)
    fdoc = doc_stack(b, 760, 214, "pink", 1, 50, 60)
    b.text(785, 250, ".co", 14, "pink", "700")
    b.arrow(rr.right(), co.left(), color="green")
    b.text(850, 250, "rules live in .co files", 12, "pink", "700", anchor="start")

    b.group(20, 320, 1260, 380, "Define →", "pink")
    du = b.card(50, 370, 280, 120, "define user", ["examples of an intent", "e.g. user → off topic",
                                                    "\"tell me a joke\""], "blue", size=11, title_size=16)
    db = b.card(50, 520, 280, 120, "define bot", ["the canned reply", "Bot ↳ \"I'm an IT", "assistant ...\""], "green",
                size=11, title_size=16)
    df = b.card(430, 440, 260, 130, "define flow", ["user intent", "→ bot response", "", "flow → user / Bot"],
                "yellow", size=11, title_size=16)
    b.arrow(du.right(), df.left(0.3), color="pink")
    b.arrow(db.right(), df.left(0.7), color="pink")
    code_card(b, 740, 360, 510, [
        "define user ask off topic",
        '  "tell me a joke"',
        '  "what is the capital of france"',
        "",
        "define bot refuse off topic",
        "  \"I'm an Enterprise IT Assistant ...\"",
        "",
        "define flow handle off topic",
        "  user ask off topic",
        "  bot refuse off topic",
    ], "the same pattern in app/guardrails/colang_rules.py")
    b.arrow(df.right(), (740, 505), color="grey", dashed=True, width=1.4)
    b.text(995, 672, "off-topic, jailbreak, greeting, capabilities and farewell rules all follow it", 11, FAINT)
    return b


@board
def s2_guard_approaches():
    b = Board(1300, 760, "Two ways to build a guard",
              "Whiteboard page 15 (1:22 to 1:24, added to at 1:33)", title_color="yellow")

    b.group(20, 92, 560, 648, "fine-tuned guard", "yellow")
    ft = b.card(60, 140, 240, 64, "Fine tuning ✓", ["train a model on the rules"], "yellow", size=11,
                title_size=16)
    raw(b, f'<ellipse cx="300" cy="330" rx="170" ry="64" fill="{PALETTE["yellow"]["fill"]}" '
           f'stroke="{PALETTE["yellow"]["stroke"]}" stroke-width="2.4"/>')
    b.text(300, 326, "Llama Guard", 22, "yellow", "700")
    b.text(300, 350, "Meta's safety classifier", 11, FAINT)
    b.arrow(ft.left(0.7), (136, 330), via=[(40, ft.y + 45), (40, 330)], color="yellow", width=2)
    fn = b.card(120, 440, 360, 64, "finetuned", ["already trained on unsafe scenarios"], "yellow", size=11)
    ot = b.card(120, 540, 360, 64, "off topic", ["detected from its training"], "yellow", size=11)
    b.arrow((300, 394), fn.top(), color="yellow")
    b.arrow(fn.bottom(), ot.top(), color="yellow")
    b.text(300, 660, "the knowledge is inside the weights", 12, "yellow", "700")

    b.group(620, 92, 660, 648, "nemo guardrails · open source", "pink")
    llm = b.card(840, 140, 220, 70, "LLM ✓", ["decides the intent"], "yellow", size=11, title_size=20)
    underline(b, 880, 1020, 218, "yellow", double=True)
    fe = b.card(840, 260, 220, 66, "fast embed", ["FastEmbed similarity"], "yellow", size=11, title_size=16)
    b.arrow(fe.top(), llm.bottom(), color="yellow")
    ng = b.card(760, 370, 380, 60, "nemo guardrails", ["compares the message to your examples"], "pink",
                size=11)
    b.arrow(ng.top(), fe.bottom(), color="pink")
    sc = b.card(660, 480, 280, 74, "scenarios ✓", ["the examples you defined", "in Colang"], "pink", size=11)
    un = b.card(980, 480, 270, 74, "unseen scenarios", ["phrasings not in the", "examples"], "red", size=11)
    mark(b, 1210, 500, False, 16)
    mark(b, 1232, 500, False, 16)
    b.arrow(sc.top(0.7), ng.bottom(0.3), color="pink")
    b.arrow(un.top(0.3), ng.bottom(0.7), color="red", dashed=True)
    verdict(b, 700, 604, "LLM: not safe → blocked", False, size=12)
    verdict(b, 980, 604, "LLM: safe → bypassed to RAG", True, size=12)
    b.text(950, 680, "similar examples narrow it down; the LLM makes the call", 12, "pink", "700")
    return b


# ---------------------------------------------------------- 3 · Gateways (Portkey)


@board
def s2_gateway():
    b = Board(1300, 740, "Gateways → Portkey", "Whiteboard page 9 (0:41 to 0:44)", title_color="pink")

    b.group(20, 92, 1260, 190, "the problem", "red")
    us = b.card(60, 150, 220, 90, "100000", ["users at once"], "red", size=12, title_size=24)
    ap = b.card(460, 136, 220, 60, "Api", ["one provider's API"], "red", size=11, title_size=18)
    lr = b.card(460, 208, 220, 60, "Late response ✓", ["rate limits, slow replies"], "red", size=11)
    b.arrow(us.right(0.4), ap.left(), color="red")
    b.arrow(us.right(0.6), lr.left(), color="red")
    b.card(760, 150, 480, 90, "why a gateway", ["the provider's limits and outages", "become your users' problem"], "grey",
           size=12, dashed=True)

    b.group(20, 302, 1260, 418, "the gateway", "yellow")
    u = b.person(80, 360, "blue", 1.1, "user")
    gw = b.card(200, 370, 250, 80, "gateway", ["Portkey"], "yellow", size=13, title_size=20)
    b.arrow((110, 400), gw.left(), color="blue", width=2.2)
    provs = [("Op", "OpenAI"), ("an", "Anthropic"), ("ge", "Gemini")]
    boxes = []
    for i, (short, name) in enumerate(provs):
        bx = b.card(560 + i * 190, 360, 160, 90, short, [name], "blue", size=11, title_size=20)
        boxes.append(bx)
    extra = b.card(1130, 360, 120, 90, "…", ["another", "provider"], "grey", size=11, dashed=True)
    for bx in boxes + [extra]:
        b.arrow(gw.right(0.5), bx.left(0.5) if bx is boxes[0] else bx.top(), color="yellow", width=1.4,
                via=None if bx is boxes[0] else [(gw.x + gw.w + 20, gw.cy), (gw.x + gw.w + 20, 340), (bx.cx, 340)])
    b.arrow(boxes[0].bottom(0.8), boxes[1].bottom(0.2), via=[(boxes[0].x + 128, 480), (boxes[1].x + 32, 480)],
            color="red", width=2, label="unavailable")
    b.arrow(boxes[1].bottom(0.8), boxes[2].bottom(0.2), via=[(boxes[1].x + 128, 480), (boxes[2].x + 32, 480)],
            color="red", width=2, label="unavailable")
    b.text(830, 516, "fallback: if OpenAI is down, route to Anthropic, then Gemini", 12, "red", "700")

    mcp = b.card(200, 560, 250, 64, "MCP's", ["tool servers"], "yellow", size=11, title_size=18)
    tool = b.card(540, 560, 200, 64, "Tool", ["called through it"], "yellow", size=11, title_size=18)
    b.arrow(gw.bottom(0.5), mcp.top(0.5), color="yellow")
    b.arrow(mcp.right(), tool.left(), color="yellow")
    vk = b.card(820, 560, 420, 110, "Virtual Keys", ["the app holds a gateway key;", "real provider keys stay",
                                                     "inside the gateway"], "red", size=11, title_size=18)
    b.arrow(gw.right(0.8), (vk.x, 650), via=[(500, gw.y + 64), (500, 650)], color="red", dashed=True, width=1.4)
    return b


@board
def s2_gateway_keys():
    b = Board(1300, 640, "10 LLMs, one API", "Whiteboard page 10 (0:44 to 0:45)", title_color="yellow")
    red, yel = PALETTE["red"]["stroke"], PALETTE["yellow"]["stroke"]

    b.group(20, 92, 610, 528, "without a gateway", "red")
    b.text(325, 150, "LLM → Api ✓", 20, "red", "700")
    app1 = b.card(60, 180, 140, 70, "App", ["your code"], "blue", size=11, title_size=16)
    b.text(450, 190, "↳ 10 LLM's", 16, "red", "700")
    underline(b, 380, 520, 200, "red", double=True)
    raw(b, f'<path d="M130,250 V275 H559 M235,275 V425 H559" fill="none" stroke="{red}" stroke-width="1.4"/>')
    for i in range(10):
        row, col = divmod(i, 5)
        kx, ky = 250 + col * 70, 290 + row * 150
        bus = 275 if row == 0 else 425
        b.arrow((kx + 29, bus), (kx + 29, ky), color="red", width=1.2)
        b.card(kx, ky, 58, 40, "key", [], "red", size=10, title_size=11)
        b.card(kx, ky + 60, 58, 40, f"LLM{i + 1}", [], "blue", size=10, title_size=10)
        b.arrow((kx + 29, ky + 40), (kx + 29, ky + 60), color="red", width=1.2)
    b.text(325, 580, "10 api keys to store, rotate and secure", 14, "red", "700")
    underline(b, 175, 475, 592, "yellow")

    b.group(670, 92, 610, 528, "with a gateway", "yellow")
    app2 = b.card(700, 150, 140, 70, "App", ["your code"], "blue", size=11, title_size=16)
    raw(b, f'<ellipse cx="975" cy="185" rx="80" ry="42" fill="{PALETTE["yellow"]["fill"]}" '
           f'stroke="{yel}" stroke-width="2.4"/>')
    b.text(975, 182, "api", 20, "yellow", "700")
    b.text(975, 202, "one key", 11, FAINT)
    b.arrow(app2.right(), (895, 185), color="yellow", width=2.2)
    gw = b.card(890, 262, 170, 64, "gateway", ["one endpoint"], "yellow", size=11, title_size=18)
    b.arrow((975, 227), gw.top(), color="yellow", width=2.2)
    raw(b, f'<path d="M975,326 V372 M705,372 H1205 M705,372 V457 H1205" fill="none" stroke="{yel}" stroke-width="1.6"/>')
    for i in range(10):
        row, col = divmod(i, 5)
        mx, my = 720 + col * 110, 400 + row * 70
        bus = 372 if row == 0 else 457
        b.arrow((mx + 45, bus), (mx + 45, my), color="yellow", width=1.2)
        b.card(mx, my, 90, 44, f"LLM {i + 1}", [], "blue", size=10, title_size=11)
    b.text(975, 580, "api ⟸ 10 LLM's", 16, "yellow", "700")
    underline(b, 870, 1080, 592, "yellow", double=True)
    return b


@board
def s2_gateway_explorer_streaming():
    b = Board(1300, 700, "LLM Gateway Explorer · Streaming",
              "Portkey's demo app as shown in the session (1:48)", title_color="purple")
    items = ["Home", "Baseline (No Gateway)", "Routing & Observability", "Metadata & Tracking",
             "Automatic Retries", "Request Timeouts", "Fallback Routing", "Retry + Timeout + Fallback",
             "Load Balancing", "Response Caching", "Rate Limiting", "Streaming", "Production Config"]
    panel(b, 20, 92, 360, 588, "Experiments", "grey", align="left")
    for i, it in enumerate(items):
        y = 150 + i * 40
        on = it == "Streaming"
        col = PALETTE["red"]["stroke"] if on else "#adb5bd"
        raw(b, f'<circle cx="48" cy="{y - 4}" r="8" fill="{"#fff" if not on else col}" stroke="{col}" stroke-width="2"/>')
        b.text(66, y, it, 13, "red" if on else INK, "700" if on else "400", anchor="start")

    panel(b, 410, 92, 870, 588, "Streaming", "purple", size=20, align="left",
          lines=["Instead of waiting for the full response, streaming shows tokens as they",
                 "arrive. The user sees the answer building in real time. Portkey supports",
                 "streaming: all gateway features still work, and the full request is logged."],
          body_size=13)
    b.text(430, 256, "Architecture", 14, "purple", "700", anchor="start")
    lanes = [("App", 560), ("Portkey", 845), ("LLM", 1130)]
    for name, x in lanes:
        b.card(x - 90, 280, 180, 50, name, [], "purple", size=12, title_size=15)
        raw(b, f'<line x1="{x}" y1="330" x2="{x}" y2="650" stroke="#adb5bd" stroke-width="1.6" stroke-dasharray="4 4"/>')
    b.arrow((560, 380), (845, 380), color="purple", width=2)
    b.text(700, 370, "request", 11, FAINT)
    b.arrow((845, 430), (1130, 430), color="purple", width=2)
    b.text(988, 420, "Forward streaming request", 12, "purple", "700")
    b.arrow((1130, 500), (845, 500), color="grey", dashed=True, width=1.6)
    b.text(988, 490, "token chunks", 12, FAINT)
    b.arrow((845, 560), (560, 560), color="grey", dashed=True, width=1.6)
    b.text(700, 550, "chunks as they arrive · still logged", 12, FAINT)
    b.pill(700, 600, "grey arrows: schematic continuation, NOT visible in the session", "grey", size=11,
           anchor="middle")
    return b

# ------------------------------------------------------ Evaluation posters


@board
def s2_evaluation_approaches():
    b = Board(1320, 760, 'Two approaches to AI evaluation', 'Evaluation measures performance against objectives, criteria and expected outcomes.')
    b.card(120, 98, 1080, 68, 'Measure and assess', ['Quality · effectiveness · reliability'], 'yellow', size=15)
    b.group(20, 200, 520, 530, 'LLM evals · model alone', 'purple')
    brain(b, 280, 320, 'purple', 45)
    b.card(40, 410, 230, 150, 'Response qualities', ['Correctness · hallucination', 'Reasoning · safety', 'Bias · coherence'], 'purple', size=13)
    b.card(290, 410, 230, 150, 'Response qualities', ['Relevance · faithfulness', 'Toxicity · helpfulness', 'Consistency'], 'purple', size=13)
    a = b.card(50, 615, 170, 60, 'Input / prompt', [], 'blue', size=13)
    z = b.card(340, 615, 170, 60, 'Output / response', [], 'green', size=13)
    b.arrow(a.right(), z.left(), label='LLM', color='purple')
    b.group(570, 200, 730, 530, 'AI application evals · end to end', 'teal')
    u = b.card(600, 280, 170, 80, 'User', ['Question / request'], 'blue', size=13)
    r = b.card(820, 280, 190, 80, 'Retrieval (RAG)', [], 'teal', size=13)
    l = b.card(1090, 280, 170, 80, 'LLM', [], 'purple', size=14)
    b.arrow(u.right(), r.left()); b.arrow(r.right(), l.left())
    k = b.cylinder(820, 405, 190, 90, 'Knowledge base', [], 'teal', size=12)
    p = b.card(1080, 415, 190, 70, 'Prompt / instructions', [], 'purple', size=12)
    b.arrow(k.top(), r.bottom(), color='teal'); b.arrow(p.top(), l.bottom(), color='purple')
    t = b.card(610, 570, 200, 75, 'Tools / APIs / agents', [], 'orange', size=12)
    o = b.card(850, 570, 180, 75, 'Response / answer', [], 'green', size=12)
    g = b.card(1080, 570, 190, 110, 'User goal achieved?', ['Evaluate the whole path'], 'green', size=12)
    b.arrow(l.right(), t.left(), via=[(1285,320),(1285,535),(590,535),(590,607)], color='purple')
    b.arrow(t.right(), o.left()); b.arrow(o.right(), g.left())
    return b


@board
def s2_exam_analogy():
    b = Board(1320, 620, 'Evaluation analogy · examination', 'The student produces outputs; the teacher checks them using correct answers and scoring rules.', title_color='red')
    b.group(20, 110, 385, 340, 'Student · RAG pipeline', 'blue')
    b.person(82, 190, 'blue', scale=1.2)
    b.card(150, 185, 230, 180, 'Writes the exam', ['Creates questions', 'Generates answers', 'Real outputs on real data'], 'blue', size=13, bullets=True)
    b.text(212, 410, 'Exam paper / answers = RAG outputs', 12, 'blue', '700')
    b.arrow((410,280),(460,280),color='red',width=3)
    b.group(470, 110, 400, 340, 'Teacher · evaluation pipeline', 'green')
    b.person(530, 190, 'green', scale=1.2)
    b.card(600, 170, 245, 200, 'Evaluates the exam', ['Knows correct answers', '(ground truths)', 'Knows how to give marks', '(metrics)', 'Awards scores'], 'green', size=12)
    b.text(670, 403, 'Score per answer → total score', 12, 'green', '700')
    b.text(670, 425, '= RAG evaluation results', 12, 'green', '700')
    b.table(900, 135, [170,205], [['Exam','RAG evaluation'],['Student','RAG pipeline'],['Answers','Real outputs'],['Correct answers','Ground truths'],['Teacher','Evaluation pipeline'],['Marks / scoring','Metrics'],['Total score','Overall evaluation']], header_color='red', size=11, row_h=40)
    b.card(150, 500, 1020, 70, 'Generate → evaluate → aggregate', ['Keep application output separate from the expected answer and the marking rules.'], 'red', size=13)
    return b


@board
def s2_exam_roles():
    b = Board(1240, 830, 'Set up the exam, then write it', 'Side panels from the evaluation poster: teacher preparation and student execution.', title_color='orange')
    b.group(20, 100, 570, 700, 'A · Teacher sets up the exam', 'orange')
    setup=['Define syllabus / topics','Create questions','Prepare correct answers (ground truths)','Decide marks per question','Define rules / rubrics / metrics','Set total marks and grading criteria']
    for i,t in enumerate(setup):
        circled(b,65,195+i*89,str(i+1),'orange')
        b.card(105,163+i*89,450,64,t,[],'orange',size=13)
    b.group(620, 100, 600, 700, '2 · Student writes the exam', 'blue')
    prev=None
    for i,(t,s) in enumerate([('Read the question','Receive the user query'),('Think / retrieve information','Retrieve relevant context'),('Write the answer','Generate a response'),('Submit the paper','Real outputs to be evaluated')]):
        c=b.card(670,160+i*125,500,85,t,[s],'blue',size=14)
        if prev: b.arrow(prev.bottom(),c.top(),color='blue')
        prev=c
    b.card(670,695,500,70,'Real exam paper = RAG output',['These are the outputs evaluation will score.'],'green',size=13)
    return b


@board
def s2_exam_marking():
    b = Board(1100, 650, 'B · Evaluating the exam', 'The teacher’s five-step checklist becomes the evaluation loop.', title_color='green')
    b.person(150, 220, 'green', scale=2.4, label='Teacher / evaluator')
    doc_stack(b,210,475,'orange',n=1,w=80,h=90)
    rows=['Compare student answer with the correct answer','Apply scoring rules (metrics)','Give marks for each question','Calculate the total score','Provide feedback']
    for i,t in enumerate(rows):
        y=130+i*92
        mark(b,360,y+35,True,24)
        b.card(400,y,650,72,t,[],'green',size=15)
    return b


@board
def s2_dataset_flow():
    b = Board(1300, 850, 'Build an evaluation dataset from real RAG runs', 'Golden inputs and expected answers are combined with actual output and actual retrieved context.')
    g=b.card(45,140,310,160,'GOLDEN',['1 · Input: question / query','2 · Expected output:','ground-truth answer'],'yellow',size=14)
    b.group(550,100,700,240,'RAG application','blue')
    r=b.card(600,170,240,90,'Retriever',['Retrieved documents'],'teal',size=13)
    l=b.card(960,170,240,90,'LLM',['Actual model answer'],'purple',size=13)
    b.arrow(g.right(),r.left(),label='input',color='blue'); b.arrow(r.right(),l.left(),label='context',color='teal')
    tc=b.card(800,445,450,170,'LLMTestCases',['Input · question','Expected output · ground truth','Actual output · model answer','Retrieved context · documents'],'purple',size=14)
    b.arrow(g.bottom(),tc.left(.3),via=[(200,465)],label='question + expected output',color='yellow')
    b.arrow(r.bottom(),tc.top(.15),via=[(720,390),(867,390)],label='retrieved context',color='teal')
    b.arrow(l.bottom(),tc.top(.8),label='actual output',color='purple')
    ds=b.cylinder(850,695,350,110,'DATASET',['Store the test cases','Evaluate different metrics'],'green',size=14)
    b.arrow(tc.bottom(),ds.top(),label='store',color='green')
    b.table(40,555,[210,290],[['What flows in?','Source'],['Input','Golden'],['Expected output','Golden'],['Actual output','RAG application'],['Retrieved context','Same RAG run']],header_color='teal',size=12,row_h=42)
    return b


@board
def s2_judge_flow():
    b = Board(1360, 650, 'Evaluation flow · LLM as judge', 'A test case is judged with each metric’s own criteria, prompts and rules.', title_color='purple')
    b.group(20,120,330,380,'Test case from dataset','blue')
    for i,(t,c) in enumerate([('Question / user query','blue'),('Actual output / model answer','purple'),('Expected output / ground truth','green'),('Context / retrieved documents','teal')]):
        b.card(40,175+i*75,290,60,t,[],c,size=12)
    b.text(185,535,'e.g. 1000 questions',16,'blue','700')
    j=b.card(420,230,240,155,'LLM as judge',['Apply the evaluation','rules to this test case'],'purple',size=13)
    brain(b,540,170,'purple',30)
    b.card(420,450,240,90,'Human reviewer / expert',['Alternative shown at 3:21'],'orange',size=12)
    m=b.card(730,165,330,290,'METRICS',['Answer relevancy','Faithfulness (groundedness)','Context relevancy','Context precision','Context recall','…'],'orange',size=14)
    s=b.card(1130,195,210,240,'SCORE',['Metric-wise score','Overall score','Pass / fail','Feedback / reason'],'green',size=13)
    b.arrow((350,310),j.left(),color='blue'); b.arrow(j.right(),m.left(),color='purple'); b.arrow(m.right(),s.left(),color='orange')
    b.text(880,500,'Each metric has its own criteria.',12,'orange','700')
    b.text(880,522,'Only then aggregate across cases.',12,FAINT)
    b.card(80,580,1200,48,'Final evaluation result: scores plus reasons that can be inspected',[],'grey',size=13)
    return b


@board
def s2_goldens_metrics():
    b=Board(1140,460,'Goldens and metrics','The expected answer and the scoring rule play different roles in an evaluation.')
    b.card(30,115,510,300,'Goldens',['Carefully curated test cases','with expected or correct information.','','Input / question','Expected output / ground truth','Optional reference context or metadata'],'blue',size=16)
    b.card(600,115,510,300,'Metrics',['Predefined quality criteria','and scoring rules.','','Context relevancy · precision · recall','Answer relevancy','Faithfulness'],'purple',size=16)
    return b


@board
def s2_evaluation_schedule():
    b=Board(1400,930,'Where evaluation fits and when to run','Offline evaluation supports frequent checks, release checks and continuous improvement.',title_color='red')
    b.group(20,105,265,405,'RAG application','blue')
    b.person(65,210,'blue',scale=.8,label='User')
    r=b.card(130,200,130,70,'RAG pipeline',[],'blue',size=11)
    b.arrow((90,240),r.left(),color='blue')
    kb=b.cylinder(110,355,150,90,'Knowledge base',[],'teal',size=10)
    b.arrow(kb.top(),r.bottom(),both=True,color='teal')
    b.text(194,310,'→ answer',13,'blue','700')
    g=b.card(325,150,260,290,'Golden dataset',['Prepared in advance','','Input / question','Expected output','Optional context / reference','Metadata / tags'],'yellow',size=13)
    e=b.card(630,150,290,290,'Offline evaluation',['1 · Run golden inputs','2 · Collect outputs and','    retrieved context','3 · Evaluate with metrics','    (LLM as judge)','4 · Scores and reports'],'red',size=13)
    un=b.card(990,115,385,170,'Unit / nightly · frequent',['Small set · key metrics','Fast · low cost','Nightly / per commit'],'teal',size=14)
    co=b.card(990,315,385,170,'Comprehensive · less frequent',['Large set · many metrics','Slower · higher cost','Before major releases'],'orange',size=14)
    b.arrow((285,290),g.left(),color='blue'); b.arrow(g.right(),e.left(),color='yellow')
    b.arrow(e.right(.25),un.left(),dashed=True,color='teal'); b.arrow(e.right(.75),co.left(),dashed=True,color='orange')
    b.card(260,540,890,80,'Outcomes',['Track quality over time · detect regressions · compare versions · data for improvement'],'green',size=13)
    b.arrow(un.right(),(1150,580),via=[(1385,200),(1385,580)],color='teal')
    b.arrow(co.bottom(),(1080,540),color='orange')
    ch=b.card(20,740,190,70,'Code / config changes',[],'blue',size=12)
    ci=b.card(260,740,180,70,'CI / CD pipeline',[],'blue',size=13)
    d=b.diamond(565,775,165,110,'Evaluation\ntype?',size=13)
    small=b.card(730,675,275,65,'Unit / nightly',['Small change / daily'],'teal',size=12)
    full=b.card(730,825,275,65,'Comprehensive',['Major release / milestone'],'orange',size=12)
    res=b.card(1115,735,260,90,'Results stored',['Dashboard / alerts'],'green',size=13)
    b.arrow(ch.right(),ci.left()); b.arrow(ci.right(),d.left()); b.arrow(d.right(),small.left(),color='teal'); b.arrow(d.right(),full.left(),color='orange')
    b.arrow(small.right(),res.left(.3),color='teal'); b.arrow(full.right(),res.left(.7),color='orange')
    b.arrow(full.bottom(),ci.bottom(),via=[(867,915),(350,915)],dashed=True,color='orange',label='feedback into CI / CD')
    return b


@board
def s2_golden_synthesizer():
    b=Board(1200,470,'Generate synthetic goldens','Two source figures shown in DeepEval’s Golden Synthesizer documentation.',title_color='teal')
    for i,(name,col) in enumerate([('Documents','blue'),('Existing goldens','purple')]):
        y=135+i*160
        a=b.card(30,y,240,80,name,[],col,size=16)
        c=b.card(390,y,260,80,'Contexts',[],col,size=16)
        s=b.card(800,y,365,95,'Synthetic goldens',['Input + expected output'],col,size=15)
        b.arrow(a.right(),c.left(),color=col); b.arrow(c.right(),s.left(),color=col)
    b.text(600,435,'Four routes in the menu: documents · contexts · goldens · scratch',14,FAINT)
    return b


# ------------------------------------------------------ Retrieval and deployment

@board
def s2_jina_reranking():
    b=Board(1280,650,'Retrieve broadly, then rerank','Jina’s three steps: initial retrieval → detailed query-document scoring → improved results.',title_color='teal')
    q=b.card(420,115,230,65,'Query',[],'blue',size=15)
    doc_stack(b,60,300,'grey',w=85,h=105,label='Full database')
    se=b.card(310,300,260,95,'BM25 / vector search',['1 · Initial retrieval'],'blue',size=14)
    top=b.card(665,280,245,95,'Relevant top-N docs',[],'orange',size=14)
    rest=b.card(665,465,245,95,'Rest of documents',['Discarded from candidates'],'grey',size=12)
    rr=b.card(1010,280,245,110,'Reranker',['2 · Query + document','interaction'],'green',size=14)
    out=b.card(1010,480,245,95,'Sorted top-k docs',['3 · Improved results','k ≤ N'],'teal',size=14)
    b.arrow((170,350),se.left()); b.arrow(q.bottom(),se.top(),color='blue'); b.arrow(se.right(.3),top.left(),color='orange')
    b.arrow(se.right(.8),rest.left(),color='grey'); b.arrow(top.right(),rr.left(),color='orange'); b.arrow(rr.bottom(),out.top(),color='green')
    b.arrow(q.right(),rr.top(),via=[(1132,147)],label='same query',color='green')
    return b


@board
def s2_voyage_ecosystem():
    b=Board(1200,580,'Search and retrieval ecosystem','Voyage AI’s visual pipeline connects embeddings, retrieval, reranking and generation.',title_color='purple')
    labels=[('Unstructured data','blue'),('Embedding model','purple'),('Vector DB','teal'),('Reranker','purple'),('Relevant files','teal'),('LLM','orange')]
    coords=[(40,150),(430,150),(820,150),(820,350),(430,350),(40,350)]
    boxes=[]
    for (name,col),(x,y) in zip(labels,coords): boxes.append(b.card(x,y,300,80,name,[],col,size=17))
    for i in range(5):
        a,z=boxes[i:i+2]
        if i<2: b.arrow(a.right(),z.left(),color='purple')
        elif i==2: b.arrow(a.bottom(),z.top(),color='purple')
        else: b.arrow(a.left(),z.right(),color='purple')
    b.card(220,490,760,60,'Factual responses with lower costs',[],'green',size=16)
    b.arrow(boxes[-1].bottom(),(220,520),via=[(190,520)],color='orange')
    return b


@board
def s2_aws_architecture():
    b=Board(1600,1190,'Enterprise RAG on AWS · deployment board','Route split, dependency protocols, model fallback and the management plane shown in the session.',title_color='orange')
    u=b.card(35,115,210,65,'User / browser',[],'blue',size=14)
    b.group(295,95,1260,155,'AWS edge','orange')
    dns=b.card(325,145,260,75,'Route 53 DNS',['optional'],'orange',size=13)
    alb=b.card(805,145,500,75,'Application Load Balancer',['Public subnets · :80 / :443'],'orange',size=14)
    b.arrow(u.right(),dns.left(),label='HTTPS',color='blue'); b.arrow(dns.right(),alb.left(),color='orange')
    b.group(360,310,800,230,'AWS compute · ECS cluster on Fargate','pink')
    api=b.card(400,405,340,85,'rag-api task',['uvicorn 0.0.0.0:8080'],'pink',size=13)
    ui=b.card(800,405,320,85,'rag-ui task',['streamlit 0.0.0.0:8501'],'pink',size=13)
    b.arrow(alb.bottom(.2),api.top(),via=[(905,278),(570,278)],label='/query /health /ready /metrics',color='orange')
    b.arrow(alb.bottom(.7),ui.top(),via=[(1155,280),(960,280)],label='/ui*',color='orange')
    b.arrow(ui.left(),api.right(),label='BACKEND_URL',color='pink')
    b.group(25,575,725,280,'External LLM services','purple')
    pk=b.card(45,655,270,80,'Portkey gateway',['routing + fallback'],'purple',size=13)
    oa=b.card(400,620,320,70,'OpenAI primary',['gpt-5-mini'],'green',size=13)
    an=b.card(400,745,320,70,'Anthropic fallback',['claude-haiku-4-5'],'orange',size=13)
    b.arrow(pk.right(.25),oa.left(),label='primary',color='green'); b.arrow(pk.right(.75),an.left(),label='fallback',color='orange',dashed=True)
    b.arrow(api.left(),pk.top(),via=[(180,447)],label='gateway request',color='purple')
    ji=b.card(30,345,250,75,'Jina AI',['embeddings + reranker'],'teal',size=13)
    b.arrow(api.left(.2),ji.right(),label='embedding / rerank',color='teal')
    b.group(805,575,750,280,'Managed state','teal')
    ne=b.card(830,650,210,160,'Neon PostgreSQL',['LangGraph','checkpointer','conversation memory'],'teal',size=12)
    up=b.card(1070,650,210,160,'Upstash Redis',['rate-limiting store'],'teal',size=12)
    qd=b.card(1310,650,220,160,'Qdrant Cloud',['vector DB','enterprise_rag','collection'],'teal',size=12)
    b.arrow(api.bottom(.6),ne.top(),via=[(604,555),(935,555)],label='SQL over TLS',color='teal')
    b.arrow(api.bottom(.75),up.top(),via=[(655,565),(1175,565)],color='teal')
    b.text(1175,635,'REST API',11,'teal','700')
    b.arrow(api.right(.2),qd.top(),via=[(770,422),(770,550),(1420,550)],color='teal')
    b.text(1420,635,'gRPC / HTTP over TLS',10,'teal','700')
    b.group(25,890,1530,270,'CI / CD and AWS management · both tasks share these connections','grey')
    gh=b.card(45,955,220,80,'GitHub Actions',['ci.yml + cd.yml'],'blue',size=12)
    ec=b.card(360,945,255,100,'ECR',['enterprise-rag:latest'],'orange',size=12)
    tasks=b.card(715,940,300,55,'rag-api + rag-ui tasks',[],'pink',size=12)
    b.arrow(gh.right(),ec.left(),label='build & push',color='blue'); b.arrow(ec.right(),tasks.left(),label='pull image',color='orange')
    services=[('Secrets Manager',[], 'read secrets'),('IAM roles',['ecsTaskExecutionRole','rag-api-task-role','rag-ui-task-role'],'assume role'),('CloudWatch Logs',['/ecs/rag-api','/ecs/rag-ui'],'write logs'),('Prometheus /metrics',[], 'expose metrics')]
    for i,(title,lines,label) in enumerate(services):
        x=645+i*225
        c=b.card(x,1045,210,95,title,lines,'grey',size=10)
        b.arrow(tasks.bottom(.1+i*.26),c.top(),color='grey',width=1.2)
        b.text(x+105,1030,label,10,'grey','700')
    return b


# ------------------------------------------------------ Document parsing

@board
def s2_parsing_paradigms():
    b=Board(1440,1040,'Three paradigms for visual documents','One page image can enter layout + recognition, a single VLM, or visual retrieval.',title_color='pink')
    root=b.card(440,105,560,75,'Document input · PDF / image',['Page image'],'pink',size=15)
    specs=[(20,'1 · Dual-stage','blue'),(500,'2 · Single-stage VLM','green'),(980,'3 · Vision retrieval','orange')]
    for x,t,c in specs: b.group(x,240,440,770,t,c)
    a=b.card(45,300,390,145,'Stage 1 · layout detection',['PP-DocLayout-V3 / DocLayout-YOLO','23 region types','Bounding boxes + reading order'],'blue',size=14)
    cr=b.card(60,505,360,80,'Region cropping + prompt routing',[],'blue',size=14)
    rr=b.card(45,645,390,190,'Stage 2 · region recognition',['GLM-OCR 0.9B / PaddleOCR-VL / MinerU2.5','Text → Text Recognition:','Table → Table Recognition:','Formula → Formula Recognition:'],'blue',size=13)
    out=b.card(55,910,370,70,'Structured output',['Markdown · LaTeX · HTML tables'],'blue',size=13)
    v=b.card(525,330,390,260,'Single VLM forward pass',['Nemotron Parse 1.1 / dots.ocr','olmOCR-2 / Chandra-OCR','GOT-OCR 2.0 / Nougat','','Full page → structured output directly'],'green',size=14)
    vo=b.card(545,740,350,150,'Output formats',['Markdown / JSON / LaTeX','Optional bounding boxes + classes'],'green',size=14)
    e=b.card(1005,300,390,190,'Vision encoder',['ColPali / ColQwen2.5 / ColSmolVLM','Per-patch embeddings','from the raw page image'],'orange',size=14)
    mv=b.card(1015,580,370,150,'Multi-vector embeddings',['Late interaction scoring','MaxSim / ColBERT-style'],'orange',size=14)
    rp=b.card(1015,820,370,145,'Retrieved pages / patches',['Ranked by visual similarity','Feed to LLM for answer'],'orange',size=14)
    for first,col in [(a,'blue'),(v,'green'),(e,'orange')]: b.arrow(root.bottom(),first.top(),color=col)
    for x,y,col in [(a,cr,'blue'),(cr,rr,'blue'),(rr,out,'blue'),(v,vo,'green'),(e,mv,'orange'),(mv,rp,'orange')]: b.arrow(x.bottom(),y.top(),color=col)
    return b


@board
def s2_colpali_architecture():
    b=Board(1360,830,'ColPali architecture · late interaction','Offline page patch vectors meet online query token vectors at MaxSim.',title_color='purple')
    b.group(20,105,650,440,'Offline · index document pages','blue')
    doc_stack(b,70,195,'blue',w=75,h=95,label='Page image / patches')
    ve=b.card(240,170,370,70,'Vision encoder',[],'blue',size=15)
    ll=b.card(240,280,165,60,'LLM',[],'purple',size=15)
    pr=b.card(445,280,165,60,'Projection',[],'purple',size=14)
    dv=b.card(240,410,370,90,'Patch vectors 1 … N_d',['Multi-vector document representation'],'teal',size=13)
    b.arrow((170,240),ve.left(),color='blue'); b.arrow(ve.bottom(.2),ll.top(),color='blue'); b.arrow(ll.right(),pr.left(),color='purple'); b.arrow(pr.bottom(),dv.top(.75),color='purple')
    b.group(710,105,630,440,'Online · query','orange')
    q=b.card(775,170,500,70,'Query tokens',['What · are · ViTs?'],'orange',size=14)
    ql=b.card(800,280,180,60,'LLM',[],'purple',size=15)
    qp=b.card(1070,280,180,60,'Projection',[],'purple',size=14)
    qv=b.card(800,410,450,90,'Query token vectors',['One vector per query token'],'orange',size=14)
    b.arrow(q.bottom(.25),ql.top(),color='orange'); b.arrow(ql.right(),qp.left(),color='purple'); b.arrow(qp.bottom(),qv.top(.8),color='purple')
    ms=b.card(335,610,690,90,'For each query token: MaxSim over page patches',['Keep the best patch match for that query token.'],'green',size=15)
    b.arrow(dv.bottom(),ms.top(.2),color='teal'); b.arrow(qv.bottom(),ms.top(.8),color='orange')
    z=b.card(430,755,500,50,'Sum of MaxSims = page similarity score',[],'green',size=14)
    b.arrow(ms.bottom(),z.top(),color='green')
    return b


@board
def s2_layout_model():
    b=Board(1450,680,'PP-DocLayoutV3 · model architecture','The legible upper row of the model-card figure shown in the session.',title_color='blue')
    p=b.card(25,275,170,90,'Page image',[],'blue',size=14)
    bb=b.card(245,275,220,90,'Backbone',['PP-HGNetV2'],'blue',size=14)
    tr=b.card(515,250,245,140,'Encoder / decoder',['Transformer layers'],'purple',size=14)
    b.arrow(p.right(),bb.left(),color='blue'); b.arrow(bb.right(),tr.left(),color='blue')
    for i,(n,col) in enumerate([('Class','green'),('Box','orange'),('Mask','teal'),('Order','purple')]):
        h=b.card(830,130+i*125,175,70,n+' head',[],col,size=13)
        pp=b.card(1090,130+i*125,310,70,n+' post-process',['Raw prediction / logits'],col,size=12)
        b.arrow(tr.right(),h.left(),color=col); b.arrow(h.right(),pp.left(),color=col)
    b.card(890,600,505,55,'Filtering → final labelled regions + reading order',[],'green',size=13)
    b.text(395,560,'Four prediction heads preserve different layout properties.',12,FAINT)
    b.text(395,590,'The lower reading-order detail is too small to verify',12,FAINT)
    b.text(395,612,'in the recording and is omitted here.',12,FAINT)
    return b


@board
def s2_ocr_repository():
    b=Board(1250,600,'Nemotron-Parse + Mistral OCR · repository architecture','The visible README tree starts with PDF parsing and names the separate ingestion pipeline.',title_color='teal')
    pdf=b.card(30,230,180,90,'PDF',[],'blue',size=18)
    np=b.card(335,120,550,180,'nemotron_parse_pipeline.py',['Rasterise each page to JPEG','Auto-zoom: 1024×1280 to 1648×2048 px','POST nvidia/nemotron-parse via NVIDIA NIM'],'teal',size=14)
    out=b.card(980,150,240,135,'Results',['*_raw.json','*_parsed.md'],'green',size=14)
    ing=b.card(335,395,865,140,'scripts/ingest.py · phases 1–5',['README overview: semantic chunks → figure captions →','NVIDIA NIM embeddings → Qdrant → grounded Q&A'],'purple',size=14)
    b.arrow(pdf.right(),np.left(),color='teal'); b.arrow(pdf.bottom(),ing.left(),via=[(120,465)],color='purple'); b.arrow(np.right(),out.left(),color='green')
    b.text(775,575,'The ingestion phase details were below the fold; this line summarises the visible README introduction.',11,FAINT)
    return b


@board
def s2_multimodal_pipeline():
    b=Board(1340,1100,'Multimodal RAG · four repository phases','Raw PDF → clean Markdown → hybrid vector search → reranked answers.',title_color='teal')
    b.pill(670,100,'PDF / image','blue',size=15,anchor='middle')
    b.group(30,165,1280,190,'Phase 1 · parse','blue')
    l=b.card(65,225,325,90,'PP-DocLayout-V3',['23 element categories'],'blue',size=14)
    o=b.card(495,225,325,90,'GLM-OCR 0.9B',['Text · tables · formulas'],'blue',size=14)
    c=b.card(925,225,345,90,'Structure-aware chunker',['RAG-ready chunks'],'blue',size=14)
    b.arrow((670,135),o.top(),color='blue'); b.arrow(l.right(),o.left(),color='blue'); b.arrow(o.right(),c.left(),color='blue')
    b.group(30,410,1280,285,'Phase 2 · ingest','purple')
    cap=b.card(65,475,320,110,'GPT-4o image captioner',['Enrich figure chunks'],'purple',size=14)
    em=b.card(495,475,325,85,'Pluggable dense embeddings',['OpenAI · Gemini'],'purple',size=13)
    bm=b.card(495,590,325,75,'BM25 sparse vectors',['Feature hashing'],'orange',size=13)
    q=b.cylinder(940,500,320,140,'Qdrant hybrid store',['Dense + sparse','RRF fusion'],'teal',size=14)
    b.arrow(c.bottom(),cap.top(),via=[(1097,380),(225,380)],color='blue'); b.arrow(cap.right(),em.left(),color='purple'); b.arrow(em.right(),q.left(.3),color='purple'); b.arrow(bm.right(),q.left(.75),color='orange')
    b.group(30,750,1280,165,'Phase 3 · search and rerank','orange')
    hs=b.card(95,800,450,80,'Hybrid Qdrant search',['Top-k candidates'],'orange',size=14)
    rr=b.card(750,800,490,80,'Reranker',['OpenAI · Jina · BGE · Qwen'],'orange',size=14)
    b.arrow(q.bottom(),hs.top(),via=[(1100,723),(320,723)],color='teal'); b.arrow(hs.right(),rr.left(),color='orange')
    api=b.card(365,975,610,80,'Phase 4 · REST API',['FastAPI'],'green',size=16)
    b.arrow(rr.bottom(),api.top(),color='green')
    b.text(670,1080,'The live demo stops after parser inspection; later phases describe the repository.',12,FAINT)
    return b



def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
