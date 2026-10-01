"""Infographics for docs/projects/agentic-ai-complete-course/05-vectorless-rag.md.

Redraws the whiteboard pages and slides of the vectorless RAG (PageIndex)
section of "Complete Agentic AI Course in 10 Hours" (7:10:43 to 8:02:11), plus
four explanatory boards that the video only describes aloud. Run from the repo
root:

    python3 scripts/infographics/agentic_course_05.py              # all boards
    python3 scripts/infographics/agentic_course_05.py compare_pipelines
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


def cross(b, cx, cy, s=11, color="red", width=4):
    line(b, [(cx - s, cy - s), (cx + s, cy + s)], color, width)
    line(b, [(cx - s, cy + s), (cx + s, cy - s)], color, width)


def tick(b, cx, cy, color="green", width=3.5):
    line(b, [(cx - 8, cy), (cx - 2, cy + 7), (cx + 9, cy - 8)], color, width)


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


def rule(b, x1, x2, y, color="#ced4da"):
    line(b, [(x1, y), (x2, y)], color, 1.2, dashed=True)


# ------------------------------------------------- 1. 7:13:00 comparison board


@board
def compare_pipelines():
    b = Board(1100, 920, "Traditional vector RAG vs PageIndex (vectorless RAG)",
              "Two pipelines, side by side, as the instructor's board shows them")
    L, R = 290, 810  # column centres
    cw = 300

    def label(y, text):
        rule(b, 60, 1040, y)
        b.text(550, y + 4, f" {text} ", 12, FAINT, "700")

    b.card(L - cw / 2, 90, cw, 40, "Traditional vector RAG", [], "blue")
    b.card(R - cw / 2, 90, cw, 40, "PageIndex (vectorless RAG)", [], "green")

    label(150, "input")
    b.card(L - 110, 165, 220, 54, "PDF document", ["Any long text"], "yellow")
    b.card(R - 110, 165, 220, 54, "PDF document", ["Any long text"], "yellow")

    label(244, "indexing")
    ch = b.card(L - 150, 262, 140, 60, "Chunking", ["Split into pieces"], "blue", size=11)
    em = b.card(L + 10, 262, 140, 60, "Embedding", ["Convert to vectors"], "blue", size=11)
    tb = b.card(R - cw / 2, 262, cw, 60, "LLM tree builder", ["Generates hierarchy of sections"], "green", size=11)
    b.arrow(ch.right(), em.left(), color="green")

    label(346, "storage")
    vdb = b.card(L - 130, 364, 260, 60, "Vector database", ["Pinecone, FAISS, ChromaDB"], "blue", size=11)
    jt = b.card(R - 130, 364, 260, 60, "JSON tree index", ["No vector DB needed"], "green", size=11)

    label(448, "query time")
    uq1 = b.card(L - 110, 466, 220, 46, "User query", [], "yellow")
    uq2 = b.card(R - 110, 466, 220, 46, "User query", [], "yellow")

    label(536, "retrieval")
    eq = b.card(L - 150, 554, 140, 60, "Embed query", ["Convert to vector"], "blue", size=11)
    ann = b.card(L + 10, 554, 140, 60, "ANN search", ["Find similar vectors"], "blue", size=11)
    lts = b.card(R - cw / 2, 554, cw, 60, "LLM tree search", ["Reasons over tree structure"], "green", size=11)
    b.arrow(eq.right(), ann.left(), color="green")

    label(638, "retrieved content")
    fl = b.card(L - 130, 656, 260, 60, "Flat text chunks", ["No structural context"], "blue", size=11)
    ns = b.card(R - 130, 656, 260, 60, "Named sections", ["Title + page + summary"], "green", size=11)

    label(740, "generation")
    g1 = b.card(L - 130, 758, 260, 60, "LLM generates answer", ["No page citation"], "yellow", size=11)
    g2 = b.card(R - 130, 758, 260, 60, "LLM generates answer", ["With section + page citation"], "yellow", size=11)

    s1 = b.card(L - 150, 846, 300, 50, "Similarity search", ["Finds nearest vectors, not best answer"], "orange", size=11)
    s2 = b.card(R - 160, 846, 320, 50, "Reasoning-based retrieval", ["Navigates document like a human expert"], "green", size=11)

    for x, pairs in ((L, [(219, 262), (322, 364), (424, 466), (512, 554), (614, 656), (716, 758), (818, 846)]),
                     (R, [(219, 262), (322, 364), (424, 466), (512, 554), (614, 656), (716, 758), (818, 846)])):
        for y1, y2 in pairs:
            b.arrow((x, y1), (x, y2))

    # his red marks: the query goes to the vector DB, context comes back; no vector DB on the right
    b.arrow((60, 394), vdb.left(), color="red", label="query", label_color="red")
    mono(b, L + 8, 446, "context", 12, "red", "700")
    cross(b, 550, 394, 11, "red")
    mono(b, 550, 420, "no vector DB", 11, "red", "700", "middle")
    return b


# -------------------------------------------- 2. 7:16:00 TOC to tree to JSON


@board
def tree_builder():
    b = Board(1160, 700, "From a table of contents to a JSON tree index",
              "What the LLM tree builder produces, and how a question uses it")

    b.group(20, 95, 270, 300, "1 · The PDF's TOC", "orange")
    toc = b.card(40, 140, 230, 230, "Table of contents", [
        "1  Introduction ..... P1",
        "2  AI .............. P2",
        "   2.1 ML .......... P3",
        "   2.2 DL .......... P4",
    ], "orange", size=12)

    b.group(330, 95, 440, 300, "2 · LLM tree builder: one node per section", "green")
    n1 = b.card(350, 140, 190, 64, "Node 1: Introduction", ["page 1 + summary"], "green", size=11)
    n2 = b.card(590, 140, 160, 64, "Node 2: AI", ["page 2 + summary"], "green", size=11)
    ml = b.card(520, 290, 110, 64, "ML", ["page 3 + summary"], "green", size=10)
    dl = b.card(648, 290, 110, 64, "DL", ["page 4 + summary"], "green", size=10)
    b.arrow(n2.bottom(0.3), ml.top(), color="green")
    b.arrow(n2.bottom(0.7), dl.top(), color="green")
    b.text(380, 330, "each node: the LLM's\nsummary of that\nsection's pages", 11, FAINT, "400")

    b.group(810, 95, 330, 300, "3 · JSON tree index", "purple")
    b.card(830, 140, 290, 160, "", [
        '{ "title": "AI",',
        '  "node_id": "0002",',
        '  "page_index": 2,',
        '  "summary": "...",',
        '  "nodes": [ ML, DL ] }',
    ], "purple", size=12, align="left")
    b.card(830, 316, 290, 60, "Store it anywhere", ["file system, S3, MongoDB"], "grey", size=11)

    b.arrow(toc.right(), (330, toc.cy))
    b.arrow((770, 245), (810, 245), label="")

    b.group(20, 430, 1120, 240, "4 · At question time", "blue")
    q = b.card(40, 500, 180, 60, "User query", ["What is deep learning?"], "yellow", size=12)
    llm = b.card(300, 490, 220, 80, "LLM", ["context = the whole", "JSON tree index"], "blue", size=12)
    walk = b.card(610, 490, 230, 80, "Walk the tree", ["picks node DL, reads its", "title, page and summary"], "green", size=12)
    ans = b.card(930, 490, 200, 80, "LLM answers", ["from that section", "(with a citation)"], "yellow", size=12)
    b.arrow(q.right(), llm.left())
    b.arrow(llm.right(), walk.left(), label="reasons")
    b.arrow(walk.right(), ans.left(), label="context")
    b.text(580, 640, "No embeddings and no vector database: the tree itself is the index.", 13, "green", "700")
    return b


# --------------------------------------------- 3. 7:22:15 TOC detection flow


@board
def toc_flow():
    b = Board(960, 900, "Building the tree when the PDF has no table of contents",
              "TOC detection, section-aware splitting, summaries, hierarchical tree")
    cx = 400
    raw = b.card(cx - 130, 95, 260, 56, "Raw PDF document", ["Any long structured document"], "grey", size=11)
    det = b.card(cx - 150, 190, 300, 60, "TOC detection", ["Scan first N pages for existing headings"], "purple", size=11)
    has = b.card(120, 300, 230, 62, "Parse existing TOC", ["Extract chapter structure"], "green", size=11)
    nots = b.card(450, 300, 230, 62, "LLM reads pages", ["Infers headings + structure"], "orange", size=11)
    split = b.card(cx - 170, 410, 340, 62, "Section-aware splitting", ["Respect logical boundaries, not token counts"], "purple", size=11)
    summ = b.card(cx - 170, 510, 340, 62, "LLM summarises each section", ["Generates node_id, title, page, summary"], "purple", size=11)
    tree = b.card(cx - 170, 610, 340, 62, "Assemble hierarchical tree", ["Parent, child, grandchild nodes"], "green", size=11)

    b.arrow(raw.bottom(), det.top())
    b.arrow((cx - 60, 250), has.top(), via=[(cx - 60, 275), (235, 275)], label="has TOC", label_at=0.35)
    b.arrow((cx + 60, 250), nots.top(), via=[(cx + 60, 275), (565, 275)], label="no TOC", label_at=0.35)
    b.arrow(has.bottom(), (cx - 90, 410), via=[(235, 385), (cx - 90, 385)])
    b.arrow(nots.bottom(), (cx + 90, 410), via=[(565, 385), (cx + 90, 385)])
    b.arrow(split.bottom(), summ.top())
    b.arrow(summ.bottom(), tree.top())

    b.group(120, 700, 570, 170, "output: in-context JSON tree", "grey", label_pos="bottom")
    fs = b.card(cx - 110, 735, 220, 52, "Financial stability", ["node_id 0006 · p.21"], "green", size=11)
    c1 = b.card(150, 805, 230, 52, "Monitoring vulnerabilities", ["node_id 0007 · p.22-28"], "yellow", size=11)
    c2 = b.card(420, 805, 240, 52, "International cooperation", ["node_id 0008 · p.28-31"], "yellow", size=11)
    b.arrow(tree.bottom(), fs.top())
    b.arrow(fs.bottom(0.3), c1.top(0.5), color="green")
    b.arrow(fs.bottom(0.7), c2.top(0.5), color="green")

    b.card(730, 410, 200, 150, "His margin note", ["The LLM splits at", "section boundaries,", "so one section is", "never cut into", "arbitrary pieces."], "orange", size=11, dashed=True)
    b.arrow((730, 485), split.right(), color="orange", dashed=True)
    return b


# ---------------------------------------------- 4. 7:24:45 chunks vs sections


@board
def chunk_vs_section():
    b = Board(1060, 600, "Token-count chunking vs section-aware splitting",
              "The same three-section document cut two ways (the instructor's evenly-split scribble, redrawn)")

    def strip(x0, y0, parts):
        x = x0
        for name, w, color in parts:
            c = PALETTE[color]
            b.parts.append(
                f'<rect x="{x}" y="{y0}" width="{w}" height="52" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="1.6"/>'
            )
            mono(b, x + w / 2, y0 + 31, name, 13, color, "700", "middle")
            x += w

    secs = [("Intro", 90, "blue"), ("AI", 230, "orange"), ("Risks", 80, "purple")]

    b.group(20, 95, 490, 450, "Vector RAG: cut every N tokens", "red")
    strip(50, 160, secs)
    for i, x in enumerate((150, 250, 350)):
        line(b, [(x, 150), (x, 224)], "red", 2.4, dashed=True)
    for i in range(4):
        x0 = 50 + i * 100
        b.card(x0 + 4, 260, 92, 56, f"chunk {i + 1}", [], "red", size=11)
    b.text(265, 345, "The AI section is torn across all four chunks", 12, "red", "700")
    b.card(60, 380, 410, 140, "What retrieval sees", [
        "Only the top-scoring chunks come back.",
        "If chunk 3 wins, the start of the AI",
        "section and the Risks heading are missing.",
        "The model answers from a fragment.",
    ], "red", size=12, dashed=True)

    b.group(550, 95, 490, 450, "PageIndex: split at section boundaries", "green")
    strip(580, 160, secs)
    nx = 580
    for name, w, color in secs:
        b.card(nx + 2, 260, w - 4, 56, "node", [name], "green", size=11)
        nx += w
    tick(b, 665, 345, "green")
    b.text(800, 350, "Every section stays whole", 12, "green", "700")
    b.card(590, 380, 410, 140, "What retrieval sees", [
        "The LLM picks a node, and the whole",
        "section comes back with its title and",
        "page number. Summaries sit on each node",
        "so the choice can be made from the tree.",
    ], "green", size=12, dashed=True)
    return b


# ------------------------------------------------- 5. 7:25:30 retrieval loop


@board
def retrieval_loop():
    b = Board(900, 860, "Retrieval process in vectorless RAG",
              "Read the tree, reason, fetch, check, and loop back if the answer is not there yet")
    cx = 470
    uq = b.card(cx - 130, 95, 260, 50, "User query", [], "grey")
    s1 = b.card(cx - 190, 185, 380, 66, "Step 1: read the tree index", ["LLM scans titles, pages, summaries in context"], "purple", size=11)
    s2 = b.card(cx - 190, 291, 380, 66, "Step 2: reason and select node", ["Returns thinking + node_list JSON"], "purple", size=11)
    s3 = b.card(cx - 190, 397, 380, 66, "Step 3: extract section content", ["Fetch raw pages for the selected node_ids"], "purple", size=11)
    dd = b.diamond(cx, 548, 330, 112, "Sufficient to answer?\nLLM evaluates completeness", "yellow", size=12)
    s4 = b.card(cx + 150, 655, 330, 66, "Step 4: generate answer", ["Cited by section title + page"], "green", size=11)
    xr = b.card(60, 655, 330, 70, "Cross-reference follow", ["'See Appendix G' makes the LLM", "navigate the tree to that node"], "orange", size=11, dashed=True)

    b.arrow(uq.bottom(), s1.top())
    b.arrow(s1.bottom(), s2.top())
    b.arrow(s2.bottom(), s3.top())
    b.arrow(s3.bottom(), dd.top())
    b.arrow(dd.right(), s4.top(), via=[(s4.cx, dd.cy)], label="yes")
    b.arrow(dd.left(), s2.left(), via=[(110, dd.cy), (110, s2.cy)], color="purple", label="no: loop back", label_at=0.5)
    b.arrow(dd.bottom(), xr.top(), via=[(cx, 625), (xr.cx, 625)], color="orange", dashed=True, label="may follow\nreferences", label_at=0.6)
    b.text(cx, 790, "The answer carries its own path: which sections were read, and why.", 13, "green", "700")
    b.text(cx, 815, "Retrieval is a loop of LLM calls, not a single nearest-neighbour lookup.", 12, FAINT)
    return b


# ---------------------------------------------- 6. 7:26:45 PageIndex chat demo


@board
def pageindex_chat():
    b = Board(1160, 520, "PageIndex Chat on a long textbook",
              "What the instructor shows on the hosted chat page: the agent navigates the document with two tool calls")
    q = b.card(30, 130, 210, 120, "You ask", ['"What are the', 'challenges in', 'pattern recognition?"'], "yellow", size=12)
    b.card(30, 280, 210, 70, "Selected document", ["a long pattern-", "recognition textbook"], "grey", size=11)
    t1 = b.card(290, 120, 230, 130, "Tool call 1", ["get document structure", "", "returns the node tree:", "titles, ids, pages"], "purple", size=12)
    t2 = b.card(620, 120, 230, 130, "Tool call 2", ["get page content", "", "parameters: doc name", "and page ranges"], "purple", size=12)
    t3 = b.card(900, 120, 240, 130, "Tool result", ["raw page text for", "just those pages"], "blue", size=12)
    an = b.card(620, 320, 520, 90, "Streamed answer", ["challenges listed, grounded in the", "pages the agent chose to read"], "green", size=12)
    th = b.card(290, 290, 250, 60, "Thought for a few seconds", ["picks which sections to open"], "orange", size=11)
    b.arrow(q.right(), t1.left(), label="")
    b.arrow(t1.right(), t2.left(), label="choose pages")
    b.arrow(t2.right(), t3.left())
    b.arrow(t3.bottom(), (t3.cx, 320))
    b.arrow(t1.bottom(0.5), th.top(0.5), color="orange", dashed=True)
    b.text(580, 470, "No vector store is queried. The agent reads the tree, then reads pages.", 13, "green", "700")
    return b


# --------------------------------------- 7. explanatory: the notebook's code flow


@board
def code_flow():
    b = Board(1160, 640, "The notebook's code flow, function by function",
              "Explanatory board (not shown in the video): what runs once per document and what runs per question")

    b.group(20, 95, 1120, 215, "Once per document (PageIndex cloud builds the tree)", "orange")
    c1 = b.card(40, 150, 245, 130, "1  Set up", ["load_dotenv()", "PageIndexClient(api_key)", "OpenAI(api_key)"], "grey", size=11)
    c2 = b.card(315, 150, 245, 130, "2  Upload", ["pi_client.submit_document(", "  PDF_PATH)", "returns doc_id"], "orange", size=11)
    c3 = b.card(590, 150, 245, 130, "3  Poll", ["pi_client.get_document(", "  doc_id)", "until status is", "'completed'"], "orange", size=11)
    c4 = b.card(865, 150, 255, 130, "4  Fetch the tree", ["pi_client.get_tree(doc_id,", "  node_summary=True)", "pageindex_tree = ..."], "orange", size=11)
    b.arrow(c1.right(), c2.left())
    b.arrow(c2.right(), c3.left())
    b.arrow(c3.right(), c4.left())

    b.group(20, 340, 1120, 260, "Per question (your own OpenAI key)", "blue")
    d1 = b.card(40, 395, 300, 150, "5  llm_tree_search", ["(query, tree)", "compress the tree, ask the LLM,", "JSON back:", "thinking + node_list"], "blue", size=11)
    d2 = b.card(420, 395, 280, 150, "6  find_nodes_by_ids", ["(tree, node_list)", "walk the tree recursively,", "collect the matching nodes"], "blue", size=11)
    d3 = b.card(780, 395, 340, 150, "7  generate_answer", ["(query, nodes)", "build context from node text,", "ask the LLM to answer using only", "it and cite section + page"], "green", size=11)
    b.arrow(d1.right(), d2.left(), label="node ids")
    b.arrow(d2.right(), d3.left(), label="nodes")
    b.arrow(c4.bottom(), d1.top(), via=[(c4.cx, 325), (d1.cx, 325)], label="pageindex_tree", label_at=0.5, color="purple")
    b.text(580, 575, "vectorless_rag(query, tree) is steps 5, 6 and 7 wrapped in one function.", 12, FAINT)
    return b


# ------------------------------------------ 8. explanatory: the tree and a node


@board
def tree_anatomy():
    b = Board(1180, 750, "The tree PageIndex returned for the syllabus PDF",
              "Explanatory board (not shown in the video): 24 top-level sections, 40 nodes in total, one expanded")

    b.group(20, 95, 640, 630, "get_tree() result (excerpt)", "green")
    y = 140
    for nid, title, pg in [("0000", "Preface", 1), ("0001", "MODULE 1", 4), ("0002", "Neural Network Refresher", 4),
                           ("0003", "Hardware", 5), ("0004", "Transformers 101", 6)]:
        b.card(50, y, 580, 32, f"[{nid}] {title}  (p.{pg})", [], "grey", size=12, align="left")
        y += 40
    mono(b, 340, y + 14, ". . .  nodes 0005 to 0009  . . .", 12, FAINT, "400", "middle")
    y += 30
    parent = b.card(50, y, 580, 34, "[0010] Modern LLMs Finetuning  (p.12)", [], "green", size=13, align="left")
    kids = [("0011", "The LLM Development Lifecycle", 12), ("0012", "Pre-Training Deep Dive", 12),
            ("0013", "Data Preparation for Fine-Tuning", 12), ("0014", "Parameter-Efficient Fine-Tuning (PEFT)", 13),
            ("0015", "Supervised Fine-Tuning (SFT)", 13), ("0016", "Preference Alignment", 13),
            ("0017", "The Modern Post-Training Stack", 14), ("0018", "Evaluation", 14),
            ("0019", "Quantization & Deployment Prep", 14), ("0020", "Tooling & Frameworks", 14),
            ("0021", "Synthetic Dataset Generation", 15), ("0022", "Reasoning Models", 15)]
    ky = y + 42
    for nid, title, pg in kids:
        mono(b, 90, ky + 14, f"[{nid}] {title}  (p.{pg})", 12, INK)
        line(b, [(66, y + 34), (66, ky + 9), (82, ky + 9)], "green", 1.4)
        ky += 22
    mono(b, 340, ky + 16, ". . .  down to [0039] Programme Summary  (p.34)", 12, FAINT, "400", "middle")

    b.group(700, 95, 460, 630, "One node, field by field", "purple")
    b.card(724, 140, 412, 190, "", [
        '{',
        '  "title":      "Preface",',
        '  "node_id":    "0000",',
        '  "page_index": 1,',
        '  "summary":    "<LLM summary>",',
        '  "text":       "<section text>",',
        '  "nodes":      [ ...children ]',
        '}',
    ], "purple", size=12, align="left")
    rows = [("node_id", "the handle the LLM returns", "blue"),
            ("title, page_index", "what the LLM reads in the tree", "blue"),
            ("summary", "short LLM summary of the section", "green"),
            ("text", "section text for the answer step", "orange"),
            ("nodes", "children, so the tree can nest", "grey")]
    ry = 360
    for k, v, c in rows:
        b.card(724, ry, 150, 50, k, [], c, size=12)
        mono(b, 890, ry + 30, v, 12, INK)
        ry += 62
    return b


# ----------------------------------- 9. explanatory: why vector retrieval slips


@board
def vector_failures():
    b = Board(1160, 700, "Why vector RAG slips on professional documents",
              "Explanatory board (not shown in the video): two worked failures, scores are illustrative")

    b.group(20, 95, 540, 585, "1 · Chunking destroys context", "red")
    mono(b, 40, 140, "Section 3.2 of a contract, cut in two:", 12, INK, "700")
    c3 = b.card(40, 160, 240, 120, "chunk 3", ['"The supplier is liable', 'for delays, as defined', 'in Section 3.2 ..."'], "blue", size=11)
    c4 = b.card(300, 160, 240, 120, "chunk 4", ['"... except where the', 'delay is caused by a', 'force majeure event."'], "blue", size=11)
    b.pill(40, 300, "retrieved", "green", solid=True)
    b.pill(300, 300, "not retrieved", "red", solid=True)
    cross(b, 520, 190, 9)
    b.card(40, 345, 500, 100, "The model answers from half a rule", [
        "The exception sits in the chunk that scored lower,",
        "so the answer says the supplier is always liable.",
    ], "red", size=12, dashed=True)
    b.card(40, 470, 500, 190, "PageIndex instead", [
        "Section 3.2 is one node. When the LLM picks it,",
        "the rule and its exception arrive together, with",
        "the section title and page number so the answer",
        "can cite them.",
    ], "green", size=12)

    b.group(600, 95, 540, 585, "2 · Similarity is not relevance", "orange")
    mono(b, 620, 140, 'Query: "What are the EBITDA risks?"', 12, INK, "700")
    rowsb = [("Market conditions overview", 0.86, "red", "shares many words"),
             ("Interest-rate commentary", 0.71, "red", ""),
             ("MD&A: EBITDA drivers and risks", 0.64, "green", "the real answer")]
    y = 190
    for name, score, c, note in rowsb:
        mono(b, 620, y, name, 12, INK, "700")
        b.bar(620, y + 12, 330, score, None, c, h=16, label=f"{score:.2f}")
        if note:
            mono(b, 1010, y + 26, note, 11, c, "400")
        y += 78
    mono(b, 620, y + 4, "cosine score ranks the wrong section first", 12, "red", "700")
    b.card(620, 450, 500, 100, "Why it happens", [
        "An embedding captures topical closeness, not",
        "whether this passage answers this question.",
    ], "orange", size=12, dashed=True)
    b.card(620, 570, 500, 90, "PageIndex instead", [
        "The LLM reads titles and summaries and reasons",
        "'EBITDA lives in MD&A', then returns that node.",
    ], "green", size=12)
    return b


# ---------------------------------------------- 10. 7:46:15 reasoning slide


@board
def reasoning_slide():
    b = Board(1180, 600, "Vectorless RAG: reasoning through structure",
              "Skip embeddings entirely, let the LLM navigate the document like a human would")

    b.group(20, 95, 640, 470, "Hierarchical index", "green")
    root = b.card(240, 140, 200, 50, "Annual Report 2024", [], "green", size=13)
    biz = b.card(60, 250, 160, 46, "1. Business", [], "grey", size=12)
    risks = b.card(260, 250, 160, 46, "2. Risks", [], "green", size=12)
    fin = b.card(460, 250, 160, 46, "3. Financials", [], "grey", size=12)
    mk = b.card(200, 360, 110, 44, "Market", [], "grey", size=12)
    cr = b.card(335, 360, 110, 44, "Credit", [], "green", size=12)
    op = b.card(470, 360, 130, 44, "Operational", [], "grey", size=12)
    b.arrow(root.bottom(0.2), biz.top(), color="grey")
    b.arrow(root.bottom(0.5), risks.top(), color="green", width=2.6)
    b.arrow(root.bottom(0.8), fin.top(), color="grey")
    b.arrow(risks.bottom(0.2), mk.top(), color="grey")
    b.arrow(risks.bottom(0.5), cr.top(), color="green", width=2.6)
    b.arrow(risks.bottom(0.8), op.top(), color="grey")
    b.text(340, 450, "LLM navigates: root, chapter, section, answer", 12, "green", "700")
    b.text(340, 480, "Each node has a summary the LLM reads to\ndecide where to go.", 12, FAINT)

    b.group(700, 95, 460, 470, "How the LLM navigates", "purple")
    steps = [("Build the tree", "Parse structure (headings, sections), summarise each node. Done once, offline."),
             ("LLM reads root summary", '"Which chapter is most likely to contain the answer?"'),
             ("Descend the tree", "Repeat at each level until a leaf section is reached."),
             ("Read full section", "No chunking, full context preserved."),
             ("Answer + cite path", "Returns the answer and the navigation path.")]
    y = 140
    for i, (t, d) in enumerate(steps, 1):
        b.card(750, y, 390, 78, t, [d], "purple", size=11, align="left")
        circle_num(b, 730, y + 39, i, "purple")
        y += 88
    return b


# ---------------------------------- 11 and 12. 7:47:00 and 7:48:15 slides


def _strengths_weaknesses(title, subtitle, strengths, weaknesses, accent):
    b = Board(1160, 680, title, subtitle)
    b.group(20, 95, 550, 560, "STRENGTHS", "green")
    b.group(590, 95, 550, 560, "WEAKNESSES", "red")
    y = 135
    for head, desc in strengths:
        c = b.card(44, y, 502, 86, head, [desc], "green", size=11, align="left")
        tick(b, 520, y + 24, "green")
        y += 100
    y = 135
    for head, desc in weaknesses:
        c = b.card(614, y, 502, 86, head, [desc], "red", size=11, align="left")
        cross(b, 1090, y + 24, 8, "red", 3.2)
        y += 100
    return b


@board
def traditional_real_picture():
    return _strengths_weaknesses(
        "Traditional RAG: the real picture",
        "Powerful, but with known failure modes (instructor's slide at 7:47:00)",
        [("Massive scale", "Millions of docs, millisecond lookups"),
         ("Mature ecosystem", "Chroma, FAISS, Pinecone, Qdrant, Weaviate"),
         ("Cheap retrieval", "One embedding + ANN search per query"),
         ("Great for factoids", "Short, lookup-style questions"),
         ("Domain agnostic", "Works on any text: blogs, tickets, PDFs")],
        [("Chunking destroys context", '"As defined in Section 3.2..." is meaningless when retrieved alone'),
         ("Similarity is not relevance", "Embeddings can match wrong things confidently"),
         ("No cross-section reasoning", "Cannot answer: compare risks vs mitigations"),
         ("Hard to explain", "Why was this chunk picked? A cosine score is not an answer"),
         ("Embedding drift", "Model changes mean re-embed everything")],
        "blue")


@board
def vectorless_real_picture():
    return _strengths_weaknesses(
        "Vectorless RAG: the real picture",
        "Different trade-offs: better for some workloads, worse for others (instructor's slide at 7:48:15)",
        [("Preserves document context", "Sections stay whole, no broken references"),
         ("Cross-section reasoning", "The LLM can compare, contrast and synthesise"),
         ("Explainable retrieval", "Returns the navigation path, not a cosine score"),
         ("No embedding pipeline", "Skip the embed / index / refresh complexity"),
         ("Plays well with structure", "Reports, contracts, filings, textbooks shine")],
        [("Higher per-query cost", "Multiple LLM calls to traverse the tree"),
         ("Higher latency", "Several hundred ms to several seconds per query"),
         ("Does not scale to millions", "Works for tens to thousands of docs, not internet scale"),
         ("Needs structured docs", "Random blog posts? The tree adds little value"),
         ("Less mature tooling", "PageIndex and a few others; the ecosystem is young")],
        "green")


# ------------------------------------------------- 13. 7:58:00 side-by-side


@board
def side_by_side():
    b = Board(1100, 560, "Side-by-side: the honest comparison",
              "Eight dimensions that matter in production systems (instructor's slide at 7:58:00)")
    rows = [["Dimension", "Traditional RAG", "Vectorless RAG"],
            ["Scale", "Millions of docs (strong)", "Tens to thousands"],
            ["Latency / query", "Milliseconds (strong)", "Hundreds of ms to seconds"],
            ["Cost / query", "Cheap: 1 embed lookup (strong)", "Higher: multiple LLM calls"],
            ["Cross-section reasoning", "Weak", "Strong"],
            ["Explainability", "Cosine score (opaque)", "Navigation path (strong)"],
            ["Best for", "Factoid Q&A, mixed corpora", "Long structured docs"],
            ["Setup complexity", "Embedding pipeline + DB", "Tree builder, no DB"],
            ["Ecosystem maturity", "Very mature (strong)", "Emerging"]]
    b.table(60, 100, [290, 360, 330], rows, "blue", size=14, row_h=44)
    return b


# ------------------------------------------------- 14. 7:59:15 when to use


@board
def when_to_use():
    b = Board(1180, 720, "Which one to pick",
              "Instructor's two slides: 'Use Traditional RAG when' and 'Use Vectorless RAG when'")
    b.group(20, 95, 560, 600, "Use Traditional RAG when", "blue")
    b.group(600, 95, 560, 600, "Use Vectorless RAG when", "green")
    trad = [("Massive heterogeneous corpora", ["Millions of mixed-format", "docs: blogs, tickets,", "transcripts, knowledge-", "base articles."]),
            ("Latency-critical apps", ["Chatbots, search-as-you-", "type, voice assistants.", "Every millisecond counts."]),
            ("Short factoid queries", ['"What is the warranty', 'period?" "Who is the CEO?"', "The answer lives in one", "chunk."]),
            ("Cost-sensitive at scale", ["Thousands of queries per", "minute: embedding lookups", "cost pennies, LLM tree", "walks do not."])]
    vl = [("Long, structured documents", ["Annual reports, 10-Ks,", "legal contracts, regulatory", "filings, research papers,", "textbooks."]),
          ("Reasoning beats similarity", ['"Compare risk factors in', 'section 7 to mitigations in', 'section 12." Pure similarity', "cannot do this."]),
          ("Explainability is required", ["Compliance, audit, legal,", "financial advisory. Show", "your work, not just the", "answer."]),
          ("Chunking destroys meaning", ['Cross-references like "as', 'defined in section 3.2" lose', "their meaning when", "retrieved alone."])]
    for i, (h, ls) in enumerate(trad):
        x = 44 + (i % 2) * 260
        y = 140 + (i // 2) * 270
        b.card(x, y, 246, 240, h, ls, "blue", size=12)
    for i, (h, ls) in enumerate(vl):
        x = 624 + (i % 2) * 260
        y = 140 + (i // 2) * 270
        b.card(x, y, 246, 240, h, ls, "green", size=12)
    return b


# --------------------------------- 15. explanatory: hybrid and the decision key


@board
def hybrid_pattern():
    b = Board(1160, 620, "Hybrid retrieval: vectors narrow, trees reason",
              "Explanatory board (not shown in the video): the instructor's closing advice, drawn as a pipeline")
    b.group(20, 95, 1120, 250, "Production systems are going hybrid", "purple")
    q = b.card(40, 170, 160, 70, "User query", [], "yellow", size=13)
    v = b.card(250, 150, 240, 120, "1  Vector search", ["over the whole corpus", "(millions of documents)", "keeps the best few"], "blue", size=12)
    t = b.card(590, 150, 260, 120, "2  Tree reasoning", ["inside those documents:", "LLM reads the tree, picks", "sections, reads them whole"], "green", size=12)
    a = b.card(900, 150, 220, 120, "3  Cited answer", ["section title + page", "navigation path kept"], "yellow", size=12)
    b.arrow(q.right(), v.left())
    b.arrow(v.right(), t.left(), label="few docs")
    b.arrow(t.right(), a.left())
    b.text(580, 315, "Scale from vectors, precision and explanation from structure.", 13, "purple", "700")

    b.group(20, 370, 1120, 230, "Picking by document type (the instructor's takeaway 4)", "grey")
    b.card(50, 420, 340, 150, "Long structured filings", ["annual reports, contracts,", "textbooks"], "green", size=12)
    b.card(410, 420, 340, 150, "Mixed knowledge base", ["blogs, tickets, transcripts,", "FAQs"], "blue", size=12)
    b.card(770, 420, 340, 150, "Big system with both", ["a lot of documents, some of", "them long and structured"], "purple", size=12)
    return b


# ------------------------------------------------------------------- runner


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"05-{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
