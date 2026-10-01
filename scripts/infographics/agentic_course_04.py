"""Infographics for docs/projects/agentic-ai-complete-course/04-rag.md.

Chapter 4 of "Complete Agentic AI Course in 10 Hours" (Krish Naik, section
5:02:29 to 7:10:43). Boards redraw the instructor's whiteboard pages as original
images; boards flagged "added" are explanatory boards that are not in the video.
Run from the repo root:

    python3 scripts/infographics/agentic_course_04.py                # all boards
    python3 scripts/infographics/agentic_course_04.py llm_limits     # just one
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


def mono(b, x, y, text, size=12, color=INK, weight="400", anchor="start"):
    fill = PALETTE[color]["text"] if color in PALETTE else color
    b.parts.append(
        f'<text xml:space="preserve" x="{x}" y="{y}" text-anchor="{anchor}" font-family="{MONO}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(text)}</text>'
    )


def chips(b, x, y, items, color="grey", gap=8, size=11):
    """A row of pills starting at x; returns the x after the last pill."""
    for t in items:
        box = b.pill(x, y, t, color, size)
        x = box.x + box.w + gap
    return x


def sticker(b, x, y, w, h, text, color="yellow", size=12):
    """A small note card with a dashed border (an aside on the board)."""
    return b.card(x, y, w, h, "", text.split("\n"), color, size=size, dashed=True)


# ------------------------------------------------------------------ 1. llm limits


@board
def llm_limits():
    """Shown at 5:03:45 to 5:10:00: a plain LLM app and its two disadvantages."""
    b = Board(1100, 690, "A plain LLM app and where it falls short",
              "Redrawn from the instructor's whiteboard at 5:05:00 to 5:10:00")

    b.group(20, 90, 1060, 200, "Generative AI app without RAG", "blue")
    b.person(75, 150, "grey", 1.0, "user")
    q = b.card(150, 150, 190, 70, "query + prompt", ["the prompt is the", "instruction to the LLM"], "orange")
    llm = b.card(420, 140, 200, 90, "LLM", ["trained on billions", "of tokens"], "green", title_size=20)
    out = b.card(720, 150, 180, 70, "output", ["generate the content"], "yellow")
    b.arrow((112, 185), q.left())
    b.arrow(q.right(), llm.left())
    b.arrow(llm.right(), out.left())
    b.pill(420, 245, "e.g. OpenAI GPT-5", "grey")
    sticker(b, 930, 145, 135, 80, "two problems\nfollow from\nthis design", "red")

    # problem 1
    b.group(20, 320, 520, 340, "Disadvantage 1: hallucination", "red")
    # time line
    mono(b, 70, 390, "model released", 12, "grey", "700")
    mono(b, 70, 408, "training data ends 1 Aug", 12, "grey")
    mono(b, 340, 390, "today", 12, "grey", "700")
    mono(b, 340, 408, "31 Aug", 12, "grey")
    line(b, [(70, 440), (480, 440)], INK, 3)
    b.parts.append(f'<circle cx="70" cy="440" r="7" fill="{_col("green")}"/>')
    b.parts.append(f'<circle cx="480" cy="440" r="7" fill="{_col("red")}"/>')
    b.card(70, 462, 410, 56, "a month the model never saw",
           ["events in this gap are unknown to it"], "yellow", size=12)
    b.arrow((275, 518), (275, 556))
    b.card(70, 556, 410, 82, "asked about the gap, it answers anyway",
           ["it would rather sound sure than admit", "it does not know: that is hallucination"],
           "red", size=12)

    # problem 2
    b.group(560, 320, 520, 340, "Disadvantage 2: private data", "orange")
    data = b.card(585, 365, 150, 150, "startup data", ["", "policies", "HR", "finance", "", "not public"], "yellow", size=13)
    ft = b.card(810, 360, 240, 125, "Option A: fine-tune",
                ["billions of parameters", "slow and expensive", "data changes, so repeat"], "red", size=12)
    rg = b.card(810, 525, 240, 110, "Option B: RAG pipeline",
                ["knowledge stays outside", "the model and can be", "updated any day"], "green", size=12)
    b.arrow(data.right(0.25), ft.left(0.5), via=[(770, data.y + data.h * 0.25), (770, ft.cy)])
    b.arrow(data.right(0.8), rg.left(0.5), via=[(770, data.y + data.h * 0.8), (770, rg.cy)])
    mono(b, 660, 585, "the route the", 12, "grey", "700", "middle")
    mono(b, 660, 603, "course takes", 12, "grey", "400", "middle")
    return b


# ------------------------------------------------------------- 2. rag whiteboard


@board
def rag_whiteboard():
    """Shown at 5:11:00 to 5:19:15: the first full RAG drawing."""
    b = Board(1180, 720, "RAG: build a knowledge base, then ask it",
              "Redrawn from the instructor's whiteboard at 5:11:00 to 5:19:15")

    # ingestion
    b.group(20, 90, 1140, 270, "Data ingestion pipeline (done once, and again when data changes)", "orange")
    src = b.card(40, 150, 150, 150, "your data", ["", "pdf", "html", "excel", "sql database", "unstructured"], "yellow", size=12)
    parse = b.card(250, 160, 190, 120, "parsing", ["read the data", "and chunk it", "into pieces"], "pink", size=12)
    emb = b.card(520, 160, 190, 120, "embedding", ["text to vectors", "Gemini, OpenAI,", "Hugging Face ..."], "red", size=12)
    vdb = b.cylinder(800, 150, 150, 140, "vector DB", ["the knowledge", "base"], "purple")
    b.arrow(src.right(), parse.left())
    b.arrow(parse.right(), emb.left(), label="chunks")
    b.arrow(emb.right(), vdb.left(), label="vectors")
    sticker(b, 985, 160, 160, 120, "the LLM was never\ntrained on this data,\nand we do not\nretrain it", "yellow", 12)

    # retrieval
    b.group(20, 390, 1140, 310, "Retrieval pipeline (every time a user asks)", "blue")
    b.person(70, 470, "grey", 1.0, "user")
    qe = b.card(160, 470, 150, 80, "query", ["what is the leave", "policy?"], "orange", size=12)
    qv = b.card(370, 470, 160, 80, "embed query", ["query to vector"], "red", size=12)
    sim = b.card(590, 470, 170, 80, "similarity search", ["against the", "vector DB"], "purple", size=12)
    ctx = b.card(820, 440, 160, 62, "context", ["the matching chunks"], "teal", size=12)
    pr = b.card(820, 530, 160, 62, "+ prompt", ["instruction to LLM"], "orange", size=12)
    llm = b.card(1010, 470, 130, 90, "LLM", ["answers from", "the context"], "green", size=12, title_size=16)
    b.arrow((108, 510), qe.left())
    b.arrow(qe.right(), qv.left())
    b.arrow(qv.right(), sim.left())
    b.arrow(sim.right(), ctx.left(), via=[(790, 510), (790, 471)])
    b.arrow(ctx.bottom(), pr.top())
    b.arrow(pr.right(0.5), llm.left(0.75), via=[(995, pr.cy), (995, 537)])
    b.arrow(ctx.right(0.5), llm.left(0.25), via=[(995, ctx.cy), (995, 492)])
    # link from the vector DB down to the similarity search
    b.arrow(vdb.bottom(0.5), sim.top(0.5), via=[(875, 330), (675, 330)], dashed=True, color="purple",
            label="stored vectors are searched")
    b.arrow((1075, 560), (1075, 650), color="green")
    mono(b, 1075, 672, "final answer", 13, "green", "700", "middle")
    mono(b, 70, 690, "the model's own knowledge still helps; RAG adds the knowledge it lacks", 12, "grey")
    return b


# --------------------------------------------------------------- 3. two pipelines


@board
def two_pipelines():
    """Shown at 5:19:30 to 5:23:00: the tidied two-pipeline page."""
    b = Board(1180, 700, "RAG = two pipelines joined by a vector store",
              "Redrawn from the instructor's whiteboard at 5:19:30 to 5:23:00")

    b.group(20, 90, 480, 580, "Data ingestion pipeline", "orange")
    di = b.card(60, 140, 190, 70, "data ingest", ["pdf, html, excel, db"], "red", size=12)
    dp = b.card(60, 290, 190, 70, "data parsing", ["clean and split"], "green", size=12)
    em = b.card(60, 440, 190, 70, "embedding", ["open source or paid"], "red", size=12)
    b.arrow(di.bottom(), dp.top(), label="read data")
    b.arrow(dp.bottom(), em.top(), label="text to vectors")
    d1 = b.card(300, 148, 175, 54, "document", ["a LangChain Document"], "yellow", size=11)
    d2 = b.card(300, 298, 175, 54, "chunking", ["smaller pieces"], "yellow", size=11)
    d3 = b.card(300, 448, 175, 54, "vectors", ["numbers per chunk"], "yellow", size=11)
    for a, c in ((di, d1), (dp, d2), (em, d3)):
        b.arrow(a.right(), c.left(), dashed=True, color="grey")
    sticker(b, 60, 575, 415, 80, "later topics: chunking strategies,\nsemantic chunker, context engineering,\nembedding choice and optimisation", "yellow", 11)

    # vector store, middle
    vs = b.cylinder(560, 400, 150, 120, "vector store", ["+ retriever"], "purple")
    b.arrow(em.bottom(0.5), vs.left(0.75), via=[(155, 530), (520, 530), (520, 490)], color="red", label="vectors", label_at=0.2)

    b.group(740, 90, 420, 580, "Retrieval pipeline", "blue")
    uq = b.card(780, 140, 150, 70, "user query", ["the question"], "orange", size=12)
    cx = b.card(780, 290, 150, 60, "context", ["relevant chunks"], "teal", size=12)
    pm = b.card(960, 290, 170, 60, "prompt", ["instruction"], "orange", size=12)
    llm = b.card(850, 430, 190, 80, "LLM", ["reads context + prompt"], "green", size=12, title_size=16)
    op = b.card(850, 580, 190, 60, "output", ["generation"], "pink", size=13)
    # query goes to the vector store
    b.arrow(uq.left(), vs.top(0.5), via=[(635, 175)], color="orange", label="embedded query", label_at=0.4)
    b.arrow(vs.right(0.5), cx.left(), via=[(740, 460), (740, 320)], color="teal", label="context", label_at=0.35)
    b.arrow(cx.bottom(), llm.top(0.3))
    b.arrow(pm.bottom(), llm.top(0.7))
    b.arrow(llm.bottom(), op.top())
    # stage names
    b.pill(960, 140, "retrieval", "blue", 11, solid=True)
    mono(b, 1040, 156, "fetch context", 11, "grey")
    b.pill(960, 168, "augmentation", "teal", 11, solid=True)
    mono(b, 1067, 184, "add prompt", 11, "grey")
    b.pill(960, 196, "generation", "pink", 11, solid=True)
    mono(b, 1054, 212, "LLM writes", 11, "grey")
    return b


# ------------------------------------------------------------ 4. ingestion detail


@board
def ingestion_detail():
    """Shown at 5:25:45 to 5:29:15: how a file becomes vectors."""
    b = Board(1180, 640, "From files to a vector store",
              "Redrawn from the instructor's whiteboard at 5:25:45 to 5:29:15")

    src = b.card(30, 220, 140, 120, "data ingest", ["", "pdf", "html", "excel, db"], "yellow", size=12)
    parse = b.card(210, 225, 150, 110, "data parsing", ["", "to the document", "structure:", "metadata +", "content"], "pink", size=12)
    b.arrow(src.right(), parse.left())

    ck = []
    for i in range(4):
        ck.append(b.card(430, 90 + i * 100, 130, 62, f"chunk {i + 1}", [], "pink", size=12))
    for c in ck:
        b.arrow(parse.right(0.5), c.left(), color=INK, width=1.6)

    emb = b.card(660, 190, 180, 120, "embedding", ["", "text to vectors"], "orange", size=13, title_size=16)
    for c in ck:
        b.arrow(c.right(), emb.left(0.5), color="red", width=1.6)
    vdb = b.cylinder(980, 160, 170, 180, "vector DB", ["one record", "per chunk"], "purple")
    b.arrow(emb.right(), vdb.left(), color="red")
    b.arrow(vdb.bottom(), (1065, 440), color="green")
    b.pill(1065, 446, "similarity search", "green", 13, solid=True, anchor="middle")

    # context-size limits
    b.arrow(emb.bottom(), (750, 395))
    ctxs = b.card(615, 400, 270, 70, "context size", ["each model accepts a fixed", "number of tokens"], "yellow", size=12)
    llm = b.card(655, 530, 190, 60, "LLM", ["has a context size too"], "green", size=12)
    b.arrow(ctxs.bottom(), llm.top(), dashed=True, color="grey")
    sticker(b, 900, 495, 250, 100, "why chunk at all: a 100-page\nPDF does not fit into the\nembedding model or the LLM", "red", 12)
    return b


# --------------------------------------------------------- 5. document components


@board
def document_components():
    """Shown at 5:34:15 to 5:36:45 (the notebook's SVG): original redraw."""
    b = Board(1180, 800, "LangChain Document: text plus metadata",
              "Redrawn from the notebook picture shown at 5:34:15 to 5:36:45, with current import paths")

    b.group(20, 90, 1140, 250, "A Document holds two things", "blue")
    code = b.card(40, 135, 430, 190, "creating one by hand", [
        "from langchain_core.documents import Document",
        "",
        "doc = Document(",
        '    page_content="RAG is a technique...",',
        "    metadata={",
        '        "source": "chapter1.pdf",',
        '        "page": 5,',
        '        "date_created": "2025-01-01"',
        "    },",
        ")",
    ], "grey", size=11, align="left")
    pc = b.card(520, 135, 300, 190, "page_content (str)", [
        "the text that gets embedded",
        "and searched",
        "",
        "a page of a PDF, a paragraph,",
        "a row of a CSV, a web page",
    ], "blue", size=12)
    md = b.card(860, 135, 280, 190, "metadata (dict)", [
        "facts about the text",
        "used for filtering, tracking",
        "and citing sources",
        "",
        "any JSON-friendly values",
    ], "green", size=12)
    b.arrow(code.right(0.5), pc.left(0.5))
    mono(b, 840, 235, "+", 30, "grey", "700", "middle")

    b.group(20, 360, 1140, 160, "Metadata fields you will see", "green")
    fields = [
        ("source", "file path or URL"),
        ("page / chunk id", "where in the file"),
        ("author", "who wrote it"),
        ("date_created", "freshness filters"),
        ("file_type", "pdf, txt, csv"),
        ("your own", "department, access level"),
    ]
    x = 40
    for name, what in fields:
        b.card(x, 405, 172, 90, name, [what], "yellow", size=11)
        x += 183

    b.group(20, 540, 1140, 120, "Loaders return Documents (langchain_community.document_loaders)", "purple")
    loaders = [("TextLoader", "one .txt file"), ("PyPDFLoader", "one doc per page"),
               ("PyMuPDFLoader", "per page, richer metadata"), ("CSVLoader", "one doc per row"),
               ("WebBaseLoader", "a web page"), ("DirectoryLoader", "a whole folder")]
    x = 40
    for name, what in loaders:
        b.card(x, 585, 172, 62, name, [what], "purple", size=11, title_size=12)
        x += 183

    b.group(20, 680, 1140, 100, "Splitters turn Documents into smaller Documents (langchain_text_splitters)", "pink")
    splitters = [("CharacterTextSplitter", "one separator"), ("RecursiveCharacterTextSplitter", "tries bigger breaks first"),
                 ("TokenTextSplitter", "counts tokens"), ("SemanticChunker", "splits where meaning shifts")]
    x = 40
    for name, what in splitters:
        b.card(x, 720, 270, 50, name, [what], "pink", size=11, title_size=11)
        x += 280
    return b


# ---------------------------------------------------------- 6. augmented generation


@board
def augmented_generation():
    """Shown at 6:33:15 to 6:35:45: retrieval, augmentation, generation."""
    b = Board(1180, 580, "Retrieval, augmentation, generation",
              "Redrawn from the instructor's whiteboard at 6:33:15 to 6:35:45")

    b.person(75, 270, "grey", 1.0, "user")
    qy = b.card(150, 255, 150, 62, "query", ["a new question"], "orange", size=12)
    qv = b.card(130, 380, 190, 84, "query to vectors", ["same embedding model", "as the ingestion"], "red", size=11, title_size=12)
    vdb = b.cylinder(430, 110, 160, 130, "vector DB", ["already filled by", "the ingestion"], "purple")
    ctx = b.card(640, 150, 180, 62, "context", ["top matching chunks"], "teal", size=12)
    aug = b.card(560, 300, 330, 90, "context + prompt", ["the prompt says how to answer;", "the context says what to use"], "orange", size=12, title_size=15)
    llm = b.card(960, 290, 180, 110, "LLM", ["writes the answer"], "green", size=13, title_size=20)
    b.arrow((112, 285), qy.left())
    b.arrow(qy.bottom(), qv.top())
    b.arrow(qv.right(), vdb.left(0.55), via=[(330, 411), (330, 180)], dashed=True, color="red", label="embedding", label_at=0.15)
    b.arrow(vdb.right(0.5), ctx.left(), dashed=True, color="teal")
    b.arrow(ctx.bottom(), aug.top(0.45), color="teal")
    b.arrow(aug.right(), llm.left(), color="orange")
    b.arrow(llm.bottom(), (1050, 480), color="green")
    mono(b, 1050, 505, "output", 15, "green", "700", "middle")

    b.pill(150, 490, "retrieval", "blue", 13, solid=True)
    b.pill(560, 410, "augmentation", "orange", 13, solid=True)
    b.pill(1000, 540, "generation", "green", 13, solid=True)
    return b


# ------------------------------------------------------------ 7. chunk overlap (added)


@board
def chunk_overlap():
    """Added: what chunk_size=1000 and chunk_overlap=200 mean."""
    b = Board(1180, 600, "Chunking with overlap",
              "Explanatory board (not shown in the video)")

    mono(b, 40, 112, "one long text (a PDF page, a section, a whole file)", 13, "grey", "700")
    b.parts.append(
        f'<rect x="40" y="125" width="1100" height="34" rx="8" fill="{PALETTE["grey"]["fill"]}" '
        f'stroke="{PALETTE["grey"]["stroke"]}" stroke-width="2"/>'
    )
    # three overlapping chunks, scale: 1000 chars = 400 px, overlap 200 chars = 80 px
    cols = ["blue", "green", "orange"]
    starts = [40, 40 + 320, 40 + 640]
    for i, (sx, c) in enumerate(zip(starts, cols)):
        cc = PALETTE[c]
        b.parts.append(
            f'<rect x="{sx}" y="{210 + i * 62}" width="400" height="40" rx="8" fill="{cc["fill"]}" '
            f'stroke="{cc["stroke"]}" stroke-width="2"/>'
        )
        mono(b, sx + 200, 235 + i * 62, f"chunk {i + 1}: up to 1000 characters", 12, c, "700", "middle")
    # overlap bands
    for i in range(2):
        ox = starts[i + 1]
        b.parts.append(
            f'<rect x="{ox}" y="{210 + i * 62}" width="80" height="102" fill="{PALETTE["yellow"]["stroke"]}" '
            f'fill-opacity="0.28" stroke="{PALETTE["yellow"]["stroke"]}" stroke-dasharray="5 4"/>'
        )
    mono(b, 1000, 250, "the yellow band is shared:", 12, "yellow", "700", "middle")
    mono(b, 1000, 268, "the last 200 characters of", 12, "yellow", "400", "middle")
    mono(b, 1000, 286, "one chunk open the next", 12, "yellow", "400", "middle")
    line(b, [(40, 160), (40, 205)], "grey", 1.6, True)
    line(b, [(360, 160), (360, 270)], "grey", 1.6, True)
    line(b, [(680, 160), (680, 330)], "grey", 1.6, True)

    b.group(20, 390, 560, 190, "Why overlap", "yellow")
    mono(b, 40, 440, "a sentence cut at a chunk boundary is", 12)
    mono(b, 40, 460, "still whole in at least one chunk, so", 12)
    mono(b, 40, 480, "retrieval does not lose it at the seam.", 12)
    mono(b, 40, 520, "cost: about 20 percent more chunks to", 12, "red")
    mono(b, 40, 540, "embed and store (200 of 1000).", 12, "red")

    b.group(600, 390, 560, 190, "separators=[\"\\n\\n\", \"\\n\", \" \", \"\"]", "purple")
    steps = [("1", "paragraph break"), ("2", "line break"), ("3", "space (between words)"), ("4", "any character")]
    y = 438
    for n, t in steps:
        b.pill(630, y, f"try {n}", "purple", 12, solid=True)
        mono(b, 720, y + 16, t, 13)
        y += 32
    mono(b, 930, 458, "the splitter tries", 12, "grey")
    mono(b, 930, 476, "the first separator", 12, "grey")
    mono(b, 930, 494, "that keeps pieces", 12, "grey")
    mono(b, 930, 512, "under chunk_size and", 12, "grey")
    mono(b, 930, 530, "falls back only if", 12, "grey")
    mono(b, 930, 548, "it must", 12, "grey")
    return b


# ------------------------------------------------------- 8. notebook classes (added)


@board
def notebook_classes():
    """Added: the three classes of pdf_loader.ipynb and the data they pass."""
    b = Board(1180, 700, "pdf_loader.ipynb: classes and what flows between them",
              "Explanatory board (not shown in the video)")

    b.group(20, 90, 1140, 130, "Ingestion (functions)", "orange")
    a = b.card(40, 135, 230, 70, "process_all_pdfs()", ["folder of PDFs to", "64 page Documents"], "yellow", size=12)
    c = b.card(370, 135, 230, 70, "split_documents()", ["64 pages to", "359 chunks"], "pink", size=12)
    t = b.card(700, 135, 230, 70, "texts = [...]", ["chunk.page_content", "for every chunk"], "grey", size=12)
    b.arrow(a.right(), c.left(), label="Documents")
    b.arrow(c.right(), t.left(), label="chunks")

    b.group(20, 250, 1140, 270, "Three small classes", "blue")
    em = b.card(40, 300, 300, 200, "EmbeddingManager", [
        "model: all-MiniLM-L6-v2",
        "_load_model()",
        "generate_embeddings(texts)",
        "",
        "returns numpy array",
        "shape (n, 384)",
    ], "red", size=12)
    vs = b.card(440, 300, 300, 200, "VectorStore", [
        "ChromaDB PersistentClient",
        "collection: pdf_documents",
        "add_documents(docs, embeddings)",
        "",
        "ids: doc_<uuid>_<i>",
        "saved under data/vector_store",
    ], "purple", size=12)
    rr = b.card(840, 300, 300, 200, "RAGRetriever", [
        "takes a VectorStore and an",
        "EmbeddingManager",
        "retrieve(query, top_k, threshold)",
        "",
        "embeds the query, asks Chroma,",
        "returns a list of dicts",
    ], "teal", size=12)
    b.arrow(t.bottom(0.5), em.top(0.5), via=[(815, 232), (190, 232)], label="texts", label_at=0.5)
    b.arrow(em.right(), vs.left(), label="vectors")
    b.arrow(vs.right(), rr.left(), label="collection")
    b.arrow(em.bottom(0.8), rr.bottom(0.2), via=[(280, 550), (900, 550)], dashed=True, color="red",
            label="same model embeds the query", label_at=0.5)

    b.group(20, 580, 1140, 100, "Each hit returned by retrieve()", "teal")
    chips(b, 45, 625, ["id", "content", "metadata", "similarity_score", "distance", "rank"], "teal", 10, 12)
    mono(b, 1135, 647, "list of dicts, best first", 12, "grey", "400", "end")
    return b


# --------------------------------------------------------- 9. src modules (added)


@board
def src_modules():
    """Added: how the modular src/ package links up."""
    b = Board(1180, 720, "The modular pipeline in src/",
              "Explanatory board (not shown in the video)")

    b.group(20, 90, 1140, 400, "src/ package", "blue")
    dl = b.card(40, 150, 240, 150, "data_loader.py", [
        "load_all_documents(dir)",
        "",
        "PDF, TXT, CSV, Excel,",
        "Word, JSON",
        "returns Documents",
    ], "yellow", size=12)
    ep = b.card(330, 150, 240, 150, "embedding.py", [
        "EmbeddingPipeline",
        "chunk_documents()",
        "embed_chunks()",
        "",
        "MiniLM, 1000 / 200",
    ], "pink", size=12)
    vs = b.card(620, 150, 240, 150, "vectorstore.py", [
        "FaissVectorStore",
        "build_from_documents()",
        "save() / load()",
        "query(text, top_k)",
    ], "purple", size=12)
    se = b.card(910, 150, 230, 150, "search.py", [
        "RAGSearch",
        "loads the store",
        "search_and_summarize()",
        "",
        "Groq chat model",
    ], "green", size=12)
    b.arrow(dl.right(), ep.left(), label="Documents")
    b.arrow(ep.right(), vs.left(), label="chunks + vectors")
    b.arrow(vs.right(), se.left(), label="store")
    # persistence
    disk = b.cylinder(655, 360, 170, 100, "faiss_store/", ["faiss.index", "metadata.pkl"], "purple", size=11)
    b.arrow(vs.bottom(), disk.top(), label="save / load", color="purple")
    llm = b.card(950, 350, 150, 60, "Groq LLM", ["via langchain_groq"], "green", size=11)
    b.arrow(se.bottom(0.5), llm.top(0.5))
    data = b.card(60, 350, 200, 90, "data/", ["pdf/, text_files/ ..."], "grey", size=12)
    b.arrow(data.top(0.5), dl.bottom(0.5))
    env = b.card(930, 430, 190, 44, ".env", ["GROQ_API_KEY"], "red", size=11)
    b.arrow(env.top(0.5), llm.bottom(0.5), dashed=True, color="red")

    b.group(20, 520, 1140, 170, "app.py: the entry point, grew in five steps", "orange")
    steps = [("1", "load and print", "documents"), ("2", "chunk and embed", "print vectors"),
             ("3", "build the FAISS", "store, save it"), ("4", "store.load() and", "store.query()"),
             ("5", "RAGSearch:", "retrieve + summarise")]
    x = 45
    for n, l1, l2 in steps:
        b.card(x, 575, 205, 90, f"step {n}", [l1, l2], "orange", size=12)
        x += 222
    return b


# ---------------------------------------------------------- 10. agentic rag (added)


@board
def agentic_rag_graph():
    """Added (repo notebook agenticrag/1-agenticrag.ipynb): the decide-retrieve-generate graph."""
    b = Board(1100, 560, "Agentic RAG with LangGraph: decide, then maybe retrieve",
              "Explanatory board of the repository notebook (not taught in this section of the video)")
    start = b.pill(70, 262, "START", "grey", 13, solid=True)
    dec = b.card(170, 235, 200, 90, "decide", ["keyword check on", "the question"], "yellow", size=12, title_size=15)
    dm = b.diamond(500, 280, 160, 100, "needs\nretrieval?", "orange", 12)
    ret = b.card(620, 110, 200, 90, "retrieve", ["retriever.invoke()", "fills documents"], "teal", size=12, title_size=15)
    gen = b.card(780, 235, 200, 90, "generate", ["LLM answers with", "or without context"], "green", size=12, title_size=15)
    end = b.pill(859, 400, "END", "grey", 13, solid=True)
    b.arrow((start.x + start.w, start.y + start.h / 2), dec.left())
    b.arrow(dec.right(), dm.left())
    b.arrow(dm.top(), ret.left(), via=[(500, 155)], label="yes", color="teal", label_at=0.5)
    b.arrow(ret.right(0.5), gen.top(0.5), via=[(880, 155)], color="teal")
    b.arrow(dm.bottom(), gen.left(0.8), via=[(500, 370), (720, 370), (720, 307)], label="no", color="orange", label_at=0.5)
    b.arrow(gen.bottom(0.5), end.top(), color="green")
    b.card(40, 420, 520, 110, "state passed between nodes", [
        "question: str",
        "documents: list of Document",
        "answer: str",
        "needs_retrieval: bool",
    ], "grey", size=12, align="left")
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"04-{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
