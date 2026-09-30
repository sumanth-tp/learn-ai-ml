"""Infographics for docs/projects/secure-ehr-insight (the FDE live marathon).

Each function redraws one of the session's whiteboards as an original image:
Monal's handwritten pages (instructor_notes/monal-handwritten-notes.pdf) and
Bappy's Excalidraw deployment sketch. Run from the repo root:

    python3 scripts/infographics/secure_ehr.py                 # all boards
    python3 scripts/infographics/secure_ehr.py ehr_overview    # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import FAINT, INK, MONO, PALETTE, Board, esc  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "secure-ehr"
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
    """A hand-drawn style X, as used on the board for 'this is wrong'."""
    line(b, [(cx - s, cy - s), (cx + s, cy + s)], color, width)
    line(b, [(cx - s, cy + s), (cx + s, cy - s)], color, width)


def strike(b, box, color="red"):
    """Strike a card through, the way Monal crossed entity types out in red."""
    line(b, [(box.x + 8, box.cy), (box.x + box.w - 8, box.cy)], color, 2.4)


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


def circle_num(b, cx, cy, n, color="orange", r=13):
    c = PALETTE[color]
    b.parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="2"/>'
    )
    b.parts.append(
        f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-family="{MONO}" font-size="14" '
        f'font-weight="700" fill="{c["text"]}">{n}</text>'
    )


def dot(b, cx, cy, color="green", r=5):
    b.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{_col(color)}"/>')


def table_icon(b, x, y, w, h, color="grey", rows=4):
    """A little 'rows of data' icon: a box with a header band and row lines."""
    c = PALETTE[color]
    b.parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#ffffff" '
        f'stroke="{c["stroke"]}" stroke-width="1.6"/>'
    )
    b.parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h * 0.22:.1f}" rx="4" fill="{c["fill"]}" '
        f'stroke="{c["stroke"]}" stroke-width="1.2"/>'
    )
    step = (h * 0.78) / (rows + 1)
    for i in range(1, rows + 1):
        yy = y + h * 0.22 + i * step
        line(b, [(x + 6, yy), (x + w - 6, yy)], c["stroke"], 1.2)


def frame(b, x, y, w, h, color="grey", fill="#ffffff", width=2.0, rx=10):
    """An empty outlined rectangle: a window, a chat box, an input field."""
    b.parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{_col(color)}" stroke-width="{width}"/>'
    )


def ellipse(b, cx, cy, rx, ry, color="purple", fill=True, width=2.0):
    c = PALETTE[color]
    f = c["fill"] if fill else "none"
    b.parts.append(
        f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{f}" stroke="{c["stroke"]}" '
        f'stroke-width="{width}"/>'
    )


def mono(b, x, y, text, size=12, color=INK, weight="400", anchor="start"):
    fill = PALETTE[color]["text"] if color in PALETTE else color
    b.parts.append(
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{MONO}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" xml:space="preserve">{esc(text)}</text>'
    )


# ------------------------------------------------------------------ page 1


@board
def ehr_problem():
    b = Board(1200, 790, "Secure EHR Insight & Clinical Validator · Gen-AI",
              "Page 1: what the hospital has, what it wants, and why the obvious answer breaks HIPAA")

    b.text(30, 113, "prerequisite:", 14, "grey", "700", anchor="start")
    x = 160
    for p in ["python", "Database", "RAG", "OOP", "LLM", "workflow", "Agent"]:
        x += b.pill(x, 97, p, "grey", size=12).w + 10

    b.group(20, 140, 1160, 200, "What the hospital has", "blue")
    hosp = b.card(50, 195, 150, 50, "Hospitals", [], "blue")
    ehr = b.card(270, 180, 250, 76, "EHR", ["Electronic Health Record"], "blue")
    rows = b.card(300, 288, 190, 36, "millions of rows", [], "blue", title_size=13)
    b.arrow(hosp.right(), ehr.left())
    b.arrow(ehr.bottom(), rows.top())
    kinds = []
    for i, k in enumerate(["admission", "prescription", "lab results"]):
        kinds.append(b.card(610, 172 + i * 52, 190, 40, k, [], "teal", title_size=13))
    for i, k in enumerate(kinds):
        b.arrow(ehr.right(0.3 + i * 0.2), k.left())
    rel = b.card(890, 205, 250, 70, "relational data", ["tables of rows"], "blue")
    for i, k in enumerate(kinds):
        b.arrow(k.right(), rel.left(0.3 + i * 0.2), color="grey", width=1.4)

    b.card(20, 360, 1160, None, "Problem statement",
           ["An LLM that lets their doctors ask questions in natural language",
            "about patient histories, to save time."], "yellow", size=13)

    b.group(20, 460, 620, 305, "The obvious answer", "red")
    data = b.card(40, 525, 90, 50, "Data", [], "grey")
    rag = b.card(165, 520, 255, 60, "RAG · Text2SQL · Vector DB", [], "orange")
    llm = b.card(480, 520, 90, 60, "LLM", [], "dark")
    b.arrow(data.right(), rag.left())
    b.arrow(rag.right(), llm.left(), label="extract", label_dy=-16)
    cross(b, 603, 550)
    hosp2 = b.card(40, 630, 200, 64, "hospital", ["(raw patient data)"], "blue")
    llm2 = b.card(300, 634, 90, 56, "LLM", [], "dark")
    viol = b.card(250, 715, 190, 34, "violation of HIPAA", [], "red", title_size=13)
    b.arrow(hosp2.right(), llm2.left())
    b.arrow(llm2.bottom(), viol.top())

    hipaa = b.card(660, 590, 110, 70, "HIPAA", ["the rules"], "purple")
    b.arrow(viol.right(), hipaa.bottom(0.3), label="breaks", color="red")

    b.group(820, 460, 360, 305, "Solution", "green")
    sol = [b.card(880, 510, 270, 50, "privacy-first pipeline", [], "green"),
           b.card(880, 580, 270, 50, "guardrails", [], "green"),
           b.card(880, 650, 270, 60, "personal data", ["protected"], "green")]
    tip = brace(b, 866, sol[0].y, sol[2].y + sol[2].h, "green", facing="left")
    b.arrow(tip, hipaa.right(), label="meets", color="green", label_dy=-16)
    return b


# ------------------------------------------------------------------ page 2


@board
def ehr_ai_sdlc():
    b = Board(940, 1100, "AI-SDLC",
              "Page 2: the classic SDLC, then the same stages as 'vibe coding on steroids'")

    b.group(20, 90, 900, 400, "SDLC", "grey")
    proj = b.card(50, 130, 130, 40, "project", [], "grey")
    sdlc = b.card(50, 215, 130, 46, "SDLC", [], "dark")
    b.arrow(proj.bottom(), sdlc.top())
    pm = b.card(225, 215, 170, 46, "product manager", [], "blue")
    wk = b.card(435, 215, 90, 46, "work", [], "blue")
    tk = b.card(565, 215, 110, 46, "tickets", [], "blue")
    dv = b.card(715, 215, 180, 46, "development", [], "blue")
    for a, c in [(sdlc, pm), (pm, wk), (wk, tk), (tk, dv)]:
        b.arrow(a.right(), c.left())
    sp = b.card(565, 295, 330, 38, "sprints", [], "blue", title_size=13)
    b.arrow(wk.bottom(), sp.left(), via=[(wk.cx, sp.cy)])

    stages = [b.card(160, 320, 150, 40, "coding", [], "grey", title_size=13),
              b.card(160, 375, 150, 40, "QA", [], "grey", title_size=13),
              b.card(160, 430, 150, 40, "code review", [], "grey", title_size=13)]
    wt = b.card(350, 375, 160, 40, "write test", [], "grey", title_size=13)
    b.arrow(stages[1].right(), wt.left())
    line(b, [sdlc.bottom(), (115, 505)], "yellow", 2.4)
    for s in stages:
        b.arrow((115, s.cy), s.left(), color="grey")
    b.arrow((115, 500), (115, 522), color="yellow", width=2.4)

    b.group(20, 525, 900, 555, "AI-SDLC · vibe coding on steroids", "yellow")
    spine_top, rows_y = 575, []

    dev = b.card(90, 578, 160, 44, "Developer", [], "yellow")
    plan = b.card(300, 562, 320, None, "plan", ["steps.md · task · edge cases"], "orange")
    b.arrow(dev.right(), plan.left())
    rows_y.append(dev.cy)

    bp = b.card(90, 668, 160, 44, "boilerplate", [], "yellow")
    pmd = b.card(300, 668, 120, 44, "plan.md", [], "orange")
    st = b.card(470, 650, 180, 36, "structure", [], "orange", title_size=13)
    sd = b.card(470, 694, 180, 36, "system design", [], "orange", title_size=13)
    ex = b.card(160, 736, 150, 34, "execution", [], "orange", title_size=13)
    b.arrow(bp.right(), pmd.left())
    b.arrow(pmd.right(0.4), st.left())
    b.arrow(pmd.right(0.6), sd.left())
    b.arrow(bp.bottom(0.15), ex.left(), via=[(bp.x + 24, ex.cy)])
    rows_y.append(bp.cy)

    ai1 = b.card(90, 800, 160, 44, "AI", [], "yellow")
    tests = b.card(300, 800, 380, 44, "unit · test case · report", [], "orange")
    rev = b.card(400, 872, 180, 36, "review", [], "green", title_size=13)
    b.arrow(ai1.right(), tests.left())
    b.arrow(tests.bottom(), rev.top())
    rows_y.append(ai1.cy)

    ai2 = b.card(90, 935, 160, 44, "AI", [], "yellow")
    prs = b.card(300, 935, 100, 44, "PRs", [], "orange")
    sec = b.card(450, 918, 300, 36, "secrets (security flaw)", [], "red", title_size=13)
    perf = b.card(450, 962, 300, 36, "performance regression", [], "red", title_size=13)
    b.arrow(ai2.right(), prs.left())
    b.arrow(prs.right(0.4), sec.left())
    b.arrow(prs.right(0.6), perf.left())
    rows_y.append(ai2.cy)

    mon = b.card(90, 1018, 160, 52, "Monitoring", ["update"], "yellow")
    hot = b.card(300, 1024, 320, 40, "hotfixes, update the doc", [], "orange", title_size=13)
    b.arrow(mon.right(), hot.left())
    rows_y.append(mon.cy)

    line(b, [(55, spine_top), (55, rows_y[-1])], "yellow", 2.4)
    for yy in rows_y:
        b.arrow((55, yy), (90, yy), color="yellow", width=2.2)
    return b


# ------------------------------------------------------------------ page 3


@board
def ehr_overview():
    b = Board(1120, 1280, "In-depth overview",
              "Page 3: the whole system on one board, from the hospital's data to the Streamlit UI")

    emb = b.card(560, 96, 200, 36, "embeddings · note", [], "teal", title_size=13)
    data = b.card(30, 180, 90, 48, "Data", [], "grey")
    ing = b.card(160, 162, 250, 82, "ingest data · Postgres 14", ["EC2", "(hospital DB)"], "blue")
    pgv = b.card(560, 162, 200, 82, "pgvector", ["(add new column)"], "purple")
    b.arrow(data.right(), ing.left())
    b.arrow(ing.right(), pgv.left())
    b.arrow(emb.bottom(), pgv.top())
    circle_num(b, 485, 176, 1)
    b.arrow((485, 262), (485, 210), color="orange")
    b.text(485, 280, "start", 13, "orange", "700")

    # the doctor's side
    b.person(95, 322, "blue", 0.95)
    doc = b.card(30, 408, 140, 58, "doctor", ["healthcare prof"], "blue")
    dd = b.card(215, 415, 140, 44, "drop-down", [], "blue")
    q = b.card(470, 415, 100, 44, "query", [], "orange")
    sel = b.card(200, 500, 170, 54, "select patient", ["from id"], "blue")
    b.arrow(doc.right(), dd.left())
    b.arrow(dd.right(), q.left())
    b.arrow(dd.bottom(), sel.top())
    table_icon(b, 385, 300, 64, 44, "blue")
    b.text(417, 364, "~100 rows", 12, "blue", "700")
    b.arrow(dd.top(0.75), (385, 322), color="blue", width=1.5)
    b.arrow((449, 322), q.top(0.3), color="blue", width=1.5)

    rx = b.card(580, 318, 270, 40, "what medicine should I prescribe?", [], "red", title_size=12)
    b.arrow(q.top(0.8), rx.bottom(0.2), color="red", width=1.5)

    sim = b.card(640, 406, 300, 62, "similarity search on DB", ["(25 mil rows)"], "purple")
    b.arrow(q.right(), sim.left())
    b.arrow(pgv.right(), sim.top(0.867), via=[(900, pgv.cy)], label="search", color="purple")

    # redaction
    b.group(555, 585, 545, 305, "Redaction", "orange")
    cc1 = b.card(810, 495, 250, 46, "My credit card info is: 345", [], "red", title_size=12)
    red = b.card(645, 625, 150, 44, "redaction", [], "orange")
    phi = b.card(895, 625, 170, 44, "remove PHI", [], "red")
    mask = b.card(620, 705, 200, 62, "mask personal data", ["(age, location)"], "orange")
    pres = b.card(875, 711, 200, 50, "Microsoft Presidio", [], "orange")
    cc2 = b.card(595, 805, 250, 58, "My credit card info is:", ["<card-no>"], "green", title_size=12)
    b.arrow(sim.bottom(0.267), red.top())
    line(b, [cc1.left(), (sim.x + 80 + 12, cc1.cy)], "red", 1.4, dashed=True)
    b.arrow(red.right(), phi.left())
    b.arrow(red.bottom(), mask.top(0.5))
    b.arrow(mask.right(), pres.left())
    b.arrow(mask.bottom(), cc2.top(0.5))

    # guardrail
    b.group(380, 925, 720, 190, "Guardrail", "teal", label_pos="bottom")
    gr = b.card(635, 955, 170, 46, "guardrail", [], "teal")
    nemo = b.card(410, 955, 190, 46, "NVIDIA NeMo", [], "teal")
    b.arrow(cc2.bottom(), gr.top(0.5))
    b.arrow(q.bottom(), gr.top(0.2), via=[(q.cx, 905), (gr.x + 34, 905)], label="the question",
            color="orange", label_at=0.35)
    b.arrow(gr.left(), nemo.right())
    ok = b.card(560, 1035, 120, 40, "allowed", [], "green")
    rj = b.card(800, 1035, 120, 40, "reject", [], "red")
    b.arrow(gr.bottom(0.2), ok.top())
    b.arrow(gr.bottom(0.8), rj.top())
    b.arrow(rj.right(), (975, rj.cy), color="red")
    cross(b, 995, rj.cy, 9, width=3)

    ctx = b.card(520, 1150, 330, 90, "Context: ~100 rows", ["~", "query: question"], "orange")
    llm = b.card(345, 1170, 100, 50, "LLM", [], "dark")
    ui = b.card(110, 1165, 170, 60, "Streamlit", ["(UI)"], "pink")
    b.arrow(ok.bottom(), ctx.top(0.3))
    b.arrow(ctx.left(), llm.right())
    b.arrow(llm.left(), ui.right())

    # where HIPAA lives
    b.text(60, 722, "redaction", 14, "orange", "700", anchor="start")
    b.text(60, 752, "guardrail", 14, "teal", "700", anchor="start")
    tip = brace(b, 160, 702, 762)
    hip = b.card(215, 712, 110, 40, "HIPAA", [], "purple")
    b.arrow(tip, hip.left(), color="purple")
    return b


# ------------------------------------------------------------------ page 4


@board
def ehr_phase0():
    b = Board(1120, 720, "Phase 0",
              "Page 4, top: give the client their data. AWS, Postgres, then ingestion from a laptop")

    b.group(20, 90, 440, 262, "AWS · in this order", "orange")
    ec2 = b.card(80, 140, 250, 46, "② EC2 instance", [], "orange")
    sg = b.card(80, 210, 250, 46, "① security group", [], "orange")
    eip = b.card(80, 280, 250, 46, "③ Elastic IP", [], "orange")
    b.arrow(sg.left(), ec2.left(0.65), via=[(62, sg.cy), (62, ec2.y + 30)])
    b.arrow(eip.left(), ec2.left(0.3), via=[(44, eip.cy), (44, ec2.y + 14)])

    b.group(490, 90, 610, 262, "On the instance", "purple")
    attr = b.card(510, 140, 170, 90, "attribute", ["4 GB", "20 GB storage"], "purple")
    launch = b.card(720, 160, 160, 50, "launch instance", [], "purple")
    pg = b.card(910, 140, 175, 90, "install postgres", ["config of postgres"], "purple")
    b.arrow(ec2.right(), attr.left(0.2))
    b.arrow(attr.right(0.4), launch.left())
    b.arrow(launch.right(), pg.left(0.33))

    b.group(20, 380, 760, 230, "Local system", "blue")
    ip = b.card(40, 425, 170, 56, "xx.xxx.xxx", ["the Elastic IP"], "orange")
    loc = b.card(270, 430, 170, 46, "local system", [], "blue")
    ing = b.card(500, 430, 250, 46, "data ingestion (python)", [], "blue")
    steps = b.card(500, 508, 250, None, "", ["schema", "csv → EC2 database", "test"], "blue",
                   align="left", bullets=True)
    b.arrow(ing.left(), loc.right())
    b.arrow(loc.left(), ip.right())
    b.arrow(ing.bottom(), steps.top())
    b.arrow(ip.top(0.5), eip.bottom(0.18), color="orange")

    b.group(800, 380, 300, 230, "Python env · uv", "teal")
    uv1 = b.card(820, 430, 70, 40, "uv", [], "teal")
    genv = b.card(925, 430, 160, 40, "global env", [], "teal")
    uv2 = b.card(820, 510, 110, 40, "uv venv", [], "teal")
    fold = b.card(960, 500, 125, 60, "folder", ["↳ env"], "teal")
    b.arrow(uv1.right(), genv.left())
    b.arrow(uv2.right(), fold.left())

    done = b.card(20, 640, 1080, 50, "Data is ingested", [], "green")
    b.arrow(steps.bottom(), (steps.cx, done.y))

    # Monal's green ticks for what was finished live
    for box in [ec2, sg, eip, attr, launch, pg, ip]:
        dot(b, box.x + box.w - 10, box.y + 10)
    dot(b, 900, 704)
    b.text(912, 709, "ticked off live", 12, "green", "700", anchor="start")
    return b


@board
def ehr_phase1():
    b = Board(1020, 380, "Phase 1",
              "Page 4, bottom: turn the client's Postgres into a vector store")

    inst = b.card(30, 175, 200, 56, "Install pgvector", ["on the EC2 box"], "purple")
    b.group(270, 90, 500, 260, "Database  ·  Vector", "purple")
    t = b.table(290, 140, [200, 260], [
        ["patient_encounters", "clinical_embedding vector(768)"],
        ["subject_id", "NULL"],
        ["drug · dose · route", "NULL"],
        ["test_name · comments", "NULL"],
        ["description · ...", "NULL"],
    ], "purple", size=12)
    b.arrow(inst.right(), (290, inst.cy))
    b.text(520, 325, "a new column beside the existing ones", 12, FAINT)

    llm = b.card(840, 120, 160, 56, "LLM", ["embedding model"], "dark")
    emb = b.card(840, 230, 160, 44, "embedding", [], "teal")
    b.arrow(llm.bottom(), emb.top())
    b.arrow(emb.left(), (t.x + t.w, emb.cy), label="768 numbers", color="teal", label_dy=-16)
    return b


# ------------------------------------------------------------------ page 5


@board
def ehr_phase2():
    b = Board(1160, 600, "Phase 2",
              "Page 5, top: which embedder fills the new column?")

    b.group(20, 90, 500, 480, "EC2", "purple")
    db = b.card(50, 130, 160, 40, "Database", [], "purple")
    t = b.table(50, 210, [190, 250], [
        ["Table: data", "embedding column"],
        ["row", "[ ]"],
        ["row", "[ ]"],
        ["row", "[ ]"],
        ["...", "..."],
    ], "purple", size=12)
    b.arrow(db.bottom(), (db.cx, t.y))

    null = b.card(50, 420, 90, 40, "NULL", [], "red")
    ce = b.card(190, 420, 220, 40, "clinical_embedding", [], "purple", title_size=13)
    nn = b.card(50, 500, 230, 40, "not null anymore", [], "green")
    b.arrow(null.right(), ce.left())
    b.arrow(null.bottom(), (null.cx, nn.y))

    b.group(550, 90, 590, 250, "Which embedder?", "teal")
    mini = b.card(575, 135, 140, 40, "MiniLM", [], "grey", dashed=True)
    qm = b.diamond(645, 250, 100, 64, "?", "yellow", size=22)
    embd = b.card(770, 228, 150, 44, "embedder", [], "teal")
    com = b.card(770, 135, 140, 40, "comments", [], "blue")
    txt = b.card(960, 135, 150, 40, "Text", [], "blue")
    vec = b.card(955, 215, 160, 58, "embedding", ["(768)"], "teal")
    b.arrow(mini.bottom(0.5), qm.top(), dashed=True, color="grey")
    b.arrow(embd.left(), qm.right())
    b.arrow(com.right(), txt.left())
    b.arrow(txt.bottom(), vec.top())
    b.arrow(vec.bottom(), (t.x + t.w, 312), via=[(vec.cx, 312)], label="embedding pushed",
            color="green", label_at=0.25)

    b.group(550, 370, 590, 200, "Healthcare text needs a clinical model", "green")
    hc = b.card(575, 445, 160, 46, "healthcare", [], "green")
    for i, k in enumerate(["medicine", "drug", "disease"]):
        box = b.card(800, 412 + i * 46, 160, 36, k, [], "green", title_size=13)
        b.arrow(hc.right(), box.left())
    return b


@board
def ehr_recap():
    b = Board(1160, 680, "Recap before search",
              "Page 5, bottom: everything built so far, and the shortcut the demo takes")

    b.card(20, 90, 1120, 44, "pushing embeddings takes time", [], "orange")

    b.group(20, 160, 1120, 250, "Done so far", "green")
    row1 = []
    x = 45
    for name, w in [("EC2", 100), ("Postgres", 140), ("Database", 140), ("Table", 110),
                    ("ingested data", 180)]:
        row1.append(b.card(x, 205, w, 46, name, [], "green"))
        x += w + 45
    for a, c in zip(row1, row1[1:]):
        b.arrow(a.right(), c.left())

    row2_spec = [("added vector", "extension to db", 200), ("created a new", "embedding column", 200),
                 ("created embeddings from", "a combination of patient details", 310),
                 ("pushed all", "embeddings to database", 220)]
    row2 = []
    x = 1100
    for title, sub, w in row2_spec:
        x -= w
        row2.append(b.card(x, 305, w, 62, title, [sub], "green", size=11))
        x -= 45
    b.arrow(row1[-1].right(), row2[0].top(), via=[(row2[0].cx, row1[-1].cy)])
    for a, c in zip(row2, row2[1:]):
        b.arrow(a.left(), c.right())

    b.group(20, 450, 560, 200, "For the demo", "blue")
    ec2 = b.card(45, 505, 90, 46, "EC2", [], "blue")
    rows = b.card(185, 495, 290, 66, "11,000 rows of embedding", ["(20 min)"], "blue")
    env = b.card(45, 590, 250, 40, "selected this in .env", [], "yellow", title_size=13)
    b.arrow(ec2.right(), rows.left())
    b.arrow(ec2.bottom(), (ec2.cx, env.y))
    last = row2[-1]
    b.arrow(last.bottom(0.3), ec2.top(), via=[(last.x + last.w * 0.3, 430), (ec2.cx, 430)],
            color="red", dashed=True)
    cross(b, last.x + last.w * 0.3, 392, 8, width=3)
    b.text(last.x + last.w * 0.3 + 18, 397, "not for the demo: 230k rows would take hours", 12, "red",
           "700", anchor="start")

    b.group(610, 450, 530, 200, "Search", "purple")
    qq = b.card(630, 530, 90, 44, "query", [], "orange")
    ee = b.card(750, 530, 130, 44, "embedding", [], "teal")
    pg = b.cylinder(1010, 505, 110, 96, "Postgres", [], "purple")
    b.arrow(qq.right(), ee.left())
    b.arrow(ee.right(), pg.left(), label="search\ncosine similarity", color="purple", label_dy=-28)
    return b


# ------------------------------------------------------------------ page 6


@board
def ehr_presidio():
    b = Board(1240, 880, "Redaction with Microsoft Presidio",
              "Page 6, top: where redaction sits, what the script needs, and its two engines")

    b.group(20, 90, 1200, 220, "Where redaction sits", "blue")
    hp = b.card(40, 135, 110, 46, "H.P", [], "blue")
    pid = b.card(50, 225, 90, 36, "id", [], "blue", title_size=13)
    q = b.card(200, 135, 110, 46, "query", [], "orange")
    s = b.card(360, 135, 120, 46, "search", [], "purple")
    db = b.cylinder(530, 118, 170, 84, "Database", [], "purple")
    users = b.card(540, 230, 150, 46, "users", ["their rows"], "purple")
    txt = b.card(760, 230, 100, 46, "text", [], "grey")
    red = b.card(910, 230, 140, 46, "redaction", [], "orange")
    st = b.card(1090, 230, 115, 46, "Streamlit", [], "pink", title_size=13)
    b.arrow(hp.bottom(), pid.top())
    b.arrow(hp.right(), q.left())
    b.arrow(q.right(), s.left())
    b.arrow(s.right(), db.left())
    b.arrow(db.bottom(), users.top())
    b.arrow(users.right(), txt.left())
    b.arrow(txt.right(), red.left())
    b.arrow(red.right(), st.left())

    b.group(20, 340, 760, 240, "The redaction script", "purple")
    sc = b.card(40, 405, 100, 44, "script", [], "purple")
    sp = b.card(185, 380, 130, 40, "spaCy", [], "purple")
    pr = b.card(185, 432, 190, 40, "Microsoft Presidio", [], "orange", title_size=13)
    eng = b.card(420, 380, 140, 40, "eng-lang", [], "purple")
    b.arrow(sc.right(0.4), sp.left())
    b.arrow(sc.right(0.6), pr.left())
    b.arrow(sp.right(), eng.left(), label="loads")
    b.arrow(pr.right(), eng.bottom(0.35), label="uses", label_dx=24)
    tip = brace(b, 578, 380, 472)
    tm = b.card(630, 406, 130, 44, "time", [], "red")
    b.arrow(tip, tm.left(), color="red")

    sc2 = b.card(40, 500, 100, 44, "script", [], "purple")
    ssn = b.card(185, 488, 190, 34, "SSN: failing", [], "red", title_size=13)
    rgx = b.card(185, 530, 190, 34, "regex: XX", [], "green", title_size=13)
    b.arrow(sc2.right(0.4), ssn.left())
    b.arrow(sc2.right(0.6), rgx.left())

    b.group(810, 340, 410, 240, "ID formats differ by region", "orange")
    region = []
    for i, (code, val) in enumerate([("IND", "Aadhaar"), ("US", "SSN"), ("CA", "-"), ("EU", "-")]):
        y = 385 + i * 46
        c = b.card(840, y, 80, 34, code, [], "orange", title_size=13)
        v = b.card(970, y, 150, 34, val, [], "orange", title_size=13)
        b.arrow(c.right(), v.left())
        region.append(c)
    b.arrow(rgx.right(), region[1].left(), label="own regex", color="green", label_at=0.4)

    b.group(20, 610, 1200, 245, "Presidio: two engines", "orange")
    tx = b.card(40, 668, 100, 44, "text", [], "grey")
    code = b.card(200, 652, 400, 76, "", ["self.analyzer = AnalyzerEngine()",
                                           "self.anonymizer = AnonymizerEngine()"], "dark", size=13,
                  align="left")
    b.arrow(tx.right(), code.left(0.35))
    ents = []
    for i, (name, removed) in enumerate([("person", True), ("organisation", False), ("num", False),
                                         ("email", True)]):
        box = b.card(720, 640 + i * 42, 180, 32, name, [], "red" if removed else "grey",
                     title_size=13)
        if removed:
            strike(b, box)
        b.arrow(code.right(0.3), box.left(), color="grey", width=1.4)
        ents.append(box)
    b.arrow(ents[-1].bottom(0.5), code.bottom(0.72),
            via=[(ents[-1].cx, 830), (code.x + code.w * 0.72, 830)], label="redact these types",
            color="orange")
    out = b.card(200, 770, 220, 56, "person · email", ["replaced in the text"], "red")
    b.arrow(code.bottom(0.2), (code.x + code.w * 0.2, out.y))
    b.text(930, 700, "red strike = the types", 12, "red", "700", anchor="start")
    b.text(930, 718, "the anonymizer removes", 12, "red", "700", anchor="start")
    return b


@board
def ehr_scores():
    b = Board(1240, 780, "One score per piece of text",
              "Page 6, bottom: how the analyzer decides what gets redacted (cut-off from page 7)")

    b.group(20, 90, 330, 280, "Each piece gets a slot", "grey")
    inp = b.card(40, 205, 80, 44, "input", [], "grey")
    for i, k in enumerate(["X", "Y", "Z"]):
        y = 150 + i * 60
        b.text(170, y + 20, k, 16, INK, "700")
        b.arrow(inp.right(), (158, y + 14), color="grey", width=1.3)
        b.arrow((184, y + 14), (215, y + 14), color="grey", width=1.3)
        frame(b, 218, y, 110, 28, "grey")

    b.group(370, 90, 850, 280, "Worked example", "orange")
    toks = [("Hello", 0.1, None), ("How", 0.2, None), ("are", 0.1, None), ("you", 0.2, None),
            ("SSN: 111-22-3333", 0.9, "US_SSN"), ("my", 0.1, None), ("name", 0.1, None),
            ("is", 0.1, None), ("John", 0.7, "PERSON")]
    x = 395
    for word, score, ent in toks:
        w = max(len(word) * 8.4 + 16, 46)
        cx = x + w / 2
        b.text(cx, 150, word, 14, INK, "700")
        if ent:
            box = b.card(x - 4, 205, w + 8, 40, ent, [], "red", title_size=12)
        else:
            box = b.card(cx - 17, 208, 34, 34, "", [], "grey")
        b.arrow((cx, 160), box.top(), color="grey", width=1.3)
        b.arrow(box.bottom(), (cx, 300), color="grey", width=1.3)
        b.text(cx, 322, f"{score:g}", 16, "red" if score > 0.45 else "grey", "700")
        x += w + 22

    b.group(20, 400, 1200, 360, "Score against the cut-off: 0.45 > redaction", "purple")
    bx, bw = 330, 640
    for i, (word, score, ent) in enumerate(toks):
        y = 450 + i * 32
        hit = score > 0.45
        b.text(310, y + 12, word, 13, INK, "700" if hit else "400", anchor="end")
        b.bar(bx, y, bw, score, None, "red" if hit else "grey", h=14)
        b.text(bx + bw + 16, y + 12, f"{score:g}  " + ("redacted" if hit else "kept"), 13,
               "red" if hit else "grey", "700", anchor="start")
    tx = bx + bw * 0.45
    line(b, [(tx, 438), (tx, 742)], INK, 2, dashed=True)
    b.text(tx, 434, "0.45", 13, INK, "700")
    return b


# ------------------------------------------------------------------ page 7


@board
def ehr_guardrails():
    b = Board(1000, 500, "Next phase: guardrails",
              "Page 7, top: a Colang flow, and the LLM that decides accept or reject")

    b.group(20, 90, 960, 130, "Guardrails", "teal")
    g = b.card(40, 140, 150, 46, "guardrails", [], "teal")
    co = b.card(240, 140, 90, 46, ".co", [], "teal")
    cr = b.card(380, 140, 110, 46, "create", [], "teal")
    fl = b.card(540, 140, 100, 46, "flow", [], "teal")
    for a, c in [(g, co), (co, cr), (cr, fl)]:
        b.arrow(a.right(), c.left())
    b.text(670, 170, "rails.co holds the flows", 13, FAINT, anchor="start")

    b.group(20, 250, 960, 225, "The LLM behind the guardrail", "purple")
    llm = b.card(40, 300, 100, 50, "LLM", [], "dark")
    rj = b.card(220, 290, 130, 36, "reject", [], "red", title_size=13)
    ac = b.card(220, 338, 130, 36, "accept", [], "green", title_size=13)
    ne = b.card(400, 338, 150, 36, "no error", [], "green", title_size=13)
    b.arrow(llm.right(0.35), rj.left())
    b.arrow(llm.right(0.65), ac.left())
    b.arrow(ac.right(), ne.left())
    api = b.card(40, 410, 100, 40, "API", [], "purple")
    ds = b.card(200, 410, 200, 40, "DeepSeek API", [], "purple")
    cp = b.card(450, 410, 230, 40, "cheap, powerful", [], "green")
    b.arrow(llm.bottom(), api.top())
    b.arrow(api.right(), ds.left())
    b.arrow(ds.right(), cp.left())
    return b


@board
def ehr_request_apis():
    b = Board(1240, 860, "One request, then the three APIs",
              "Page 7, bottom: the path a question takes, and what api.py exposes")

    b.group(20, 90, 1200, 390, "One request, end to end", "blue")
    b.person(70, 136, "blue", 0.9, "User")
    q = b.card(130, 145, 110, 46, "query", [], "orange")
    e = b.card(290, 145, 130, 46, "embedding", [], "teal")
    db = b.cylinder(520, 118, 190, 100, "EC2 database", [], "purple")
    rec = b.card(800, 145, 140, 46, "records", [], "purple")
    red = b.card(800, 230, 140, 44, "redaction", [], "orange")
    gr = b.card(800, 310, 140, 44, "guardrails", [], "teal")
    d = b.card(800, 390, 140, 44, "data", [], "green")
    b.arrow((96, q.cy), q.left())
    b.arrow(q.right(), e.left())
    b.arrow(e.right(), db.left(), label="cosine", color="purple")
    b.arrow(db.right(), rec.left())
    for a, c in [(rec, red), (red, gr), (gr, d)]:
        b.arrow(a.bottom(), c.top())
    b.arrow(q.bottom(0.6), d.left(0.35), dashed=True, color="orange", label="the question", label_at=0.4)
    llm = b.card(430, 385, 110, 50, "LLM", [], "dark")
    resp = b.card(210, 390, 140, 40, "respond", [], "green", title_size=13)
    b.arrow(d.left(0.7), llm.right(0.6), label="clean data + question")
    b.arrow(llm.left(), resp.right())

    b.group(20, 510, 1200, 320, "api.py: three APIs", "green")
    api = b.card(40, 645, 120, 50, "api.py", [], "dark")
    a1 = b.card(220, 548, 120, 58, "api-1", ["(trial api)"], "green")
    t1 = b.card(390, 555, 170, 44, "text (query)", [], "grey")
    r1 = b.card(610, 555, 130, 44, "response", [], "grey")
    b.arrow(a1.right(), t1.left())
    b.arrow(t1.right(), r1.left())

    a2 = b.card(220, 655, 120, 44, "api-2", [], "green")
    prev = a2
    for name, w in [("chat", 90), ("query", 90), ("db", 70), ("get", 70), ("etc.", 90)]:
        box = b.card(prev.x + prev.w + 40, 655, w, 44, name, [], "grey")
        b.arrow(prev.right(), box.left())
        prev = box

    a3 = b.card(220, 755, 120, 44, "api-3", [], "green")
    pt = b.card(390, 755, 130, 44, "patients", [], "grey")
    ga = b.card(570, 755, 240, 44, "get all patient-id", [], "grey", title_size=13)
    b.arrow(a3.right(), pt.left())
    b.arrow(pt.right(), ga.left())

    for a in (a1, a2, a3):
        b.arrow(api.right(), a.left(), color="grey", width=1.4)
    b.pill(1010, 566, "/api/v1/clinical-query", "grey", size=11)
    b.pill(1010, 666, "/api/v1/chat", "grey", size=11)
    b.pill(1010, 766, "/api/v1/patients", "grey", size=11)
    b.text(1010, 548, "in main.py:", 12, FAINT, anchor="start")
    return b


# ------------------------------------------------------------------ pages 8-9


@board
def ehr_ui_prompt():
    b = Board(1160, 700, "The Streamlit page and the prompt",
              "Page 8, top: what the UI shows, and what it sends to the guardrail")

    b.group(20, 90, 760, 320, "The Streamlit page", "pink")
    frame(b, 150, 140, 410, 240, "pink")
    table_icon(b, 170, 160, 70, 80, "pink")
    b.text(205, 258, "patients", 11, "pink", "700")
    pop = b.pill(40, 186, "populate", "pink", size=12)
    b.arrow(pop.right(), (170, pop.cy), color="pink")
    frame(b, 290, 330, 250, 30, "grey", "#f8f9fa", 1.4, 6)
    mono(b, 302, 350, "Ask about patient ...", 11, FAINT)
    b.card(600, 170, 160, 36, "user: msg", [], "blue", title_size=13)
    b.card(600, 222, 160, 36, "bot: msg", [], "green", title_size=13)
    b.text(680, 290, "the chat, turn by turn", 11, FAINT)

    b.group(800, 90, 340, 320, "Building the prompt", "teal")
    usr = b.card(860, 135, 120, 40, "user", [], "blue")
    rd = b.card(1015, 185, 110, 36, "redaction", [], "orange", title_size=13)
    pr = b.card(815, 240, 310, 76, "prompt =", ["{clinical_context{db}:", " user: ( ) }"], "dark",
                align="left")
    gr = b.card(860, 350, 120, 40, "guardrail", [], "teal")
    b.arrow(usr.bottom(), (usr.cx, pr.y))
    b.arrow(rd.left(), (usr.cx + 2, rd.cy), color="orange")
    b.arrow(pr.bottom(0.29), gr.top())

    b.group(20, 440, 1120, 240, "messages[-1]", "grey")
    frame(b, 60, 480, 400, 180, "grey")
    bubbles = [(80, 496, 170, "blue"), (270, 530, 170, "green"), (80, 564, 150, "blue"),
               (270, 598, 170, "green"), (80, 626, 190, "orange")]
    for i, (x, y, w, c) in enumerate(bubbles):
        frame(b, x, y, w, 22, c, PALETTE[c]["fill"], 1.4, 8)
    b.arrow((360, 670), (272, 640), color="orange", label="-1", label_dx=14)
    b.card(520, 505, 580, None, "messages[-1] is the newest question",
           ["Everything before it is earlier turns. Only the newest one",
            "is joined to the clinical context from the DB."], "orange", size=12)
    return b


@board
def ehr_memory():
    b = Board(1160, 790, "No agent: a workflow with a message list",
              "Page 8, bottom, and page 9: why there's no ReAct agent, and what memory is instead")

    b.group(20, 90, 1120, 260, "Not needed here: a LangGraph agent", "red")
    lg = b.card(40, 140, 130, 44, "langgraph", [], "grey")
    ag = b.card(220, 140, 100, 44, "Agent", [], "grey")
    ts = b.card(370, 140, 150, 44, "task simple", [], "yellow")
    ra = b.card(570, 140, 160, 44, "react agent", [], "red")
    for a, c in [(lg, ag), (ag, ts), (ts, ra)]:
        b.arrow(a.right(), c.left())
    r1 = b.card(560, 230, 110, 40, "reason", [], "red", title_size=13)
    ac = b.card(720, 230, 110, 40, "action", [], "red", title_size=13)
    r2 = b.card(880, 230, 110, 40, "reason", [], "red", title_size=13)
    tl = b.card(720, 295, 110, 36, "tools", [], "grey", title_size=13)
    b.arrow(ra.bottom(0.3), r1.top())
    b.arrow(r1.right(), ac.left())
    b.arrow(ac.right(), r2.left())
    b.arrow(ac.bottom(), tl.top())

    b.group(20, 380, 1120, 200, "Used: a plain workflow", "green")
    wf = b.card(40, 425, 480, 110, "workflow = [", ["{ role: user,  que:    [ ] },",
                                                   "{ role: agent, respon: [ ] }  ]"],
                "green", size=13, align="left")
    hist = b.card(600, 455, 240, 50, "conversation history", [], "green")
    llm = b.card(900, 455, 210, 50, "LLM API", ["on every call"], "dark")
    b.arrow(wf.right(), hist.left())
    b.arrow(hist.right(), llm.left())

    b.group(20, 610, 1120, 160, "The list grows one message at a time", "blue")
    b.card(40, 655, 430, 76, "chat = [ ]", ["→ { role: user, content: Hello }"], "blue", size=13,
           align="left")
    b.card(510, 670, 160, 40, "user: Hello", [], "blue", title_size=13)
    frame(b, 720, 670, 40, 40, "blue", PALETTE["blue"]["fill"], 1.6, 6)
    b.arrow((770, 690), (830, 690), color="blue")
    for i in range(4):
        frame(b, 845 + (i % 2) * 50, 650 + (i // 2) * 44, 40, 36, "blue", PALETTE["blue"]["fill"], 1.6, 6)
    b.text(1030, 695, "turn by turn", 12, FAINT, anchor="start")
    return b


# ------------------------------------------------------------------ Bappy


@board
def ehr_docker_plan():
    b = Board(1100, 670, "Deployment plan",
              "Bappy's Excalidraw sketch: GitHub to Docker to a container on AWS EC2")

    b.group(20, 90, 520, 550, "The plan", "blue")
    gh = b.card(60, 130, 300, 60, "project · GitHub", [], "blue")
    dz = b.card(90, 245, 240, 50, "Dockerize", [], "blue")
    im = b.card(90, 345, 240, 50, "Docker images", [], "blue")
    run = b.card(60, 445, 300, 50, "run image as container", [], "blue", title_size=13)
    ci = b.card(60, 550, 300, 56, "CI / CD", ["named, not built"], "grey", dashed=True)
    for a, c in [(gh, dz), (dz, im), (im, run)]:
        b.arrow(a.bottom(), c.top())

    b.group(570, 90, 510, 250, "AWS", "orange")
    aws = b.card(600, 125, 140, 46, "AWS", [], "orange")
    ec2 = b.card(700, 200, 150, 40, "EC2", [], "orange")
    dk = b.card(700, 265, 150, 40, "Docker", [], "orange")
    tx = aws.x + 40
    line(b, [(tx, aws.y + aws.h), (tx, dk.cy)], "orange", 2)
    b.arrow((tx, ec2.cy), ec2.left(), color="orange")
    b.arrow((tx, dk.cy), dk.left(), color="orange")
    b.arrow((1000, ec2.cy), ec2.right(), color="orange", label="1 · launch")
    b.arrow((1000, dk.cy), dk.right(), color="orange", label="2 · install")
    b.arrow(gh.right(), aws.left(), label="clone onto EC2")

    b.group(570, 370, 510, 270, "Why Docker: same box on any OS", "purple")
    osb = b.card(600, 420, 110, 44, "OS", ["any"], "grey")
    ellipse(b, 850, 540, 190, 70, "purple")
    b.text(850, 500, "container", 13, "purple", "700")
    ellipse(b, 790, 548, 52, 22, "purple", fill=False)
    b.arrow((842, 548), (925, 548), color="purple")
    b.text(955, 553, "App", 16, "purple", "700")
    b.arrow(osb.bottom(0.7), (700, 520), color="grey")
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
