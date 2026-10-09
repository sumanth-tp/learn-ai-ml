"""Boards for the three information retrieval industry projects.

Run from the repo root:

    python3 scripts/infographics/ir_projects.py            # all boards
    python3 scripts/infographics/ir_projects.py techdocs   # boards whose name contains the word
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import FAINT, INK, PALETTE, Board

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "ir-projects"
BOARDS = {}
NAMES = {}


def board(name):
    def deco(fn):
        BOARDS[fn.__name__] = fn
        NAMES[fn.__name__] = name
        return fn

    return deco


def line(b, x1, y1, x2, y2, color="#adb5bd", width=1.2, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    b.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{extra}/>')


def dot(b, x, y, color, r=7, solid=True):
    c = PALETTE[color]
    fill = c["stroke"] if solid else "#fffdf7"
    b.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{c["stroke"]}" stroke-width="2.2"/>')


def rect(b, x, y, w, h, color, opacity=0.9):
    c = PALETTE[color]
    b.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="{c["stroke"]}" fill-opacity="{opacity}"/>')


@board("techdocs-architecture")
def techdocs_architecture():
    b = Board(1240, 640, "Technical documentation search: two paths", "A build job writes a versioned index; a small service reads whichever version is current")
    b.group(20, 90, 560, 520, "Index build job (offline, repeatable)", "orange")
    snap = b.card(40, 135, 250, 80, "snapshot", ["walk installed packages", "6,514 pages, JSONL"], "orange", size=11)
    ana = b.card(310, 135, 250, 80, "code-aware analyser", ["split read_csv, np to numpy", "stop words, light stemming"], "orange", size=11)
    bm = b.card(40, 250, 250, 80, "BM25 fields", ["name field x2", "signature + body"], "purple", size=11)
    enc = b.card(310, 250, 250, 80, "dense encoder", ["MiniLM, 384 numbers", "name + summary text"], "purple", size=11)
    ver = b.card(40, 375, 520, 70, "index-<timestamp>-<hash>/", ["dense.npy   pages.jsonl.gz   manifest.json"], "yellow", size=11)
    gate = b.card(40, 480, 250, 90, "release gate", ["dev nDCG@10 must not", "fall by more than 0.03"], "red", size=11)
    ptr = b.card(310, 480, 250, 90, "CURRENT + HISTORY", ["atomic pointer swap", "rollback = repoint"], "green", size=11)
    b.arrow(snap.right(), ana.left())
    b.arrow(ana.bottom(), bm.top(0.8))
    b.arrow(snap.bottom(), bm.top(0.3))
    b.arrow(ana.bottom(), enc.top(0.5))
    b.arrow(bm.bottom(), ver.top(0.2))
    b.arrow(enc.bottom(), ver.top(0.8))
    b.arrow(ver.bottom(), gate.top(0.5))
    b.arrow(gate.right(), ptr.left(), label="pass")
    b.group(620, 90, 600, 520, "Query service (online)", "blue")
    q = b.card(640, 135, 170, 60, "GET /search", ["q, k"], "blue", size=11)
    lex = b.card(640, 235, 170, 70, "BM25 top 50", ["0.2 ms"], "teal", size=11)
    den = b.card(840, 235, 170, 70, "dense top 50", ["3.6 ms with encode"], "teal", size=11)
    fuse = b.card(740, 340, 220, 70, "min-max blend", ["lexical weight 0.2"], "yellow", size=11)
    rr = b.card(990, 340, 210, 70, "rerank (off)", ["20 ms to 50 ms, no gain"], "grey", size=11, dashed=True)
    hit = b.card(740, 450, 220, 60, "hits + confidence", ["abstain if cosine < 0.457"], "green", size=11)
    log = b.card(990, 450, 210, 60, "query log", ["JSONL, clicks, /metrics"], "purple", size=11)
    b.arrow(q.bottom(0.3), lex.top(0.5))
    b.arrow(q.bottom(0.7), den.top(0.5))
    b.arrow(lex.bottom(), fuse.top(0.2))
    b.arrow(den.bottom(), fuse.top(0.8))
    b.arrow(fuse.right(), rr.left(), dashed=True)
    b.arrow(fuse.bottom(), hit.top())
    b.arrow(hit.right(), log.left())
    b.arrow(ptr.right(), (740, 540), via=[(600, 540)], color="green", dashed=True, label="load")
    b.text(930, 560, "the service loads the index that CURRENT names", 11, FAINT)
    b.text(930, 578, "POST /admin/reload swaps it, no restart", 11, FAINT)
    return b


@board("techdocs-fusion-example")
def techdocs_fusion_example():
    b = Board(1200, 560, "Worked example: blending two ranked lists", "Three pages, one query; numbers are the ones the code prints")
    lex = b.card(40, 110, 300, 190, "BM25 scores", ["page 1: 9.0", "page 2: 3.0", "page 3: not retrieved"], "teal", size=14)
    den = b.card(40, 330, 300, 190, "dense cosine", ["page 2: 0.8", "page 3: 0.5", "page 1: 0.2"], "purple", size=14)
    sc1 = b.card(400, 110, 300, 190, "scaled to 0..1", ["(x - min) / (max - min)", "page 1: 1.0   page 2: 0.0", "page 3: 0.0 (absent)"], "teal", size=13)
    sc2 = b.card(400, 330, 300, 190, "scaled to 0..1", ["(0.5 - 0.2) / 0.6 = 0.5", "page 2: 1.0   page 3: 0.5", "page 1: 0.0"], "purple", size=13)
    b.arrow(lex.right(), sc1.left())
    b.arrow(den.right(), sc2.left())
    res = b.card(790, 110, 370, 410, "blend, lexical weight 0.2", ["score = 0.2 x lexical + 0.8 x dense", "", "page 2: 0.2 x 0.0 + 0.8 x 1.0 = 0.80", "page 3: 0.2 x 0.0 + 0.8 x 0.5 = 0.40", "page 1: 0.2 x 1.0 + 0.8 x 0.0 = 0.20", "", "order: 2, 3, 1", "", "weight 0.5 would tie pages 1 and 2", "at 0.50 each: the weight is a decision,", "not a constant of nature"], "yellow", size=13, align="left")
    b.arrow(sc1.right(), res.left(0.3))
    b.arrow(sc2.right(), res.left(0.7))
    return b


@board("techdocs-frontier")
def techdocs_frontier():
    b = Board(1200, 620, "Quality against latency on 106 test queries", "Each point is one configuration; x is the p95 latency of a single query on this machine, log scale")
    left, right, top, bottom = 110, 1130, 110, 530
    lo, hi = 0.52, 0.74

    def px(ms):
        return left + (math.log10(ms) - math.log10(0.2)) / (math.log10(150) - math.log10(0.2)) * (right - left)

    def py(v):
        return bottom - (v - lo) / (hi - lo) * (bottom - top)

    line(b, left, bottom, right, bottom, INK, 1.6)
    line(b, left, top, left, bottom, INK, 1.6)
    for v in (0.55, 0.60, 0.65, 0.70):
        line(b, left, py(v), right, py(v), "#dee2e6")
        b.text(left - 10, py(v) + 4, f"{v:.2f}", 12, FAINT, anchor="end")
    for ms in (0.3, 1, 3, 10, 30, 100):
        line(b, px(ms), top, px(ms), bottom, "#f1f3f5")
        b.text(px(ms), bottom + 22, f"{ms:g} ms", 12, FAINT)
    b.text((left + right) / 2, bottom + 52, "p95 latency (log scale)", 13, INK, "700")
    b.text(40, (top + bottom) / 2, "nDCG@10", 13, INK, "700")
    points = [
        ("BM25", 0.561, 0.3, "teal", "below", 0),
        ("BM25 + rerank 20", 0.603, 70.1, "grey", "below", 0),
        ("dense", 0.657, 4.0, "purple", "below", 0),
        ("hybrid (shipped)", 0.705, 4.4, "green", "above", 0),
        ("hybrid + rerank 10", 0.707, 29.1, "orange", "below", 0),
        ("hybrid + rerank 20", 0.707, 56.9, "orange", "above", 0),
        ("hybrid + rerank 30", 0.709, 103.6, "orange", "below", 22),
        ("rerank 20 replaces blend", 0.634, 48.0, "red", "below", 0),
    ]
    for name, v, ms, color, side, extra in points:
        x, y = px(ms), py(v)
        dot(b, x, y, color, 8)
        dy = -16 if side == "above" else 26 + extra
        anchor = "end" if ms > 60 else "middle"
        b.text(x + (-12 if anchor == "end" else 0), y + dy, f"{name}  {v:.3f}", 12, color, "700", anchor=anchor)
    b.text(620, 604, "The three hybrid + rerank points sit within 0.004 of the shipped hybrid; their confidence interval is wider than that.", 12, FAINT)
    return b


@board("techdocs-slices")
def techdocs_slices():
    b = Board(1200, 560, "Error analysis by query slice", "nDCG@10 on the 106 test queries; the best method differs by kind of query")
    slices = [("identifier", 20, 0.930, 0.749, 0.813), ("keyword", 17, 0.599, 0.778, 0.794), ("howto", 34, 0.385, 0.634, 0.677), ("behaviour", 22, 0.370, 0.544, 0.592), ("ambiguous", 13, 0.728, 0.609, 0.683)]
    left, bottom, top = 90, 470, 120
    width = 190
    line(b, left, bottom, left + 5 * width + 20, bottom, INK, 1.6)
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - v * (bottom - top)
        line(b, left, y, left + 5 * width + 20, y, "#dee2e6")
        b.text(left - 10, y + 4, f"{v:.2f}", 12, FAINT, anchor="end")
    for i, (name, n, bm, de, hy) in enumerate(slices):
        x0 = left + 20 + i * width
        for j, (val, color) in enumerate(((bm, "teal"), (de, "purple"), (hy, "green"))):
            h = val * (bottom - top)
            rect(b, x0 + j * 50, bottom - h, 42, h, color)
            b.text(x0 + j * 50 + 21, bottom - h - 6, f"{val:.2f}", 11, color, "700")
        b.text(x0 + 70, bottom + 22, name, 13, INK, "700")
        b.text(x0 + 70, bottom + 40, f"{n} queries", 11, FAINT)
    b.pill(330, 78, "BM25", "teal", 12, True)
    b.pill(410, 78, "dense", "purple", 12, True)
    b.pill(500, 78, "hybrid (shipped)", "green", 12, True)
    return b


@board("techdocs-release")
def techdocs_release():
    b = Board(1200, 470, "Refresh and rollback", "A new index is built beside the live one, judged, then published by moving one pointer")
    a = b.card(30, 150, 200, 110, "1 build", ["index-B written to a", ".building- folder, then", "renamed into place"], "orange", size=12)
    c = b.card(270, 150, 200, 110, "2 judge", ["dev nDCG@10 of B", "against A: 0.605 vs", "0.731 in the demo"], "yellow", size=12)
    d = b.card(510, 150, 200, 110, "3 publish", ["CURRENT := B", "HISTORY gets a line"], "green", size=12)
    e = b.card(750, 150, 200, 110, "4 reload", ["POST /admin/reload", "no restart needed"], "blue", size=12)
    f = b.card(990, 150, 180, 110, "5 watch", ["p95, abstain rate,", "click MRR"], "purple", size=12)
    for p, q in ((a, c), (c, d), (d, e), (e, f)):
        b.arrow(p.right(), q.left())
    x = b.card(270, 320, 200, 90, "gate fails", ["B is kept on disk,", "CURRENT stays A"], "red", size=12)
    r = b.card(750, 320, 420, 90, "rollback", ["techdocs rollback repoints CURRENT to the previous", "published index; reload applies it in seconds"], "red", size=12)
    b.arrow(c.bottom(), x.top(), label="drop > 0.03")
    b.arrow(f.bottom(0.5), r.top(0.85), label="metrics regress", color="red")
    return b


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = [name for name in BOARDS if not argv or any(a in NAMES[name] for a in argv)]
    for name in wanted:
        path = OUT / f"{NAMES[name]}.svg"
        BOARDS[name]().save(path)
        print("wrote", path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
