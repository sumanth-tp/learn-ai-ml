import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "ir-enrich"
BOARDS = {}


def board(name):
    def deco(fn):
        BOARDS[name] = fn
        return fn
    return deco


def label(b, x, y, text, size=13, fill=INK, anchor="start", weight="400"):
    b.parts.append(
        f'<text xml:space="preserve" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MONO}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(text)}</text>'
    )


def hbar(b, x, y, width, value, maximum, color, text, name="", name_w=0, h=20):
    c = PALETTE[color]
    if name:
        label(b, x, y + h - 5, name, 13, INK)
    bx = x + name_w
    b.parts.append(f'<rect x="{bx}" y="{y}" width="{width}" height="{h}" rx="5" fill="#e9ecef"/>')
    b.parts.append(
        f'<rect x="{bx}" y="{y}" width="{max(1.5, width * value / maximum):.1f}" height="{h}" rx="5" fill="{c["stroke"]}"/>'
    )
    label(b, bx + width + 10, y + h - 5, text, 13, c["text"], weight="700")


@board("ir1-keyword-vs-ranked")
def keyword_vs_ranked():
    b = Board(1120, 520, "Strict keyword matching versus ranking", "300 SciFact claims over 5,183 abstracts; every figure is a mean over the queries")
    rows = [
        ("AND of all terms", 0.027, 0.028, "290 of 300 queries return nothing", "red"),
        ("at least half the terms", 0.179, 0.634, "median 6 documents, 26 empty", "orange"),
        ("OR of any term", 0.001, 0.978, "median 1,754 documents returned", "yellow"),
        ("BM25 top 1", 0.547, 0.526, "one document shown", "green"),
        ("BM25 top 10", 0.088, 0.794, "about 1.1 relevant per query", "teal"),
    ]
    label(b, 40, 112, "method", 13, FAINT, weight="700")
    label(b, 290, 112, "precision", 13, FAINT, weight="700")
    label(b, 560, 112, "recall", 13, FAINT, weight="700")
    for i, (name, p, r, note, colour) in enumerate(rows):
        y = 135 + i * 72
        label(b, 40, y + 16, name, 14, INK, weight="700")
        label(b, 40, y + 40, note, 11, FAINT)
        hbar(b, 290, y + 2, 200, p, 1, colour, f"{p:.3f}")
        hbar(b, 560, y + 2, 200, r, 1, colour, f"{r:.3f}")
    b.card(870, 150, 220, 250, "Read it as", ["AND is exact but starves", "OR finds almost all", "relevant documents", "and buries them", "ranking keeps both"], "yellow", size=12)
    return b


@board("ir1-skip-pointers")
def skip_pointers():
    b = Board(1120, 500, "Skip pointers pay off only on unequal lists", "Mean comparisons per AND of two real postings lists (12,015 term pairs, 5,183 documents)")
    groups = [
        ("lists of similar length (ratio 1 to 4, 3,876 pairs)", [("plain merge", 1532.4, "grey"), ("skips every sqrt(L)", 1558.8, "orange"), ("skips every 16", 1582.5, "purple")]),
        ("middle (ratio 4 to 30, 4,544 pairs)", [("plain merge", 2122.5, "grey"), ("skips every sqrt(L)", 2089.9, "orange"), ("skips every 16", 1726.8, "purple")]),
        ("one list 30 times longer or more (3,595 pairs)", [("plain merge", 2710.8, "grey"), ("skips every sqrt(L)", 1228.6, "orange"), ("skips every 16", 735.3, "purple")]),
    ]
    y = 105
    for title, bars in groups:
        label(b, 40, y + 14, title, 14, INK, weight="700")
        for j, (name, value, colour) in enumerate(bars):
            hbar(b, 40, y + 28 + j * 28, 560, value, 2800, colour, f"{value:,.1f}", name, 200, h=20)
        y += 130
    b.card(930, 150, 170, 220, "Reading it", ["similar lengths:", "skips cost extra", "very uneven:", "sqrt(L) halves it", "span 16 does better"], "yellow", size=11)
    return b


@board("ir1-tolerant-search")
def tolerant_search():
    b = Board(1120, 500, "A spelling fix is two jobs: propose, then verify", "797 misspelt queries made from a 30,058-term SciFact vocabulary")
    b.group(30, 100, 500, 330, "Which distance, which typo", "blue")
    kinds = [("deletion", 0.90, 0.90, 0.90), ("insertion", 1.00, 1.00, 1.00), ("substitution", 0.98, 0.98, 0.98), ("adjacent swap", 0.00, 0.78, 0.99)]
    label(b, 220, 160, "Lev max 1", 11, FAINT, "middle", "700")
    label(b, 330, 160, "Lev max 2", 11, FAINT, "middle", "700")
    label(b, 440, 160, "Damerau 1", 11, FAINT, "middle", "700")
    for i, (name, a, c, d) in enumerate(kinds):
        y = 185 + i * 58
        label(b, 50, y + 17, name, 13, INK, weight="700")
        for x, v in ((220, a), (330, c), (440, d)):
            colour = "red" if v < 0.5 else ("orange" if v < 0.95 else "green")
            b.card(x - 38, y, 76, 36, f"{v:.2f}", [], colour, size=14)
    b.group(570, 100, 520, 330, "Candidate filter before the distance check", "purple")
    rows = [("all terms", 30058, 10.65, "grey"), ("k-gram J >= 0.2", 347, 2.04, "orange"), ("k-gram J >= 0.3", 42, 1.93, "teal"), ("k-gram J >= 0.4", 9, 1.85, "green")]
    label(b, 590, 160, "filter", 11, FAINT, weight="700")
    label(b, 790, 160, "candidates", 11, FAINT, weight="700")
    label(b, 960, 160, "ms per query", 11, FAINT, weight="700")
    for i, (name, count, ms, colour) in enumerate(rows):
        y = 185 + i * 58
        b.card(585, y, 190, 36, name, [], colour, size=12)
        label(b, 800, y + 24, f"{count:,}", 15, INK, weight="700")
        label(b, 975, y + 24, f"{ms:.2f}", 15, INK, weight="700")
    label(b, 580, 420, "recall of the right term stayed 1.00 at every threshold", 12, FAINT)
    b.card(150, 445, 820, 40, "431 of 1,000 valid terms have another dictionary term one edit away", [], "yellow", size=13)
    return b


@board("ir1-compression")
def compression():
    b = Board(1120, 520, "Gap coding wins; the best code depends on list length", "633,514 postings in 35,734 lists built from SciFact")
    label(b, 40, 112, "Whole index, share of raw 32-bit identifiers", 14, INK, weight="700")
    rows = [("raw 32-bit ids", 100.0, "grey"), ("gaps + variable byte", 29.8, "orange"), ("gaps + gamma", 27.7, "green"), ("gaps + zlib 6", 52.5, "purple")]
    for i, (name, value, colour) in enumerate(rows):
        hbar(b, 40, 135 + i * 34, 380, value, 100, colour, f"{value:.1f}%", name, 190, h=22)
    label(b, 40, 300, "Bytes per posting by list length (document frequency)", 14, INK, weight="700")
    label(b, 280, 325, "variable byte", 12, PALETTE["orange"]["text"], "middle", "700")
    label(b, 420, 325, "gamma", 12, PALETTE["green"]["text"], "middle", "700")
    for i, (name, vb, gm) in enumerate([("1 doc (17,001 lists)", 1.97, 2.74), ("2 to 9 (12,307)", 1.88, 2.30), ("10 to 99 (5,272)", 1.38, 1.56), ("100 or more (1,154)", 1.01, 0.69)]):
        y = 335 + i * 38
        label(b, 40, y + 22, name, 13, INK)
        b.card(235, y, 90, 30, f"{vb:.2f}", [], "green" if vb < gm else "orange", size=14)
        b.card(375, y, 90, 30, f"{gm:.2f}", [], "green" if gm < vb else "orange", size=14)
    b.card(710, 140, 390, 130, "Decoding all lists", ["numpy variable byte: slower than", "plain Python gamma, about 1.7 to 1.9 times", "(per-list overhead on tiny lists)"], "teal", size=12)
    b.card(710, 300, 390, 150, "Heaps' law on this corpus", ["1,167,322 tokens, 35,734 terms", "fitted M = 30.2 x T^0.506", "k = 44, b = 0.49 predicts 41,341"], "yellow", size=12)
    return b


@board("ir1-weighting")
def weighting():
    b = Board(1120, 520, "Which weighting matters more", "nDCG@10 on 300 SciFact claims; tf-idf variants span 0.114, the BM25 grid only 0.035")
    rows = [("tf-idf, no length normalisation", 0.518, "red"), ("tf-idf cosine, no idf, sublinear", 0.571, "orange"), ("tf-idf cosine", 0.580, "orange"),
            ("tf-idf cosine, sublinear tf", 0.632, "teal"), ("BM25 worst grid cell (k1 0.2, b 0)", 0.638, "green"), ("BM25 k1 1.2, b 0.75", 0.669, "green"), ("BM25 best grid cell (k1 1.2, b 1)", 0.673, "green")]
    for i, (name, value, colour) in enumerate(rows):
        hbar(b, 40, 110 + i * 40, 420, value, 0.7, colour, f"{value:.3f}", name, 330, h=24)
    b.card(40, 410, 520, 90, "Do this first", ["use length-normalised, sublinear tf, with idf", "then tune: the whole BM25 grid spans only 0.035 nDCG@10"], "yellow", size=12)
    b.card(600, 410, 480, 90, "Caution", ["the best cell was picked on the same 300 queries", "so it is optimistic; quote the spread"], "red", size=12)
    return b


@board("ir1-feature-selection")
def feature_selection():
    b = Board(1120, 520, "Fewer features never beat all of them here", "Six newsgroups, 3,494 training and 2,326 test posts, 19,812 features")
    b.group(30, 100, 560, 400, "Supervised accuracy by chi-squared features", "blue")
    label(b, 330, 150, "Naive Bayes", 12, PALETTE["blue"]["text"], "middle", "700")
    label(b, 480, 150, "Rocchio", 12, PALETTE["orange"]["text"], "middle", "700")
    for i, (k, nb, ro) in enumerate([("20", 0.325, 0.374), ("100", 0.605, 0.620), ("500", 0.791, 0.760), ("2,000", 0.852, 0.810), ("19,812", 0.871, 0.829)]):
        y = 170 + i * 58
        label(b, 50, y + 24, f"{k} features", 14, INK, weight="700")
        b.card(280, y, 100, 38, f"{nb:.3f}", [], "blue", size=15)
        b.card(430, y, 100, 38, f"{ro:.3f}", [], "orange", size=15)
    b.group(630, 100, 460, 400, "k-means, adjusted Rand index", "purple")
    for i, (name, value, colour) in enumerate([("raw tf-idf", 0.166, "red"), ("SVD to 100 dims", 0.330, "orange"), ("SVD to 20 dims", 0.455, "green")]):
        hbar(b, 650, 190 + i * 70, 250, value, 0.5, colour, f"{value:.3f}", name, 0, h=24)
        label(b, 650, 183 + i * 70, name, 13, INK, weight="700")
    b.card(650, 410, 420, 70, "ten random starts on raw tf-idf: ARI from 0.000 to 0.235", [], "yellow", size=12)
    return b


@board("ir1-confidence")
def confidence():
    b = Board(1120, 520, "Overlapping intervals, separate verdicts", "nDCG@10 with 95% bootstrap intervals over 300 queries (10,000 resamples)")
    x0, x1 = 80, 640
    lo, hi = 0.56, 0.74

    def px(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    b.parts.append(f'<line x1="{x0}" y1="400" x2="{x1}" y2="400" stroke="{INK}" stroke-width="1.5"/>')
    for t in (0.56, 0.60, 0.64, 0.68, 0.72):
        b.parts.append(f'<line x1="{px(t):.1f}" y1="395" x2="{px(t):.1f}" y2="405" stroke="{INK}"/>')
        label(b, px(t), 424, f"{t:.2f}", 12, FAINT, "middle")
    systems = [("tf-idf cosine", 0.632, 0.589, 0.677, "orange"), ("BM25 b=0", 0.657, 0.609, 0.703, "teal"), ("BM25 b=0.75", 0.669, 0.624, 0.714, "green")]
    for i, (name, m, a, c, colour) in enumerate(systems):
        y = 160 + i * 70
        stroke = PALETTE[colour]["stroke"]
        label(b, 40, y - 18, name, 13, INK, weight="700")
        b.parts.append(f'<line x1="{px(a):.1f}" y1="{y}" x2="{px(c):.1f}" y2="{y}" stroke="{stroke}" stroke-width="5" stroke-linecap="round"/>')
        b.parts.append(f'<circle cx="{px(m):.1f}" cy="{y}" r="8" fill="{stroke}"/>')
        label(b, px(m), y + 28, f"{m:.3f}  [{a:.3f}, {c:.3f}]", 12, PALETTE[colour]["text"], "middle")
    b.card(700, 120, 380, 120, "Paired difference, BM25 b=0.75 minus tf-idf", ["+0.037, interval [+0.017, +0.056]", "Wilcoxon p = 0.0004", "98 of 300 queries changed"], "green", size=12)
    b.card(700, 260, 380, 110, "BM25 b=0.75 minus b=0", ["+0.013, interval [-0.000, +0.025]", "Wilcoxon p = 0.0311", "the two checks disagree: no claim"], "yellow", size=12)
    b.card(700, 390, 380, 90, "Same comparison, 50 random queries", ["+0.024, interval [-0.031, +0.079]"], "red", size=12)
    return b


def main(names):
    for name in names or list(BOARDS):
        print(BOARDS[name]().save(OUT / f"{name}.svg").relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
