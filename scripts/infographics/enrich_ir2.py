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


@board("ir2-index-size")
def index_size():
    b = Board(1120, 520, "The ratio survives, the web size does not", "Synthetic web of 200,000 pages; engine A holds 40%, engine B 25%; 400 repeats per row, 500 samples per engine")
    label(b, 40, 112, "estimated size of A over B (true value about 1.60)", 14, INK, weight="700")
    rows = [
        ("independent, fair sample", 1.610, "green", "1.610"),
        ("tilted to popular, fair sample", 1.610, "green", "1.610"),
        ("tilted to popular, popular-biased sample", 1.369, "red", "1.369"),
    ]
    for i, (name, value, colour, text) in enumerate(rows):
        y = 140 + i * 50
        label(b, 40, y + 15, name, 13, INK)
        hbar(b, 400, y, 300, value, 2.0, colour, text, h=22)
    label(b, 40, 322, "estimated size of the whole web over the true size (1.000 is perfect)", 14, INK, weight="700")
    rows2 = [
        ("independent, fair sample", 1.001, "green", "1.001"),
        ("tilted to popular, fair sample", 0.566, "red", "0.566"),
        ("tilted to popular, popular-biased sample", 0.486, "red", "0.486"),
    ]
    for i, (name, value, colour, text) in enumerate(rows2):
        y = 350 + i * 50
        label(b, 40, y + 15, name, 13, INK)
        hbar(b, 400, y, 300, value, 1.2, colour, text, h=22)
    b.card(820, 130, 270, 130, "Why", ["the shared count cancels", "in the ratio, so the", "ratio needs only fair", "samples"], "green", size=12)
    b.card(820, 330, 270, 150, "Why not", ["the web size also needs", "independent engines;", "both favour popular pages", "so the share found looks", "too high"], "red", size=12)
    return b


@board("ir2-crawl-and-ring")
def crawl_and_ring():
    b = Board(1120, 560, "Hosts set the crawl speed; virtual points even the ring", "12,000 simulated pages on 100 hosts; 100,000 documents on 10 machines")
    b.group(30, 95, 520, 440, "Pages fetched by second 100, 300, 600", "blue")
    rows = [
        ("fifo, 50 workers, 1s", "4,063", "6,167", "7,474", "orange"),
        ("priority, 50 workers, 1s", "4,062", "6,159", "7,470", "green"),
        ("priority, 500 workers, 1s", "4,158", "6,159", "7,470", "teal"),
        ("priority, 50 workers, 5s", "1,364", "3,691", "4,968", "red"),
    ]
    label(b, 50, 150, "setting", 12, FAINT, weight="700")
    for x, t in ((330, "t=100"), (400, "t=300"), (470, "t=600")):
        label(b, x, 150, t, 12, FAINT, weight="700")
    for i, (name, a, c, d, colour) in enumerate(rows):
        y = 170 + i * 44
        b.card(46, y, 480, 34, "", [], colour, size=12)
        label(b, 56, y + 22, name, 12, INK, weight="700")
        for x, t in ((330, a), (400, c), (470, d)):
            label(b, x, y + 22, t, 12, INK)
    label(b, 50, 380, "key pages found (of 120) by t=100: fifo 75, priority 101", 12, INK)
    label(b, 50, 405, "biggest host (2,885 pages): 200 fetched in 600 s, 120 at a 5 s delay", 12, INK, weight="700")
    b.card(50, 430, 480, 90, "Read it as", ["500 workers add 96 pages by t=100 and nothing later", "the delay and the biggest host decide the finish"], "yellow", size=12)
    b.group(570, 95, 520, 440, "Ring: busiest machine and keys moved", "purple")
    label(b, 590, 150, "points per machine", 12, FAINT, weight="700")
    label(b, 800, 150, "max/mean load", 12, FAINT, weight="700")
    label(b, 960, 150, "moved", 12, FAINT, weight="700")
    ring = [("1", 2.789, "0.114", "red"), ("10", 1.626, "0.093", "orange"), ("100", 1.102, "0.076", "green"), ("1000", 1.053, "0.094", "green")]
    for i, (n, v, moved, colour) in enumerate(ring):
        y = 170 + i * 46
        label(b, 590, y + 16, n, 13, INK, weight="700")
        hbar(b, 700, y, 180, v, 3.0, colour, f"{v:.3f}", h=22)
        label(b, 970, y + 16, moved, 13, INK)
    y = 170 + 4 * 46
    label(b, 590, y + 16, "modulo", 13, INK, weight="700")
    hbar(b, 700, y, 180, 1.020, 3.0, "grey", "1.020", h=22)
    label(b, 970, y + 16, "0.910", 13, PALETTE["red"]["text"], weight="700")
    b.card(590, 420, 480, 100, "Read it as", ["one point per machine leaves the busiest at 2.79x", "modulo is even but moves 91% of keys when a", "machine joins; the ideal is 9.1%"], "yellow", size=12)
    return b


@board("ir2-pagerank-spam")
def pagerank_spam():
    b = Board(1120, 540, "Damping changes the cost, link farms change the rank", "4,592 Wikipedia school articles, 119,882 links; target article starts at rank 2,297 of 4,592")
    b.group(30, 95, 480, 430, "Iterations to converge by damping", "blue")
    rows = [(0.5, 21, 0.986), (0.7, 32, 0.997), (0.85, 46, 1.000), (0.95, 62, 0.997), (0.99, 71, 0.993)]
    label(b, 50, 150, "d", 12, FAINT, weight="700")
    label(b, 330, 150, "Spearman vs 0.85", 12, FAINT, weight="700")
    for i, (d, it, rho) in enumerate(rows):
        y = 170 + i * 46
        label(b, 50, y + 16, f"{d}", 13, INK, weight="700")
        hbar(b, 100, y, 200, it, 80, "blue", str(it), h=22)
        label(b, 350, y + 16, f"{rho:.3f}", 13, INK)
    b.card(50, 420, 440, 90, "Read it as", ["iterations triple from d=0.5 to 0.99", "the order barely moves", "PageRank vs in-links: Spearman 0.966"], "yellow", size=12)
    b.group(530, 95, 560, 430, "Target rank as the link farm grows (1 is best)", "red")
    label(b, 550, 150, "farm pages", 12, FAINT, weight="700")
    label(b, 680, 150, "plain", 12, FAINT, weight="700")
    label(b, 820, 150, "trusted teleport", 12, FAINT, weight="700")
    label(b, 990, 150, "HITS", 12, FAINT, weight="700")
    data = [(0, 2297, 1980, 2404), (10, 442, 1679, 2404), (50, 20, 1296, 2402), (200, 1, 1057, 2384), (1000, 1, 950, 2295)]
    for i, (n, plain, trusted, hits) in enumerate(data):
        y = 170 + i * 46
        label(b, 560, y + 16, str(n), 13, INK, weight="700")
        colour = "red" if plain < 100 else "grey"
        b.card(650, y, 110, 32, f"{plain:,}", [], colour, size=13)
        b.card(800, y, 110, 32, f"{trusted:,}", [], "green", size=13)
        b.card(950, y, 110, 32, f"{hits:,}", [], "purple", size=13)
    b.card(550, 420, 520, 90, "Read it as", ["200 fake pages make an unremarkable page first", "the trusted seed set blunts it (1,057th), not removes it", "HITS on the whole graph barely notices"], "yellow", size=12)
    return b


@board("ir2-cross-language")
def cross_language():
    b = Board(1120, 560, "A shared space bridges languages it knows, and fails on one it does not", "Tatoeba sentence pairs; English query against a pool of 1,000 sentences (390 for Swahili)")
    label(b, 40, 105, "accuracy at 1", 13, FAINT, weight="700")
    label(b, 170, 105, "char 3-gram", 12, PALETTE["orange"]["text"], weight="700")
    label(b, 330, 105, "English-only", 12, PALETTE["red"]["text"], weight="700")
    label(b, 490, 105, "multilingual", 12, PALETTE["green"]["text"], weight="700")
    data = [("spa", 0.208, 0.147, 0.975), ("fra", 0.222, 0.188, 0.944), ("deu", 0.249, 0.167, 0.984), ("tur", 0.085, 0.044, 0.971), ("swh", 0.126, 0.100, 0.233), ("ara", 0.008, 0.011, 0.915), ("cmn", 0.021, 0.065, 0.964), ("hin", 0.003, 0.004, 0.981)]
    for i, (lang, a, c, d) in enumerate(data):
        y = 120 + i * 44
        label(b, 40, y + 20, lang, 14, INK, weight="700")
        for value, colour, off in ((a, "orange", 170), (c, "red", 330), (d, "green", 490)):
            hbar(b, off, y, 90, value, 1.0, colour, f"{value:.3f}", h=24)
    b.card(760, 120, 330, 170, "Swahili, 390 pairs", ["multilingual accuracy 0.233", "right translation: mean cosine 0.386", "best wrong sentence: 0.520", "not on the model card's list"], "red", size=12)
    b.card(760, 310, 330, 170, "Same table for German", ["multilingual accuracy 0.984", "right translation: mean cosine 0.906", "best wrong sentence: 0.569", "a cosine threshold only works", "inside a pair the model knows"], "green", size=12)
    return b


@board("ir2-clip")
def clip():
    b = Board(1120, 520, "Recall falls as the gallery grows; matching cosines are small", "Flickr30k 1,000-image test split, CLIP ViT-B/32, 5,000 captions")
    b.group(30, 95, 560, 410, "Text to image recall at 1", "blue")
    rows = [("100 images", 0.844, "green"), ("300 images", 0.721, "teal"), ("1,000 images", 0.588, "orange"), ("1,000, shuffled words", 0.423, "red")]
    for i, (name, v, colour) in enumerate(rows):
        y = 150 + i * 62
        label(b, 50, y + 16, name, 13, INK, weight="700")
        hbar(b, 260, y, 240, v, 1.0, colour, f"{v:.3f}", h=24)
    b.card(50, 410, 520, 80, "Read it as", ["same model, same captions: only the number of rivals changes", "shuffled captions keep about 72% of recall at 1"], "yellow", size=12)
    b.group(620, 95, 470, 410, "Average cosine, unit vectors", "purple")
    rows = [("caption with own image", 0.313, "red"), ("caption with other captions", 0.424, "orange"), ("image with other images", 0.487, "teal")]
    for i, (name, v, colour) in enumerate(rows):
        y = 150 + i * 62
        label(b, 640, y - 4, name, 13, INK, weight="700")
        hbar(b, 640, y + 4, 300, v, 1.0, colour, f"{v:.3f}", h=24)
    b.card(640, 340, 430, 150, "Modality gap", ["centres of images and texts are 0.802 apart", "the worked toy cosine of 0.816 is", "far above the real matching average", "a fixed threshold copied from a toy fails"], "purple", size=12)
    return b


@board("ir2-recsys")
def recsys():
    b = Board(1120, 540, "Short queries are weak, idf buys breadth, cold items need content", "MovieLens 100K, last 20% of each user held out; 907 users with a liked held-out film")
    b.group(30, 95, 600, 430, "Precision at 10 by query length", "blue")
    label(b, 50, 148, "recent films", 12, FAINT, weight="700")
    label(b, 330, 148, "plain", 12, FAINT, weight="700")
    label(b, 480, 148, "idf", 12, FAINT, weight="700")
    data = [("last 1", 0.089, 0.089), ("last 3", 0.122, 0.115), ("last 10", 0.129, 0.126), ("last 20", 0.131, 0.130), ("all", 0.126, 0.129)]
    for i, (name, a, c) in enumerate(data):
        y = 165 + i * 42
        label(b, 50, y + 17, name, 13, INK, weight="700")
        hbar(b, 140, y, 150, a, 0.15, "blue", f"{a:.3f}", h=22)
        hbar(b, 400, y, 100, c, 0.15, "orange", f"{c:.3f}", h=22)
    label(b, 50, 395, "popularity baseline 0.079", 13, PALETTE["red"]["text"], weight="700")
    b.card(50, 415, 560, 90, "Read it as", ["the last 20 films work best; idf ties on precision", "coverage with the whole history: 0.127 plain, 0.182 idf"], "yellow", size=12)
    b.group(660, 95, 430, 430, "Cold films, precision at 10", "green")
    rows = [("random order", 0.054, "grey"), ("genre content", 0.090, "green"), ("hindsight popularity", 0.196, "orange")]
    for i, (name, v, colour) in enumerate(rows):
        y = 170 + i * 70
        label(b, 680, y - 4, name, 13, INK, weight="700")
        hbar(b, 680, y + 4, 240, v, 0.25, colour, f"{v:.3f}", h=24)
    b.card(680, 400, 390, 100, "Read it as", ["93 cold films: collaborative filtering has", "no similarity for any of them; hindsight uses", "the test ratings, so it is only a ceiling"], "yellow", size=12)
    return b


@board("ir2-neural-overlap")
def neural_overlap():
    b = Board(1120, 540, "Dense wins on low word overlap, BM25 on high, hybrid between", "300 SciFact claims over 5,183 abstracts, nDCG@10, all-MiniLM-L6-v2 and rank-bm25")
    label(b, 40, 108, "claims grouped by how many claim words appear in the right abstract", 13, FAINT, weight="700")
    groups = [
        ("lowest third (overlap 0.00 to 0.38)", (0.284, 0.402, 0.360)),
        ("middle third (0.38 to 0.64)", (0.765, 0.697, 0.776)),
        ("highest third (0.67 to 1.00)", (0.949, 0.838, 0.913)),
    ]
    names = (("BM25", "teal"), ("dense", "purple"), ("hybrid RRF", "orange"))
    for g, (title, values) in enumerate(groups):
        y0 = 130 + g * 120
        label(b, 40, y0 + 14, title, 14, INK, weight="700")
        for j, ((name, colour), v) in enumerate(zip(names, values)):
            hbar(b, 40, y0 + 24 + j * 26, 340, v, 1.0, colour, f"{v:.3f}", name, 110, h=20)
    b.group(700, 120, 390, 170, "First k words of each claim", "blue")
    data = [("k=3", 0.360, 0.254, 0.358), ("k=5", 0.471, 0.363, 0.470), ("k=8", 0.579, 0.526, 0.587)]
    label(b, 720, 170, "BM25", 12, PALETTE["teal"]["text"], weight="700")
    label(b, 830, 170, "dense", 12, PALETTE["purple"]["text"], weight="700")
    label(b, 940, 170, "hybrid", 12, PALETTE["orange"]["text"], weight="700")
    for i, (k, a, c, d) in enumerate(data):
        y = 190 + i * 30
        label(b, 720, y + 14, k, 12, INK, weight="700")
        for x, v in ((775, a), (880, c), (985, d)):
            label(b, x, y + 14, f"{v:.3f}", 12, INK)
    b.card(700, 310, 390, 200, "Read it as", ["outer thirds: the hybrid never wins", "it wins overall (0.686) through the middle", "dense collapses faster on short claims", "(0.254 against 0.360 at 3 words)", "overlap uses the answer: analysis only"], "yellow", size=12)
    return b


def main(names):
    for name in names or list(BOARDS):
        print(BOARDS[name]().save(OUT / f"{name}.svg").relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
