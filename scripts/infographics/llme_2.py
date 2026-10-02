"""Infographics for docs/llm-engineering/01-adapting-models, chapters 05 to 07.

Run from the repo root:

    python3 scripts/infographics/llme_2.py            # all boards
    python3 scripts/infographics/llme_2.py distill    # boards whose name contains the text
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "llme"
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


def line(b, x1, y1, x2, y2, stroke=INK, width=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    b.parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{width}"{d} stroke-linecap="round"/>'
    )


def rect(b, x, y, w, h, fill, stroke, width=1.6, rx=4):
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{max(h, 0):.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def dot(b, cx, cy, r, fill, stroke, width=2):
    b.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def bars(b, x, base, values, labels, color, max_h, scale, bar_w=30, gap=14, value_size=10, fmt="{:.3f}"):
    c = PALETTE[color]
    for i, v in enumerate(values):
        bx = x + i * (bar_w + gap)
        h = v * scale * max_h
        rect(b, bx, base - h, bar_w, h, c["fill"], c["stroke"], 1.6, 3)
        raw_text(b, bx + bar_w / 2, base - h - 5, fmt.format(v), value_size, c["text"], weight="700")
        raw_text(b, bx + bar_w / 2, base + 16, labels[i], 11, INK)


@board("knowledge-distillation-temperature")
def distill_temperature():
    b = Board(1240, 640, "Temperature turns an answer into a lesson", "A five-class teacher, then a 682-parameter student on 8x8 digits")
    b.group(20, 95, 700, 360, "Teacher softmax(z / T), logits 9.0, 5.5, 4.6, 1.0, -2.0", "blue")
    classes = ["car", "trk", "bus", "cat", "crt"]
    teacher = {
        1: [0.9589, 0.0290, 0.0118, 0.0003, 0.0000],
        2: [0.7651, 0.1330, 0.0848, 0.0140, 0.0031],
        4: [0.5131, 0.2139, 0.1708, 0.0694, 0.0328],
        8: [0.3517, 0.2271, 0.2029, 0.1294, 0.0889],
    }
    entropy = {1: "0.198", 2: "0.760", 4: "1.271", 8: "1.508"}
    for k, T in enumerate((1, 2, 4, 8)):
        gx = 38 + (k % 2) * 345
        gy = 150 + (k // 2) * 150
        raw_text(b, gx + 150, gy - 2, f"T = {T}   entropy {entropy[T]} nats", 12, PALETTE["blue"]["text"], weight="700")
        bars(b, gx, gy + 100, teacher[T], classes, "blue", 82, 1.0, bar_w=30, gap=18, value_size=10)
        line(b, gx - 6, gy + 100, gx + 5 * 30 + 4 * 18 + 6, gy + 100, FAINT, 1.2)

    b.group(740, 95, 480, 360, "Digits: 300 labelled rows, test 600", "orange")
    rows = [["model", "test accuracy"],
            ["teacher, 85,002 parameters", "0.9557"],
            ["student: hard labels only", "0.8787"],
            ["student: soft targets, T = 1", "0.8787"],
            ["student: soft targets, T = 4", "0.9213"],
            ["student: 0.3 hard + 0.7 soft, T = 4", "0.9233"],
            ["student: soft T = 4 + 897 extra rows", "0.9347"]]
    b.table(760, 140, [310, 130], rows, "orange", size=12, row_h=34)
    raw_text(b, 980, 405, "mean of 5 seeds; one seed ranges", 11, PALETTE["orange"]["text"])
    raw_text(b, 980, 422, "0.8517 to 0.9167 for hard labels", 11, PALETTE["orange"]["text"])

    b.card(20, 480, 590, 140, "Why multiply by T squared", ["soft-target gradient shrinks like 1 / T^2:", "norm x T^2 stays near 1 across temperatures", "T=1  0.7259    T=4  0.9683    T=20  0.8799"], "green", size=12)
    b.card(630, 480, 590, 140, "What T = 1 hides", ["KL(teacher || student) on the toy student:", "T=1  0.66750   T=4  0.10767   T=20  0.00468", "at T=1 truck and bus are 3% and 1%: almost no signal"], "red", size=12)
    return b


@board("knowledge-distillation-recipes")
def distill_recipes():
    b = Board(1240, 700, "Three ways to hand a student the teacher's knowledge", "SmolLM2-135M teacher, a 49.6M-parameter student that keeps 6 of its 30 layers")
    lanes = [
        ("word-level: teacher logits on fixed text", "blue",
         ["fixed corpus", "teacher logits at every token", "student matches the distribution"]),
        ("sequence-level: train on teacher text", "orange",
         ["prompts", "teacher decodes (greedy)", "student cross-entropy on that text"]),
        ("on-policy: teacher grades student samples", "green",
         ["prompts", "student samples its own text", "teacher logits on those tokens"]),
    ]
    for i, (title, color, steps) in enumerate(lanes):
        y = 100 + i * 120
        b.group(20, y, 760, 105, title, color)
        prev = None
        for j, s in enumerate(steps):
            c = b.card(40 + j * 245, y + 38, 215, 52, s, [], color, size=12)
            if prev:
                b.arrow(prev.right(), c.left())
            prev = c

    b.group(800, 100, 420, 345, "After 60 steps, same student start", "purple")
    rows = [["signal", "KL, teacher text", "KL, own samples"],
            ["start", "10.670", "7.589"],
            ["word-level", "4.662", "2.579"],
            ["sequence-level", "5.013", "3.131"],
            ["on-policy", "4.831", "2.168"]]
    b.table(815, 145, [124, 138, 138], rows, "purple", size=12, row_h=40)
    b.card(815, 365, 390, 66, "each wins on its own ground", ["on-policy is best on the student's own samples,", "word-level on the teacher's text"], "grey", size=11)

    b.card(20, 470, 590, 205, "Starting the student", ["copy teacher layers 0, 6, 12, 18, 24, 29", "held-out KL at step 0: 10.385", "after 60 steps: 4.721, top-1 agreement 0.120", "", "random weights, same shape:", "KL at step 0: 8.267, after 60 steps: 5.976"], "teal", size=12, align="left")
    b.card(630, 470, 590, 205, "Read the numbers as a toy", ["one run, 60 steps, batch 8, 16 new tokens per", "prompt, 64 held-out prompts. The ordering shows", "the mechanism, not a benchmark. Qwen3 reports", "on-policy distillation beat reinforcement learning", "on Qwen3-8B at about one tenth of the GPU hours."], "yellow", size=12, align="left")
    return b


@board("synthetic-data-generation-pipeline")
def synth_pipeline():
    b = Board(1240, 700, "A synthetic-data pipeline is a funnel", "Stub generator, no model calls: 600 candidate instructions from 12 seeds, filtered in order")
    steps = [
        ("generate", "600", "from a seed or an earlier\nrow: breadth, constraint,\ndeepen, paraphrase, copy,\nshort, refusal, multimodal", "blue"),
        ("rule filters", "-130", "5 to 40 words, no refusal\nor image words", "orange"),
        ("ROUGE-L dedup", "-281", "0.7 or more against the\n12 seeds and every kept row", "purple"),
        ("stub judge", "-88", "score under 7.0 dropped", "pink"),
        ("kept", "101", "16.8% of 600", "green"),
    ]
    prev = None
    for i, (title, count, body, color) in enumerate(steps):
        c = b.card(30 + i * 240, 100, 215, 150, title, [count, ""] + body.split("\n"), color, size=12, title_size=14)
        if prev:
            b.arrow(prev.right(), c.left())
        prev = c

    b.group(20, 280, 600, 395, "Where each recipe ends up", "teal")
    rows = [["recipe", "made", "rules", "dup", "judge", "kept"],
            ["breadth", "232", "12", "121", "72", "27"],
            ["constraint", "73", "1", "48", "0", "24"],
            ["deepen", "94", "7", "38", "0", "49"],
            ["paraphrase", "51", "0", "40", "10", "1"],
            ["copy", "40", "0", "34", "6", "0"],
            ["short", "29", "29", "0", "0", "0"],
            ["refusal", "47", "47", "0", "0", "0"],
            ["multimodal", "34", "34", "0", "0", "0"]]
    b.table(40, 325, [150, 80, 80, 80, 90, 80], rows, "teal", size=12, row_h=32)
    raw_text(b, 320, 640, "none of the four known-bad recipes survives; copies die as duplicates", 11, PALETTE["teal"]["text"])

    b.group(640, 280, 580, 395, "Kept does not mean diverse", "orange")
    rows2 = [["set", "rows", "distinct-2", "mean cosine", "near > 0.9"],
             ["human (Dolly)", "600", "0.733", "0.059", "0.007"],
             ["stub, all", "600", "0.069", "0.174", "0.737"],
             ["stub, kept", "101", "0.187", "0.249", "0.099"],
             ["human, 101", "101", "0.850", "0.056", "0.000"]]
    b.table(660, 325, [140, 60, 110, 120, 110], rows2, "orange", size=12, row_h=34)
    b.card(660, 520, 540, 135, "what dedup did and did not do", ["near-copies fell from 0.737 to 0.099 of rows,", "distinct bigrams rose from 0.069 to 0.187,", "but the kept rows share long suffixes, so mean", "cosine went up, and human text is far wider"], "red", size=12, align="left")
    return b


@board("synthetic-data-generation-dedup-and-collapse")
def synth_dedup_collapse():
    b = Board(1240, 700, "Near-duplicates and recursive training", "Left: MinHash and LSH on 3,004 real Dolly instructions. Right: a toy of training on your own samples")
    b.group(20, 95, 600, 580, "MinHash and LSH", "blue")
    b.card(40, 140, 560, 100, "compare shingles, not strings", ["each instruction becomes a set of word 3-grams;", "128 hashed minima estimate the Jaccard overlap", "mean error 0.0002 over 4,000 random pairs"], "blue", size=12)
    b.card(40, 262, 560, 100, "32 bands of 4 rows find the pairs", ["2,003 pairs compared instead of 4,510,506,", "because rows that agree in a band share a bucket"], "purple", size=12)
    rows = [["exact Jaccard", "pairs", "found", "recall"],
            ["0.5 to 0.6", "1,266", "1,138", "0.899"],
            ["0.6 to 0.8", "84", "83", "0.988"],
            ["0.8 to 1.0", "62", "62", "1.000"]]
    b.table(60, 395, [160, 120, 120, 120], rows, "blue", size=13, row_h=36)
    b.card(40, 565, 560, 90, "clusters in real instruction data", ["keep one row per cluster above Jaccard 0.5:", "3,004 rows become 2,849"], "green", size=12)

    b.group(640, 95, 580, 580, "Training on your own samples", "orange")
    b.text(930, 148, "types alive (of 200), mean of 40 runs", 12, "orange", "700")
    series = [("replace", "train only on last samples", "red", [126.2, 46.2, 21.5]),
              ("accumulate", "keep real data and all generations", "green", [162.4, 162.4, 162.4]),
              ("mix 10%", "10% fresh real data each time", "blue", [131.7, 82.0, 81.8])]
    base = 395
    for si, (name, desc, color, vals) in enumerate(series):
        c = PALETTE[color]
        gx = 670 + si * 185
        for k, v in enumerate(vals):
            h = v / 200 * 190
            rect(b, gx + k * 50, base - h, 40, h, c["fill"], c["stroke"], 1.6, 3)
            raw_text(b, gx + k * 50 + 20, base - h - 5, f"{v:.1f}", 10, c["text"], weight="700")
            raw_text(b, gx + k * 50 + 20, base + 16, f"g{(1, 10, 30)[k]}", 11, INK)
        raw_text(b, gx + 70, base + 38, name, 12, c["text"], weight="700")
    line(b, 660, base, 1205, base, FAINT, 1.2)
    b.card(660, 460, 545, 100, "the tails go first", ["replace: entropy 4.149 nats falls to 2.634, and", "21.5 of 200 types remain. accumulate keeps 162.4:", "stops the loss, not the roughly 38 missing types"], "grey", size=12)
    b.card(660, 575, 545, 85, "a refitted Gaussian shrinks", ["sigma after 0, 10, 20, 50 generations (20 samples):", "mean 1.000, 0.671, 0.441, 0.126"], "yellow", size=12)
    return b


BASE_MATRIX = [[0.23, 0.27, 0.18, 0.32, 0.27, 0.13, 0.17, 0.26], [0.13, 0.21, 0.16, 0.28, 0.21, 0.1, 0.14, 0.18], [0.26, 0.15, 0.15, 0.27, 0.25, 0.15, 0.1, 0.19], [0.22, 0.14, 0.15, 0.24, 0.18, 0.15, 0.24, 0.23], [0.24, 0.2, 0.15, 0.22, 0.24, 0.17, 0.15, 0.23], [0.22, 0.23, 0.14, 0.18, 0.23, 0.16, 0.08, 0.28], [0.34, 0.2, 0.18, 0.26, 0.22, 0.09, 0.16, 0.19], [0.2, 0.17, 0.17, 0.23, 0.23, 0.12, 0.19, 0.19]]


@board("tuning-embedding-models-and-rerankers-contrastive")
def embed_contrastive():
    b = Board(1240, 700, "Contrastive loss: win the softmax over the batch", "Eight queries against eight documents from the toy domain, base all-MiniLM-L6-v2, cosine similarity")
    b.group(20, 95, 520, 415, "Base model cosine matrix (rows: queries)", "blue")
    cell, gx, gy = 38, 128, 160
    blue, orange = PALETTE["blue"], PALETTE["orange"]
    lo, hi = 0.08, 0.34
    for i in range(8):
        raw_text(b, gx - 16, gy + i * cell + cell / 2 + 4, f"q{i + 1}", 11, INK)
        raw_text(b, gx + i * cell + cell / 2, gy - 8, f"d{i + 1}", 11, INK)
        for j in range(8):
            v = BASE_MATRIX[i][j]
            t = (v - lo) / (hi - lo)
            shade = int(250 - 120 * max(0, min(1, t)))
            fill = f"rgb({shade},{min(255, shade + 4)},255)"
            stroke = orange["stroke"] if i == j else "#ced4da"
            rect(b, gx + j * cell, gy + i * cell, cell - 3, cell - 3, fill, stroke, 3 if i == j else 1, 4)
            raw_text(b, gx + j * cell + (cell - 3) / 2, gy + i * cell + cell / 2 + 3, f"{v:.2f}", 11, INK, weight="700" if i == j else "400")
    raw_text(b, 280, 482, "orange outline: the right document for that query", 11, orange["text"])
    raw_text(b, 280, 499, "mean cosine to right 0.197, to wrong 0.196", 12, blue["text"], weight="700")

    b.group(560, 95, 660, 415, "Same batch, different settings", "purple")
    rows = [["temperature", "scale", "loss"],
            ["0.01", "100", "8.2145"],
            ["0.05 (library default)", "20", "2.5865"],
            ["0.10", "10", "2.2106"],
            ["0.50", "2", "2.0839"]]
    b.table(580, 140, [250, 100, 120], rows, "purple", size=13, row_h=36)
    rows2 = [["negatives per query", "loss at 0.05"],
             ["1", "0.6904"],
             ["3", "1.9741"],
             ["7", "2.5865"]]
    b.table(580, 345, [250, 220], rows2, "purple", size=13, row_h=34)
    b.card(1070, 140, 135, 180, "scale", ["scale = 1 / temperature", "", "the library", "multiplies cosine", "by 20"], "grey", size=11)
    b.card(1070, 345, 135, 135, "more negatives", ["a harder task", "and a higher", "loss"], "grey", size=11)

    b.card(20, 520, 600, 160, "Check against the library", ["MultipleNegativesRankingLoss on the same batch:", "library 2.586478, from scratch 2.586478", "probability on the right document: mean 0.104", "the base model cannot read the nicknames"], "green", size=12)
    b.card(640, 520, 580, 160, "After fine-tuning on 416 pairs", ["same batch, temperature 0.05:", "in-batch negatives 2.5865 -> 1.0621", "plus one hard negative per pair -> 1.3016", "(a training batch: the pairs were in the training set)"], "orange", size=12)
    return b


@board("tuning-embedding-models-and-rerankers-results")
def embed_results():
    b = Board(1240, 720, "What tuning bought, and what it did not", "128 documents about 32 invented internal tools; queries use staff nicknames the documents never contain")
    b.group(20, 95, 700, 330, "Recall before and after (a toy, one run)", "teal")
    rows = [["model", "split", "recall@1", "recall@5", "recall@10", "nDCG@10"],
            ["base", "seen docs", "0.404", "0.572", "0.615", "0.501"],
            ["in-batch", "seen docs", "0.635", "0.841", "0.923", "0.771"],
            ["+ hard neg", "seen docs", "0.606", "0.827", "0.875", "0.734"],
            ["base", "unseen", "0.132", "0.208", "0.292", "0.196"],
            ["in-batch", "unseen", "0.174", "0.215", "0.333", "0.232"],
            ["+ hard neg", "unseen", "0.167", "0.188", "0.222", "0.187"]]
    b.table(40, 140, [110, 110, 105, 105, 115, 105], rows, "teal", size=12, row_h=34)
    raw_text(b, 370, 408, "seen: 208 new phrasings of trained documents; unseen: 144 queries on 6 tools never trained", 11, PALETTE["teal"]["text"])

    b.group(740, 95, 480, 330, "Bi-encoder against cross-encoder", "orange")
    b.card(760, 140, 210, 100, "bi-encoder", ["documents embedded once", "query: 1 pass", "cosine to every document"], "orange", size=11)
    b.card(990, 140, 210, 100, "cross-encoder", ["query + document read", "together, 10 passes", "per query here"], "purple", size=11)
    b.arrow((865, 240), (865, 285))
    b.arrow((1095, 240), (1095, 285))
    b.card(760, 285, 210, 110, "tune the bi-encoder", ["stage 1: fetch the top 10", "from 128 documents"], "green", size=11)
    b.card(990, 285, 210, 110, "reorder the top 10", ["stage 2: only helps if", "the answer is in them"], "red", size=11)

    b.group(20, 445, 1200, 255, "Two-stage retrieval with an off-the-shelf cross-encoder (ms-marco-MiniLM-L6-v2)", "red")
    rows2 = [["first stage", "split", "answer in top 10", "recall@1", "after rerank"],
             ["base bi-encoder", "seen docs", "0.615", "0.404", "0.385"],
             ["base bi-encoder", "unseen docs", "0.292", "0.132", "0.104"],
             ["tuned bi-encoder", "seen docs", "0.923", "0.635", "0.394"],
             ["tuned bi-encoder", "unseen docs", "0.333", "0.174", "0.111"]]
    b.table(40, 490, [210, 170, 190, 130, 150], rows2, "red", size=12, row_h=34)
    b.card(930, 490, 270, 170, "read it carefully", ["the reranker never saw", "the nicknames, so it", "undoes what stage 1", "learned: 0.635 -> 0.394"], "yellow", size=12)
    return b


def main(names):
    todo = [k for k in BOARDS if not names or any(n in NAMES[k] for n in names)]
    for key in todo:
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
