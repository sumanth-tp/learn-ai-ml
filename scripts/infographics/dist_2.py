"""Infographics for docs/mlops/distributed/02-dist-challenges and 03-dist-learning.

Run from the repo root:

    python3 scripts/infographics/dist_2.py            # all boards
    python3 scripts/infographics/dist_2.py ring       # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "dist"
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


def dot(b, cx, cy, r, fill, stroke, width=2):
    b.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def rect(b, x, y, w, h, fill, stroke, width=2, rx=5):
    b.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


@board("distributed-ml-challenges-ring-allreduce")
def ring():
    b = Board(1200, 640, "Ring all-reduce: traffic that stops growing", "Four workers, one ring, two phases of N-1 steps each")
    b.group(20, 95, 520, 360, "The ring for N = 4", "blue")
    w = [b.card(60, 165, 130, 60, "worker 0", ["chunks a b c d"], "blue", size=11),
         b.card(370, 165, 130, 60, "worker 1", ["chunks a b c d"], "blue", size=11),
         b.card(370, 335, 130, 60, "worker 2", ["chunks a b c d"], "blue", size=11),
         b.card(60, 335, 130, 60, "worker 3", ["chunks a b c d"], "blue", size=11)]
    b.arrow(w[0].right(), w[1].left(), label="1 chunk", color="orange")
    b.arrow(w[1].bottom(), w[2].top(), label="1 chunk", color="orange")
    b.arrow(w[2].left(), w[3].right(), label="1 chunk", color="orange")
    b.arrow(w[3].top(), w[0].bottom(), label="1 chunk", color="orange")
    b.card(205, 230, 160, 80, "each step", ["send 1 chunk right,", "receive 1 from left"], "yellow", size=11)
    b.card(40, 410, 230, 36, "1: reduce-scatter, 3 steps", [], "green", size=11)
    b.card(290, 410, 230, 36, "2: all-gather, 3 steps", [], "purple", size=11)

    b.group(560, 95, 620, 360, "Data sent per worker, in multiples of the model", "teal")
    rows = [["N", "ring", "2(N-1)/N", "single reducer"],
            ["2", "1.000", "1.000", "2"],
            ["4", "1.500", "1.500", "4"],
            ["8", "1.750", "1.750", "8"],
            ["16", "1.875", "1.875", "16"],
            ["64", "1.969", "1.969", "64"]]
    b.table(585, 140, [90, 140, 150, 210], rows, "teal", size=13, row_h=34)
    b.text(870, 370, "measured by moving real chunks between 64 simulated workers;\nsingle reducer column: the one node that receives all N copies", 11, "teal", italic=True)

    b.group(20, 480, 1160, 140, "Time for one all-reduce, 400 MB model, 10 GB/s link, 20 us per message (named parameters)", "orange")
    rows = [["workers", "2", "4", "16", "64", "256"],
            ["ring (s)", "0.0400", "0.0601", "0.0756", "0.0813", "0.0899"],
            ["single reducer (s)", "0.1600", "0.3200", "1.2800", "5.1200", "20.4800"]]
    b.table(45, 525, [250, 150, 150, 150, 150, 150], rows, "orange", size=13, row_h=32)
    return b


@board("distributed-ml-challenges-stragglers-checkpoints")
def stragglers():
    b = Board(1200, 700, "Stragglers and failures", "Two simulated costs of scale: waiting for the slowest, and losing work to a crash")
    b.group(20, 95, 560, 300, "Synchronous step time, sigma 0.5", "red")
    rows = [["workers", "step time"], ["1", "1.184"], ["4", "1.819"], ["16", "2.467"], ["64", "3.327"], ["256", "4.261"]]
    b.table(40, 140, [110, 110], rows, "red", size=13, row_h=34)
    rows = [["64 workers + backups", "step time"], ["0", "3.327"], ["2", "2.436"], ["8", "1.819"], ["16", "1.510"]]
    b.table(285, 140, [185, 90], rows, "green", size=12, row_h=34)
    b.text(300, 375, "median worker takes 1.0; backups cost 3.1% to 25% extra machines", 11, "red", italic=True)

    b.group(600, 95, 580, 300, "Checkpoint interval, 400 h of work", "orange")
    rows = [["every", "overhead"], ["0.10 h", "56.1%"], ["0.50 h", "22.3%"], ["1.00 h", "27.0%"], ["2.00 h", "48.0%"], ["8.00 h", "390.6%"]]
    b.table(625, 140, [110, 120], rows, "orange", size=13, row_h=34)
    b.card(880, 140, 285, 120, "one failure every 3.09 h", ["checkpoint 0.05 h, restart 0.05 h", "best simulated: 0.5 h", "first-order guess: 0.556 h"], "yellow", size=11)
    b.text(890, 375, "too rare: lost work. too often: time spent saving.", 11, "orange", italic=True)

    b.group(20, 420, 560, 260, "A real figure: Llama 3 405B pre-training", "purple")
    b.card(45, 465, 510, 90, "54 days, 16K GPUs", ["466 interruptions: 47 planned, 419 unexpected", "419 in 1296 h = one every 3.09 h"], "purple", size=12)
    b.card(45, 575, 510, 85, "from the paper", ["about 78% of unexpected ones: hardware", "more than 90% effective training time"], "grey", size=12)

    b.group(600, 420, 580, 260, "Consistency: sync every k steps, 4 replicas", "blue")
    rows = [["sync every", "replica gap", "distance to optimum"],
            ["1", "0.000", "0.0457"], ["5", "2.547", "0.1096"], ["25", "6.188", "0.3480"], ["never", "6.728", "0.4632"]]
    b.table(625, 465, [140, 170, 230], rows, "blue", size=13, row_h=34)
    b.text(890, 665, "strong: exact and slow. eventual: fast and drifting.", 11, "blue", italic=True)
    return b

@board("programming-models-mapreduce-spark")
def mapreduce_spark():
    b = Board(1240, 680, "MapReduce and Spark: same idea, different cost", "Move the computation to the data; keep data in memory when you loop")
    b.group(20, 95, 740, 250, "MapReduce word count, 100,000 words, 8 map tasks", "blue")
    m = b.card(40, 150, 150, 80, "map", ["each task turns", "its split into", "(word, 1) pairs"], "blue", size=11)
    c = b.card(225, 150, 150, 80, "combine", ["add up pairs", "inside each task"], "teal", size=11)
    sh = b.card(410, 150, 150, 80, "shuffle", ["send each word to", "one reducer"], "orange", size=11)
    r = b.card(595, 150, 150, 80, "reduce", ["sum the counts", "per word"], "green", size=11)
    b.arrow(m.right(), c.left())
    b.arrow(c.right(), sh.left())
    b.arrow(sh.right(), r.left())
    b.card(40, 255, 340, 70, "pairs through the shuffle", ["no combiner: 100,000", "with a combiner: 3,979 (4.0%)"], "red", size=12)
    b.card(400, 255, 345, 70, "lecture example", ["1000 GB over 100 mappers:", "10 GB each, in parallel"], "yellow", size=12)

    b.group(780, 95, 440, 250, "20 iterations of gradient descent", "orange")
    rows = [["", "bytes read"], ["re-read each time", "14.16 MB"], ["cached after first", "0.71 MB"]]
    b.table(800, 145, [210, 190], rows, "orange", size=13, row_h=36)
    b.text(1000, 285, "ratio 20, same weights either way", 12, "orange", italic=True)
    b.text(1000, 318, "named parameters: 10 GB per node at 200 MB/s\n50 s per pass: 1000 s against 50 s", 11, "orange", italic=True)

    b.group(20, 370, 1200, 290, "Spark RDD lineage: a lost partition is rebuilt, not the whole dataset", "purple")
    a = b.card(50, 430, 200, 70, "data", ["4 partitions"], "grey", size=12)
    sq = b.card(330, 430, 200, 70, "map: square", ["persisted in memory"], "purple", size=12)
    ev = b.card(610, 430, 200, 70, "filter: even", ["then reduce: sum"], "purple", size=12)
    b.arrow(a.right(), sq.left(), label="lineage")
    b.arrow(sq.right(), ev.left())
    rows = [["action", "computations"],
            ["first", "8 (4 map + 4 filter)"],
            ["second, cached", "4 (filters only)"],
            ["one part lost", "5 (4 filters + 1 map)"]]
    b.table(840, 420, [150, 220], rows, "purple", size=12, row_h=34)
    b.card(50, 540, 760, 95, "why iterative ML prefers it", ["MapReduce writes results to disk between jobs, so every iteration pays for reading again.", "RDDs keep the working set in memory and remember how to recompute it."], "green", size=12)
    return b


@board("programming-models-parameter-server")
def parameter_server():
    b = Board(1200, 640, "Parameter server: shard the model, push and pull", "Four workers, four servers, one slice of the weights each")
    b.group(20, 95, 560, 360, "Sharded by key range", "blue")
    ws = [b.card(40, 145 + i * 62, 130, 48, f"worker {i}", [], "blue", size=11) for i in range(4)]
    ss = [b.card(430, 145 + i * 62, 130, 48, f"server {i}", [f"keys {5 * i} to {5 * i + 4}"], "orange", size=11) for i in range(4)]
    for w in ws:
        for sv in ss:
            b.arrow(w.right(), sv.left(), color="grey", width=1.0)
    b.card(215, 205, 175, 110, "each step", ["push gradient slices", "pull fresh weights", "(20 keys, 4 shards)"], "yellow", size=11)
    b.card(40, 405, 520, 36, "equals single-machine GD: max difference 1.1e-16", [], "green", size=12)

    b.group(600, 95, 580, 360, "Load on one server shard, 100 MB model, 10 GB/s", "teal")
    rows = [["workers", "servers", "receives MB", "PS s", "ring s"],
            ["4", "1", "400", "0.080", "0.015"],
            ["4", "4", "100", "0.020", "0.015"],
            ["16", "1", "1600", "0.320", "0.019"],
            ["16", "4", "400", "0.080", "0.019"],
            ["64", "1", "6400", "1.280", "0.020"],
            ["64", "16", "400", "0.080", "0.020"]]
    b.table(615, 140, [90, 90, 150, 110, 110], rows, "teal", size=12, row_h=34)

    b.card(20, 480, 570, 140, "values received per server", ["30 steps, 4 workers, 5 keys per server:", "4 servers get 600 each", "one server would get 2,400"], "purple", size=12)
    b.card(610, 480, 570, 140, "reading the table", ["one server: load grows with N", "shards divide it: N/P copies each", "ring: no server, about 2x the model per worker"], "grey", size=12)
    return b


@board("core-algorithms-distributed-kmeans")
def dist_kmeans():
    b = Board(1200, 640, "Distributed k-means: send summaries, not data", "k = 10 clusters, d = 100 dimensions, 20,000 points on 4 workers")
    b.group(20, 95, 640, 330, "One iteration", "blue")
    ws = [b.card(40, 145 + i * 62, 150, 48, f"worker {i}", ["its shard"], "blue", size=11) for i in range(4)]
    red = b.card(300, 195, 160, 120, "reduce", ["add the sums", "add the counts", "divide"], "orange", size=12)
    new = b.card(490, 195, 150, 120, "new centroids", ["k x d = 1,000", "values"], "green", size=12)
    for w in ws:
        b.arrow(w.right(), red.left(), label="" )
    b.arrow(red.right(), new.left())
    b.text(245, 138, "sums + counts: 1,010 values each", 11, "blue", italic=True)
    b.arrow(new.bottom(), (565, 395), via=[(565, 395), (115, 395), (115, 400)], color="red", dashed=True, label="broadcast, repeat", label_at=0.5)

    b.group(680, 95, 500, 330, "Two checks against one machine", "teal")
    b.card(700, 145, 460, 80, "sums and counts: exact", ["max centroid difference from", "scikit-learn Lloyd: 1.1e-14"], "green", size=12)
    b.card(700, 245, 460, 80, "mean of local means: wrong", ["max centroid difference: 9.93", "the shards have different sizes per cluster"], "red", size=12)
    b.card(700, 345, 460, 60, "10 iterations, 4 workers", ["1,010 values per worker per iteration"], "grey", size=12)

    b.group(20, 450, 1160, 170, "Traffic per iteration, all 4 workers", "orange")
    rows = [["points N", "raw data values", "summary values", "ratio"],
            ["20,000", "2,000,000", "4,040", "495"],
            ["200,000", "20,000,000", "4,040", "4,950"],
            ["20,000,000", "2,000,000,000", "4,040", "495,050"]]
    b.table(45, 495, [220, 330, 300, 250], rows, "orange", size=13, row_h=30)
    return b


@board("core-algorithms-fdm-dbscan")
def fdm_dbscan():
    b = Board(1240, 660, "Pruning locally: association rules and DBSCAN", "Both end exactly where a single machine would")
    b.group(20, 95, 700, 520, "FDM-style mining, 4 sites x 1,500 transactions, 6% support", "purple")
    rows = [["level", "candidates", "counted", "large", "values: all vs FDM"],
            ["2", "171", "63", "37", "684 vs 403"],
            ["3", "44", "44", "10", "176 vs 278"],
            ["4", "5", "3", "3", "20 vs 13"],
            ["total", "", "", "50", "880 vs 694"]]
    b.table(45, 140, [90, 130, 120, 100, 220], rows, "purple", size=13, row_h=34)
    b.card(45, 330, 650, 90, "the rule that makes it safe", ["a globally large itemset is locally large at some site", "(otherwise every site is below the threshold and so is the sum)"], "yellow", size=12)
    b.card(45, 440, 650, 80, "result", ["69 frequent itemsets, identical to one machine", "the saving depends on the data: level 3 costs more"], "green", size=12)
    b.card(45, 535, 650, 60, "values = counts sent; FDM-style also counts poll requests", [], "grey", size=11)

    b.group(740, 95, 480, 520, "DBSCAN on 4 strips with an eps halo", "teal")
    xs = [770, 880, 990, 1100]
    for i, x0 in enumerate(xs):
        rect(b, x0, 150, 100, 110, PALETTE["teal"]["fill"], PALETTE["teal"]["stroke"], 2)
        raw_text(b, x0 + 50, 200, f"strip {i}", 13, PALETTE["teal"]["text"], weight="700")
        raw_text(b, x0 + 50, 222, f"halo {[117, 160, 153, 106][i]}", 11, INK)
    for x0 in xs[1:]:
        line(b, x0, 140, x0, 270, PALETTE["red"]["stroke"], 2, "5 4")
    raw_text(b, 980, 296, "dashed: strip borders; halo = points within eps of a border", 10, FAINT)
    b.card(765, 320, 430, 70, "core points", ["agree with scikit-learn: True (2,929 core)"], "green", size=12)
    b.card(765, 405, 430, 70, "clusters", ["adjusted Rand index on core points: 1.0", "2 clusters, 9 noise points, same set"], "green", size=12)
    b.card(765, 490, 430, 100, "what crossed borders", ["3,316 core-to-core edges between strips", "a smarter message would send one edge per", "pair of local clusters"], "orange", size=11)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or v.endswith(name) or name in v)
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
