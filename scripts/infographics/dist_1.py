"""Infographics for docs/mlops/distributed/01-dist-foundations.

Run from the repo root:

    ./venv/bin/python scripts/infographics/dist_1.py            # all boards
    ./venv/bin/python scripts/infographics/dist_1.py amdahl     # just one
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


def cell(b, x, y, w, h, fill, stroke, width=1.2):
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


@board("why-distribute-machine-learning-data-vs-model")
def data_vs_model():
    b = Board(1200, 650, "Two ways to split the work", "Replicate the model and split the data, or split the model itself")
    b.group(20, 95, 560, 360, "Data parallelism", "blue")
    ws = []
    for i, name in enumerate(["worker 1", "worker 2", "worker 3 ... 8"]):
        ws.append(b.card(40, 135 + i * 82, 230, 68, name, ["full model copy", "its own data shard"], "blue", size=12))
    ar = b.card(340, 195, 220, 100, "all-reduce", ["average the gradients", "every copy stays identical"], "purple", size=12)
    for w in ws:
        b.arrow(w.right(), ar.left())
    b.card(340, 330, 220, 100, "global batch", ["local 32 x 8 workers", "= 256", "learning rate 0.1 x 8 = 0.8"], "green", size=12)

    b.group(620, 95, 560, 360, "Model parallelism", "orange")
    stages = []
    for i in range(4):
        stages.append(b.card(650 + i * 130, 150, 112, 84, f"GPU {i + 1}", [f"slice {i + 1} of 4", "6 GB"], "orange", size=12))
    for i in range(3):
        b.arrow(stages[i].right(), stages[i + 1].left(), label="act" if i == 1 else "")
    b.card(650, 275, 520, 80, "when the model does not fit one device", ["24 GB model over 4 GPUs = 6 GB each,", "activations flow from slice to slice"], "yellow", size=12)
    b.card(650, 375, 520, 62, "two forms: pipeline (by layers), tensor (inside a layer)", [], "grey", size=12)

    b.group(20, 480, 1160, 150, "Why the model might not fit: fp32 training with Adam, 16 bytes per parameter before activations", "red")
    rows = [["parameters", "0.135 B", "1 B", "7 B", "70 B"],
            ["memory", "2.2 GB", "16.0 GB", "112.0 GB", "1120.0 GB"]]
    b.table(60, 530, [200, 200, 200, 200, 200], rows, "red", size=13, row_h=36)
    return b


@board("why-distribute-machine-learning-sync-async-traffic")
def sync_async():
    b = Board(1240, 640, "What synchronising costs", "Waiting for stragglers, using stale gradients, and who carries the traffic")
    b.group(20, 95, 380, 400, "Synchronous: wait for the slowest", "orange")
    rows = [["workers", "sync efficiency"], ["1", "1.000"], ["2", "0.876"], ["4", "0.785"], ["8", "0.712"],
            ["16", "0.656"], ["32", "0.609"], ["64", "0.572"]]
    b.table(45, 140, [150, 200], rows, "orange", size=13, row_h=34)
    b.text(210, 440, "mean worker step 103.3 ms; the slowest of 8\nsets the pace at 144.1 ms", 11, "orange", italic=True)

    b.group(420, 95, 390, 400, "Asynchronous: never wait, but stale", "purple")
    rows = [["staleness", "loss after 25 steps"], ["0", "0.004025"], ["1", "0.004024"], ["2", "0.027851"],
            ["3", "4.520328"], ["4", "29.170128"], ["5", "100.522979"]]
    b.table(445, 140, [130, 230], rows, "purple", size=13, row_h=34)
    b.text(615, 400, "same learning rate; beyond one or two steps of\nstaleness the same update rule diverges", 11, "purple", italic=True)

    b.group(830, 95, 390, 400, "Traffic, gradient size = 1", "teal")
    rows = [["workers", "server in", "ring/worker"], ["2", "2.0", "1.000"], ["4", "4.0", "1.500"],
            ["8", "8.0", "1.750"], ["16", "16.0", "1.875"], ["64", "64.0", "1.969"]]
    b.table(845, 140, [80, 120, 140], rows, "teal", size=12, row_h=36)
    b.text(1025, 400, "the server link grows with N;\nring traffic per worker tends to 2", 11, "teal", italic=True)

    b.card(20, 525, 590, 90, "synchronous", ["consistent, every step uses the same weights", "but the slowest worker decides the step time"], "orange", size=12)
    b.card(630, 525, 590, 90, "asynchronous", ["no waiting, higher throughput", "each gradient was computed on older weights"], "purple", size=12)
    return b


@board("scalable-frameworks-amdahl-mapreduce-spark")
def amdahl_frameworks():
    b = Board(1240, 700, "The serial part sets the ceiling", "Amdahl's law, then two frameworks built around cluster costs")
    b.group(20, 95, 560, 410, "Speed-up S(n) = 1 / ((1-p) + p/n)", "blue")
    rows = [["workers", "p=0.50", "p=0.90", "p=0.95", "p=0.99"],
            ["1", "1.00", "1.00", "1.00", "1.00"],
            ["2", "1.33", "1.82", "1.90", "1.98"],
            ["4", "1.60", "3.08", "3.48", "3.88"],
            ["10", "1.82", "5.26", "6.90", "9.17"],
            ["100", "1.98", "9.17", "16.81", "50.25"],
            ["1000", "2.00", "9.91", "19.63", "90.99"],
            ["ceiling", "2.0", "10.0", "20.0", "100.0"]]
    b.table(40, 140, [100, 100, 100, 100, 100], rows, "blue", size=13, row_h=36)
    b.card(40, 440, 520, 50, "p = 0.9, n = 10: 5.26x. It takes 9 workers to reach 5x.", [], "green", size=12)

    b.group(600, 95, 620, 200, "MapReduce: map, shuffle, reduce on commodity nodes", "orange")
    m = b.card(620, 140, 150, 70, "map", ["(word, 1) pairs", "per partition"], "orange", size=12)
    sh = b.card(795, 140, 150, 70, "shuffle", ["group by key", "across nodes"], "orange", size=12)
    rd = b.card(970, 140, 230, 70, "reduce", ["sum per word"], "orange", size=12)
    b.arrow(m.right(), sh.left())
    b.arrow(sh.right(), rd.left())
    b.card(620, 225, 580, 55, "word count, 4 lines: 23 records shuffled; with a local combiner 14", [], "grey", size=12)

    b.group(600, 315, 620, 190, "Spark: keep the dataset in memory, rebuild from lineage", "purple")
    b.card(620, 360, 280, 95, "10 passes over 1.76 MB", ["read from disk each time: 17.60 MB", "cached after pass one: 1.76 MB"], "purple", size=12)
    b.card(920, 360, 280, 95, "lose partition 1", ["3 computations to rebuild it,", "partitions 0 and 2 untouched"], "purple", size=12)

    b.card(20, 535, 1200, 140, "Reading the table", ["Even with 99% of the work parallel, 1000 workers reach 90.99x, not 1000x.", "Frameworks attack the other costs: Hadoop MapReduce is fault-tolerant batch on cheap nodes,", "Spark's in-memory RDDs make the many passes of iterative ML cheap."], "grey", size=13)
    return b


@board("scalable-frameworks-ring-all-reduce")
def ring_board():
    b = Board(1240, 660, "Ring all-reduce", "Four workers, one number per chunk: sum every chunk on every worker in 6 steps")
    b.group(20, 95, 600, 330, "Worker inputs, chunk c holds (w + 1) + 10c", "blue")
    rows = [["", "chunk 0", "chunk 1", "chunk 2", "chunk 3"],
            ["worker 0", "1", "11", "21", "31"],
            ["worker 1", "2", "12", "22", "32"],
            ["worker 2", "3", "13", "23", "33"],
            ["worker 3", "4", "14", "24", "34"],
            ["every worker ends", "10", "50", "90", "130"]]
    b.table(40, 140, [180, 90, 90, 90, 90], rows, "blue", size=13, row_h=40)
    b.text(320, 400, "reduce-scatter: 3 steps, neighbours add\nall-gather: 3 steps, sums are copied round", 12, "blue", italic=True)

    b.group(640, 95, 580, 330, "N workers: steps and traffic", "teal")
    rows = [["workers", "steps", "sent per worker", "2(N-1)/N"],
            ["2", "2", "1.0000", "1.0000"],
            ["4", "6", "1.5000", "1.5000"],
            ["8", "14", "1.7500", "1.7500"],
            ["16", "30", "1.8750", "1.8750"],
            ["64", "126", "1.9688", "1.9688"]]
    b.table(660, 140, [100, 100, 190, 150], rows, "teal", size=13, row_h=40)
    b.text(930, 400, "sent per worker, as a multiple of the gradient size", 11, "teal", italic=True)

    ring = b.group(20, 450, 600, 190, "Each step: pass one chunk to the next worker", "purple")
    cards = []
    for i in range(4):
        cards.append(b.card(38 + i * 148, 490, 108, 70, f"worker {i}", ["pass a chunk,", "add or copy"], "purple", size=11))
    for i in range(3):
        b.arrow(cards[i].right(), cards[i + 1].left())
    b.arrow(cards[3].bottom(), cards[0].bottom(), via=[(536, 598), (92, 598)], dashed=True, color="purple")

    b.group(640, 450, 580, 190, "Real run: 2 processes, gloo backend, CPU", "green")
    b.card(660, 495, 540, 125, "all_reduce(SUM) of [1,2,3] and [2,4,6]", ["sum  [3.0, 6.0, 9.0] on both ranks", "mean [1.5, 3.0, 4.5] after dividing by 2", "all_gather returns ranks [0.0, 1.0]"], "green", size=12)
    return b


@board("data-parallelism-training-loop")
def dp_loop():
    b = Board(1240, 700, "Data parallelism is the same step, split eight ways", "Replicate, shard, compute locally, average, update")
    steps = [("1. replicate", ["same initial weights", "broadcast from rank 0"], "blue"),
             ("2. shard", ["each worker takes", "its own slice of the batch"], "green"),
             ("3. local gradient", ["forward and backward", "on the shard only"], "orange"),
             ("4. all-reduce", ["average the gradients", "across workers"], "purple"),
             ("5. update", ["identical step,", "replicas stay identical"], "red")]
    cards = []
    for i, (t, ln, c) in enumerate(steps):
        cards.append(b.card(20 + i * 240, 100, 215, 90, t, ln, c, size=12))
    for i in range(4):
        b.arrow(cards[i].right(), cards[i + 1].left())
    b.arrow(cards[4].bottom(), cards[2].bottom(), via=[(1100, 225), (500, 225)], label="next step", color="red", dashed=True, label_at=0.5)

    b.group(20, 265, 590, 245, "Why averaging is exact (numpy, 8 shards of 32)", "teal")
    rows = [["check", "max error"],
            ["8 shard gradients averaged vs full batch of 256", "2.22e-15"],
            ["uneven shards 200 and 56, plain mean", "0.567"],
            ["uneven shards, weighted by shard size", "4.44e-16"],
            ["25 steps, 1 process vs 8 replicas", "4.44e-16"]]
    b.table(40, 310, [390, 160], rows, "teal", size=12, row_h=34)

    b.group(630, 265, 590, 245, "Real run: 2 processes, gloo, 161-parameter network", "green")
    rows = [["comparison after 30 steps", "max difference"],
            ["manual all-reduce vs single process", "1.19e-07"],
            ["DDP vs single process", "1.19e-07"],
            ["DDP vs manual all-reduce", "0.00e+00"]]
    b.table(650, 310, [370, 170], rows, "green", size=12, row_h=34)

    b.card(20, 535, 590, 140, "weighted mean when shards differ", ["A plain mean of per-worker means over-weights the", "small shard: error 0.567. Weighting by shard size", "restores the full-batch gradient exactly."], "yellow", size=12)
    b.card(630, 535, 590, 140, "DDP does what the manual loop does", ["broadcast state from rank 0, then all-reduce gradient", "buckets while the backward pass is still running."], "grey", size=12)
    return b


@board("data-parallelism-scaling-rule")
def dp_scaling():
    b = Board(1240, 660, "Bigger batches need a bigger step, and communication has a price", "Logistic regression, 2 epochs, mean of 10 seeds; then the scaling model with illustrative parameters")
    b.group(20, 95, 600, 330, "Linear scaling rule: batch x8, learning rate x8", "blue")
    rows = [["setting", "updates", "test loss", "accuracy"],
            ["1 worker, B=32, lr 0.1", "192", "0.2235", "0.9634"],
            ["8 workers, B=256, lr 0.1", "24", "0.4608", "0.9583"],
            ["8 workers, B=256, lr 0.8", "24", "0.2208", "0.9632"],
            ["same, with warmup", "24", "0.2300", "0.9627"]]
    b.table(40, 140, [270, 90, 110, 100], rows, "blue", size=12, row_h=38)
    b.text(320, 400, "eight times fewer updates: keep the step size and it undertrains;\nscale the learning rate and it matches the small batch", 11, "blue", italic=True)

    b.group(640, 95, 580, 330, "Scaling model: B=32, 5 ms per sample, 100 MB, 10 GB/s", "teal")
    rows = [["workers", "global batch", "step ms", "speedup", "efficiency"],
            ["1", "32", "160.00", "1.000", "1.000"],
            ["2", "64", "170.00", "1.882", "0.941"],
            ["8", "256", "177.50", "7.211", "0.901"],
            ["64", "2048", "179.69", "56.988", "0.890"],
            ["256", "8192", "179.92", "227.654", "0.889"]]
    b.table(660, 140, [90, 120, 100, 110, 110], rows, "teal", size=12, row_h=38)
    b.text(930, 400, "ring cost 2(K-1)/K x gradient / bandwidth, no overlap", 11, "teal", italic=True)

    b.group(20, 450, 1200, 190, "Eight workers: what moves the efficiency", "orange")
    rows = [["overlap", "1 GB/s link", "10 GB/s link"],
            ["0.0", "step 335.00 ms, efficiency 0.478", "step 177.50 ms, efficiency 0.901"],
            ["0.5", "step 247.50 ms, efficiency 0.646", "step 168.75 ms, efficiency 0.948"],
            ["0.9", "step 177.50 ms, efficiency 0.901", "step 161.75 ms, efficiency 0.989"]]
    b.table(40, 495, [140, 500, 500], rows, "orange", size=12, row_h=34)
    return b


@board("model-parallelism-splitting")
def mp_split():
    b = Board(1240, 660, "Two ways to cut a model", "Between layers (pipeline) or inside a layer (tensor)")
    b.group(20, 95, 590, 250, "Pipeline: whole layers on each device", "orange")
    st = []
    for i in range(4):
        st.append(b.card(40 + i * 142, 145, 126, 80, f"stage {i}", ["layers", f"{6 * i + 1} to {6 * i + 6}"], "orange", size=12))
    for i in range(3):
        b.arrow(st[i].right(), st[i + 1].left())
    b.text(315, 275, "activations cross each cut once per micro-batch\n24 GB model over 4 GPUs = 6 GB each", 12, "orange", italic=True)

    b.group(630, 95, 590, 250, "Tensor: one MLP block sliced across 2 devices", "purple")
    a = b.card(650, 140, 260, 60, "device 1", ["first matrix: left half of columns", "second matrix: top half of rows"], "purple", size=11)
    c = b.card(650, 215, 260, 60, "device 2", ["first matrix: right half of columns", "second matrix: bottom half of rows"], "purple", size=11)
    s = b.card(960, 165, 240, 90, "one sum", ["partial outputs added", "(an all-reduce)"], "green", size=12)
    b.arrow(a.right(), s.left())
    b.arrow(c.right(), s.left())

    b.group(20, 370, 1200, 260, "Checked in numpy (8 to 16 to 8 MLP, 6 rows)", "teal")
    rows = [["check", "result"],
            ["parameters, whole layer pair vs per device", "256 vs 128"],
            ["column split, row split, one sum vs unsplit layer", "max error 1.78e-15"],
            ["row-split first matrix without summing before the nonlinearity", "wrong by 3.97"],
            ["all-reduce payload (6 rows x 8 features x 4 bytes)", "192 bytes, any hidden width"]]
    b.table(40, 415, [700, 460], rows, "teal", size=13, row_h=40)
    return b


@board("model-parallelism-pipeline-bubble")
def mp_bubble():
    b = Board(1240, 700, "The pipeline bubble", "Forward pass, 4 stages, 8 micro-batches: 11 ticks, 32 busy cells of 44")
    b.group(20, 95, 700, 295, "Who is busy at each tick (digit = micro-batch)", "blue")
    x0, y0, cw, ch = 130, 150, 48, 44
    for s in range(4):
        raw_text(b, x0 - 12, y0 + s * (ch + 8) + ch / 2 + 5, f"stage {s}", 13, PALETTE["blue"]["text"], "end", "700")
        for t in range(11):
            j = t - s
            if 0 <= j < 8:
                cell(b, x0 + t * cw + 2, y0 + s * (ch + 8), cw - 4, ch, PALETTE["blue"]["fill"], PALETTE["blue"]["stroke"])
                raw_text(b, x0 + t * cw + cw / 2, y0 + s * (ch + 8) + ch / 2 + 5, str(j + 1), 14, PALETTE["blue"]["text"])
            else:
                cell(b, x0 + t * cw + 2, y0 + s * (ch + 8), cw - 4, ch, "#f1f3f5", "#ced4da", 1)
    raw_text(b, 400, 372, "idle cells: 12 of 44 = 3/11 = 0.273", 13, PALETTE["red"]["text"])

    b.group(740, 95, 480, 295, "Bubble = (S-1)/(m+S-1)", "orange")
    rows = [["micro-batches", "S=4", "S=8"], ["1", "0.750", "0.875"], ["2", "0.600", "0.778"],
            ["4", "0.429", "0.636"], ["8", "0.273", "0.467"], ["16", "0.158", "0.304"],
            ["32", "0.086", "0.179"], ["64", "0.045", "0.099"]]
    b.table(760, 140, [180, 130, 130], rows, "orange", size=12, row_h=28)

    b.card(20, 410, 580, 120, "more micro-batches, smaller bubble", ["At S=4 it takes 28 micro-batches to get under 10% idle.", "The cost: with all forwards before any backward,", "stage 0 holds m micro-batches of activations."], "yellow", size=12)
    b.card(620, 410, 600, 120, "GPipe's rule of thumb is m >= 4 x S", ["For S=4 that is m=16, where the formula still gives", "0.158. The paper reports the overhead as negligible", "in its experiments; measure on your own pipeline."], "grey", size=12)
    b.group(20, 555, 1200, 125, "Real run: 2 processes, stage 0 on rank 0 and stage 1 on rank 1, 4 micro-batches", "green")
    b.card(40, 600, 1160, 62, "gradients match the single-process model: stage 0 max difference 5.96e-08, stage 1 2.38e-07", [], "green", size=13)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or name in v)
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
