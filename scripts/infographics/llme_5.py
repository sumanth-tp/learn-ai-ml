"""Infographics for docs/llm-engineering/03-training-at-scale.

Run from the repo root:

    python3 scripts/infographics/llme_5.py            # all boards
    python3 scripts/infographics/llme_5.py llama3     # just one
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


def rect(b, x, y, w, h, fill, stroke, width=1.4, opacity=1.0, rx=4):
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def line(b, x1, y1, x2, y2, stroke=INK, width=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    b.parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{width}"{d} stroke-linecap="round"/>'
    )


@board("parallelism-strategies-for-llms-five-axes")
def five_axes():
    b = Board(1200, 640, "Five ways to split one training job", "Each axis cuts a different thing and pays a different communication bill")
    axes = [
        ("data", "blue", ["splits: the batch", "every GPU holds the model", "(or a shard of it)", "", "talks: gradients,", "once per step", "", "scales across servers"]),
        ("tensor", "orange", ["splits: each weight", "matrix inside a layer", "", "talks: 2 all-reduces", "forward, 2 backward,", "in every layer", "", "keep inside one server"]),
        ("pipeline", "purple", ["splits: the layers,", "into stages in a line", "", "talks: activations to", "the next stage", "", "pays: a bubble of idle", "time at start and end"]),
        ("context", "teal", ["splits: the sequence", "of one example", "", "talks: all-gather of", "keys and values", "(Llama 3 recipe)", "", "for 128K-token inputs"]),
        ("expert", "green", ["splits: the experts of", "a mixture-of-experts", "layer", "", "talks: all-to-all of", "tokens to their experts", "only MoE layers", "(chapter 4)"]),
    ]
    for i, (name, color, lines) in enumerate(axes):
        x = 20 + i * 236
        b.group(x, 95, 224, 330, name + " parallel", color)
        b.card(x + 12, 135, 200, 275, "", lines, color, size=12)
    b.group(20, 445, 1160, 175, "Which wall are you hitting?", "grey")
    b.card(40, 485, 360, 120, "weights and optimiser do not fit", ["shard them: FSDP / ZeRO (chapter 2),", "then tensor and pipeline parallel", "when one server is not enough"], "red", size=12)
    b.card(420, 485, 360, 120, "activations do not fit", ["sequence parallel, selective or full", "recomputation, context parallel", "for very long sequences"], "yellow", size=12)
    b.card(800, 485, 360, 120, "training is too slow", ["add data-parallel replicas;", "tokens per second grows with GPUs", "until the all-reduce shows"], "green", size=12)
    return b


@board("parallelism-strategies-for-llms-llama3-layout")
def llama3_layout():
    b = Board(1240, 700, "Llama 3 405B: 16,384 GPUs and a 78.6 GB budget", "Layout from the paper (Table 4); memory is the chapter's own arithmetic (blocks 2 and 3)")
    b.group(20, 95, 560, 330, "How the GPUs are carved up", "blue")
    b.card(40, 140, 520, 70, "128 data-parallel replicas", ["each sees a different slice of the batch"], "blue", size=12)
    b.card(60, 225, 480, 70, "16 pipeline stages per replica", ["126 layers, about 8 per stage"], "purple", size=12)
    b.card(80, 310, 440, 70, "8 tensor-parallel GPUs per stage", ["inside one server, on the fastest links"], "orange", size=12)
    raw_text(b, 300, 405, "8 x 16 x 128 = 16,384 GPUs   (context parallel = 1)", 13, INK, weight="700")

    b.group(600, 95, 620, 330, "Model states per GPU, 405.85B parameters", "teal")
    rows = [["ZeRO stage", "weights", "grads", "optimiser", "total GB"],
            ["0", "6.34", "6.34", "38.05", "50.73"],
            ["1", "6.34", "6.34", "0.30", "12.98"],
            ["2", "6.34", "0.05", "0.30", "6.69"],
            ["3", "0.85", "0.05", "0.30", "1.19"]]
    b.table(620, 140, [110, 110, 110, 130, 130], rows, "teal", size=13, row_h=34)
    raw_text(b, 910, 362, "bf16 weights and grads, fp32 master weights, m and v", 12, FAINT)
    raw_text(b, 910, 382, "Llama 3 shards optimiser and gradients;", 12, FAINT)
    raw_text(b, 910, 402, "weights stay whole during a step: the stage 2 row", 12, FAINT)

    b.group(20, 445, 700, 235, "Activations on the first stage, 8,192 tokens, micro-batch 1", "orange")
    rows = [["what is saved", "GB"],
            ["nothing, no parallelism", "5,986.6"],
            ["tensor parallel only", "896.3"],
            ["tensor + sequence parallel", "748.3"],
            ["+ selective recompute (no attention scores)", "71.9"],
            ["full recomputation", "33.8"]]
    b.table(40, 490, [520, 150], rows, "orange", size=13, row_h=30)

    b.group(740, 445, 480, 235, "Does it fit?", "green")
    b.card(760, 490, 440, 80, "6.7 GB states + 71.9 GB activations", ["= 78.6 GB per GPU"], "green", size=13, title_size=15)
    b.card(760, 585, 440, 80, "against 80 GB of HBM on an H100", ["only just: every saving above is needed"], "red", size=12)
    return b


@board("parallelism-strategies-for-llms-worked-70b")
def worked_70b():
    b = Board(1200, 640, "Worked example: a 70B model on 64 GPUs", "TP 8 x PP 2 x DP 4, the numbers printed by block 2")
    b.group(20, 95, 520, 515, "Step by step", "blue")
    b.card(40, 140, 480, 90, "1. count what must be stored", ["70.55 billion parameters x 16 bytes", "= 1,128.9 GB of model state"], "blue", size=13, title_size=14)
    b.arrow((280, 230), (280, 262), color=INK)
    b.card(40, 262, 480, 90, "2. cut it across the model-parallel GPUs", ["tensor 8 x pipeline 2 = 16 GPUs share one copy", "1,128.9 / 16 = 70.55 GB each (ZeRO stage 0)"], "orange", size=13, title_size=14)
    b.arrow((280, 352), (280, 384), color=INK)
    b.card(40, 384, 480, 90, "3. shard the state across the 4 replicas", ["optimiser 52.92 / 4 = 13.23 GB (stage 1)", "gradients 8.82 / 4 = 2.20 GB (stage 2)"], "purple", size=13, title_size=14)
    b.arrow((280, 474), (280, 506), color=INK)
    b.card(40, 506, 480, 85, "4. last, shard the weights", ["8.82 / 4 + 0.21 gathered layer = 2.42 GB (stage 3)", "17.85 GB in total"], "green", size=13, title_size=14)

    b.group(560, 95, 620, 515, "Per-GPU model state by ZeRO stage (GB)", "teal")
    scale = 6.2
    x0 = 650
    totals = [70.55, 30.87, 24.25, 17.85]
    comps = [("weights", "blue"), ("grads", "orange"), ("optimiser", "purple")]
    data = [(8.82, 8.82, 52.92), (8.82, 8.82, 13.23), (8.82, 2.20, 13.23), (2.42, 2.20, 13.23)]
    for i, row in enumerate(data):
        y = 170 + i * 90
        raw_text(b, x0 - 10, y + 26, f"stage {i}", 13, INK, anchor="end", weight="700")
        cx = x0
        for (name, color), v in zip(comps, row):
            w = v * scale
            rect(b, cx, y, w, 40, PALETTE[color]["fill"], PALETTE[color]["stroke"], 1.6, 1.0, 3)
            if w > 44:
                raw_text(b, cx + w / 2, y + 25, f"{v:.2f}", 12, PALETTE[color]["text"], weight="700")
            cx += w
        raw_text(b, cx + 8, y + 26, f"{totals[i]:.2f}", 14, INK, anchor="start", weight="700")
    line(b, x0 + 80 * scale, 150, x0 + 80 * scale, 520, "#e03131", 2.2, "6 4")
    raw_text(b, x0 + 80 * scale - 6, 143, "80 GB card", 12, "#c92a2a", anchor="end", weight="700")
    for j, (name, color) in enumerate(comps):
        lx = 620 + j * 160
        rect(b, lx, 548, 18, 18, PALETTE[color]["fill"], PALETTE[color]["stroke"], 1.6, 1.0, 3)
        raw_text(b, lx + 26, 562, name, 13, INK, anchor="start")
    raw_text(b, 870, 594, "activations come on top and are not shown here", 12, FAINT)
    return b


@board("ddp-fsdp-and-zero-memory")
def zero_memory():
    b = Board(1240, 660, "What each GPU keeps: DDP and the three ZeRO stages", "Boxes drawn for 4 GPUs; the numbers are the ZeRO paper's 7.5B-parameter, 64-GPU example (block 2)")
    cols = [
        ("DDP (stage 0)", 1.0, 1.0, 1.0, "16 bytes per parameter", "120.0 GB", "2.0x gradient size"),
        ("ZeRO stage 1", 1.0, 1.0, 0.25, "4 + 12/N bytes", "31.4 GB", "2.0x, same as DDP"),
        ("ZeRO stage 2", 1.0, 0.25, 0.25, "2 + 14/N bytes", "16.6 GB", "2.0x, same as DDP"),
        ("ZeRO stage 3 (FSDP)", 0.25, 0.25, 0.25, "16/N bytes", "1.9 GB", "3.0x, one extra gather"),
    ]
    comps = [("weights 2 B", "blue", 2), ("gradients 2 B", "orange", 2), ("optimiser 12 B", "purple", 12)]
    unit = 15.0
    for i, (name, fw, fg, fo, formula, total, comm) in enumerate(cols):
        x = 30 + i * 300
        b.group(x, 95, 280, 545, name, "grey")
        top = 165
        fracs = (fw, fg, fo)
        for (label, color, size), frac in zip(comps, fracs):
            full = size * unit
            pal = PALETTE[color]
            rect(b, x + 40, top, 200, full, "#ffffff", pal["stroke"], 1.4, 0.6, 3)
            held = full * frac
            rect(b, x + 40, top + (full - held), 200, held, pal["fill"], pal["stroke"], 1.8, 1.0, 3)
            text_y = top + full / 2 + 4 if (frac == 1.0 or size > 4) else top + 19
            raw_text(b, x + 140, text_y, label + ("" if frac == 1.0 else f"  x {frac:g}"), 12, pal["text"], weight="700")
            top += full + 8
        raw_text(b, x + 140, 450, formula, 14, INK, weight="700")
        raw_text(b, x + 140, 488, total, 26, PALETTE["red"]["text"] if i == 0 else PALETTE["green"]["text"], weight="700")
        raw_text(b, x + 140, 510, "per GPU, 7.5B model, N = 64", 11, FAINT)
        b.card(x + 20, 528, 240, 60, "traffic per step", [comm], "yellow", size=12)
        raw_text(b, x + 140, 618, "solid = this GPU keeps it", 11, FAINT)
    return b


@board("ddp-fsdp-and-zero-toy-step")
def zero_toy_step():
    b = Board(1240, 620, "One sharded step, by hand: 8 parameters on 2 GPUs", "Plain gradient descent with learning rate 0.1, every weight starts at 1.0 (block 1)")

    def cells(x, y, values, color, cw=62, size=14, label=""):
        pal = PALETTE[color]
        for k, v in enumerate(values):
            rect(b, x + k * cw, y, cw - 4, 34, pal["fill"], pal["stroke"], 1.5, 1.0, 3)
            raw_text(b, x + k * cw + (cw - 4) / 2, y + 22, v, size, pal["text"], weight="700")
        if label:
            raw_text(b, x - 12, y + 22, label, 13, INK, anchor="end", weight="700")

    b.group(20, 95, 1200, 130, "1. Each GPU computes gradients on its own half of the batch", "blue")
    cells(150, 135, ["1", "2", "3", "4", "5", "6", "7", "8"], "blue", label="GPU 0")
    cells(150, 180, ["3", "2", "1", "0", "1", "2", "3", "4"], "blue", label="GPU 1")
    raw_text(b, 900, 175, "every slot is a gradient; GPU 0 and GPU 1 saw different data", 13, INK)

    b.group(20, 240, 1200, 130, "2. Reduce-scatter: add them up, average, and hand each GPU only its own half", "orange")
    cells(150, 280, ["2", "2", "2", "2"], "orange", label="GPU 0")
    cells(150, 325, ["3", "4", "5", "6"], "orange", label="GPU 1")
    raw_text(b, 560, 303, "(1 + 3) / 2 = 2,   (2 + 2) / 2 = 2,   (3 + 1) / 2 = 2,   (4 + 0) / 2 = 2", 13, INK, anchor="start")
    raw_text(b, 560, 348, "(5 + 1) / 2 = 3,   (6 + 2) / 2 = 4,   (7 + 3) / 2 = 5,   (8 + 4) / 2 = 6", 13, INK, anchor="start")

    b.group(20, 385, 1200, 100, "3. Each GPU updates only the weights it owns: new = 1.0 - 0.1 x gradient", "purple")
    cells(150, 425, ["0.8", "0.8", "0.8", "0.8"], "purple", label="GPU 0")
    cells(480, 425, ["0.7", "0.6", "0.5", "0.4"], "purple", label="GPU 1")
    raw_text(b, 1000, 447, "optimiser state exists only for the owned half", 13, INK)

    b.group(20, 500, 1200, 105, "4. All-gather: every GPU rebuilds the full weights for the next forward pass", "green")
    cells(150, 540, ["0.8", "0.8", "0.8", "0.8", "0.7", "0.6", "0.5", "0.4"], "green", label="both")
    raw_text(b, 900, 562, "same on both GPUs, as plain data parallelism requires", 13, INK)
    return b


@board("ddp-fsdp-and-zero-stage3-timeline")
def stage3_timeline():
    b = Board(1240, 580, "Stage 3 (FSDP): gather a layer, use it, drop it", "One GPU, three layers. The next layer's gather runs while the current layer computes, so the wait is hidden")
    b.group(20, 95, 1200, 320, "Time runs left to right", "teal")
    lanes = [("all-gather", 150), ("compute", 215), ("reduce-scatter", 280)]
    for name, y in lanes:
        raw_text(b, 145, y + 24, name, 13, INK, anchor="end", weight="700")

    def block(lane_y, pos, label, color):
        pal = PALETTE[color]
        x = 160 + pos * 72
        rect(b, x, lane_y, 66, 38, pal["fill"], pal["stroke"], 1.6, 1.0, 4)
        raw_text(b, x + 33, lane_y + 24, label, 11, pal["text"], weight="700")

    for pos, label in [(0, "layer 1"), (1, "layer 2"), (2, "layer 3"), (6, "layer 3"), (7, "layer 2"), (8, "layer 1")]:
        block(150, pos, label, "blue")
    for pos, label in [(1, "fwd 1"), (2, "fwd 2"), (3, "fwd 3")]:
        block(215, pos, label, "green")
    for pos, label in [(7, "bwd 3"), (8, "bwd 2"), (9, "bwd 1")]:
        block(215, pos, label, "red")
    for pos, label in [(8, "grad 3"), (9, "grad 2"), (10, "grad 1")]:
        block(280, pos, label, "orange")
    raw_text(b, 160 + 4.5 * 72 + 33, 238, "loss", 13, FAINT, weight="700")
    raw_text(b, 160 + 2 * 72 + 33, 345, "forward: gather each layer once", 13, PALETTE["green"]["text"], weight="700")
    raw_text(b, 160 + 8 * 72 + 33, 345, "backward: gather again, then scatter the gradient", 13, PALETTE["red"]["text"], weight="700")
    raw_text(b, 620, 388, "each layer is freed after use, so only about two full layers are resident at once", 13, INK)
    b.card(20, 435, 380, 125, "why it works", ["each GPU stores only 1/N of every layer;", "the rest is fetched when needed"], "teal", size=12)
    b.card(430, 435, 380, 125, "what it costs", ["weights cross the wire twice (forward and", "backward) plus the gradient scatter:", "3 units against DDP's 2"], "yellow", size=12)
    b.card(840, 435, 380, 125, "the Llama 3 variant", ["keep the weights gathered after forward:", "no second gather, but the full weights", "stay in memory until backward is done"], "purple", size=12)
    return b


def bit_row(b, x, y, sign, exp, man, bit_w=15):
    cursor = x
    for count, color in ((sign, "red"), (exp, "orange"), (man, "blue")):
        pal = PALETTE[color]
        rect(b, cursor, y, count * bit_w - 2, 26, pal["fill"], pal["stroke"], 1.5, 1.0, 3)
        cursor += count * bit_w


@board("mixed-precision-and-numerics-formats")
def formats_board():
    b = Board(1240, 640, "Five number formats: where the bits go", "Red sign, orange exponent (how big), blue mantissa (how many digits). All figures are printed by block 1")
    rows = [
        ("fp32", 1, 8, 23, ["3.40e38", "1.18e-38", "1.40e-45", "1.19e-7"]),
        ("bf16", 1, 8, 7, ["3.39e38", "1.18e-38", "9.18e-41", "0.0078"]),
        ("fp16", 1, 5, 10, ["65,504", "6.10e-5", "5.96e-8", "0.00098"]),
        ("fp8 e4m3", 1, 4, 3, ["448", "0.0156", "0.00195", "0.125"]),
        ("fp8 e5m2", 1, 5, 2, ["57,344", "6.10e-5", "1.53e-5", "0.25"]),
    ]
    b.group(20, 95, 1200, 330, "Bit layout and the numbers that follow from it", "blue")
    heads = ["largest", "smallest normal", "smallest subnormal", "gap above 1"]
    hx = [760, 870, 1000, 1130]
    for h, x in zip(heads, hx):
        raw_text(b, x, 150, h, 12, INK, weight="700")
    for i, (name, sg, ex, ma, vals) in enumerate(rows):
        y = 175 + i * 50
        raw_text(b, 150, y + 18, name, 14, INK, anchor="end", weight="700")
        bit_row(b, 165, y, sg, ex, ma)
        raw_text(b, 165 + 32 * 15 + 10, y + 18, f"{ex} exp, {ma} man", 11, FAINT, anchor="start") if False else None
        for v, x in zip(vals, hx):
            raw_text(b, x, y + 18, v, 13, INK, weight="400")
    b.card(20, 445, 390, 170, "bf16: fp32's range, fewer digits", ["same 8 exponent bits as fp32, so the same", "largest and smallest numbers.", "Only 7 mantissa bits: gap 0.0078 near 1.", "No loss scaling needed."], "blue", size=12)
    b.card(425, 445, 390, 170, "fp16: more digits, small range", ["10 mantissa bits, but the largest number is", "65,504 and numbers below 6.1e-5 lose digits.", "Small gradients vanish: use a loss scale."], "orange", size=12)
    b.card(830, 445, 390, 170, "fp8: a scale per block", ["3 or 2 mantissa bits, so 2% to 4% typical error.", "e4m3 for values, e5m2 for gradients.", "One scale per tensor is not enough;", "use one per small block."], "purple", size=12)
    return b


@board("mixed-precision-and-numerics-worked")
def precision_worked():
    b = Board(1240, 600, "Three things that go wrong with few bits, by hand", "Every number is printed by block 2")
    b.group(20, 95, 390, 480, "1. Storing 0.1", "blue")
    b.card(40, 140, 350, 105, "fp32", ["0.1 -> 0.10000000149", "error 1.5e-9"], "blue", size=13)
    b.card(40, 258, 350, 105, "fp16", ["0.1 -> 0.0999755859375", "error 2.4e-5"], "orange", size=13)
    b.card(40, 376, 350, 105, "bf16", ["0.1 -> 0.10009765625", "error 9.8e-5, four times fp16's"], "purple", size=13)
    raw_text(b, 215, 520, "fewer mantissa bits = a bigger step", 13, INK, weight="700")
    raw_text(b, 215, 545, "between neighbouring numbers", 13, INK, weight="700")

    b.group(425, 95, 390, 480, "2. Adding 0.01 to 32.0", "orange")
    b.card(445, 140, 350, 105, "fp16: gap near 32 is 0.03125", ["half a gap is 0.0156, bigger than 0.01", "so 32 + 0.01 = 32"], "orange", size=12)
    b.card(445, 258, 350, 105, "ten thousand additions", ["exact answer 100", "fp16 stops at 32.0 (after 2,798)", "bf16 stops at 4.0 (after 350)"], "red", size=12)
    b.card(445, 376, 350, 105, "the cure", ["add in fp32: 100.0030", "(100.0977 if the inputs are bf16)"], "green", size=12)
    raw_text(b, 620, 520, "small numbers added to big totals", 13, INK, weight="700")
    raw_text(b, 620, 545, "disappear: accumulate in fp32", 13, INK, weight="700")

    b.group(830, 95, 390, 480, "3. A gradient of 2e-8 in fp16", "purple")
    b.card(850, 140, 350, 105, "smallest fp16 number is 5.96e-8", ["2e-8 is under half of that", "stored as 0: the gradient is lost"], "red", size=12)
    b.card(850, 258, 350, 105, "multiply the loss by 1,024 first", ["gradient becomes 2.05e-5", "stored as 2.0504e-5"], "yellow", size=12)
    b.card(850, 376, 350, 105, "divide by 1,024 before the update", ["2.0023e-8, within 0.12% of 2e-8", "that is loss scaling"], "green", size=12)
    raw_text(b, 1025, 520, "the scale moves small gradients into", 13, INK, weight="700")
    raw_text(b, 1025, 545, "the range fp16 can hold", 13, INK, weight="700")
    return b


@board("mixed-precision-and-numerics-step")
def precision_step():
    b = Board(1240, 640, "One mixed-precision training step", "The recipe behind the 2 + 2 + 12 = 16 bytes per parameter of chapters 1 and 2")
    b.group(20, 95, 1200, 300, "Where each number lives and what format it is in", "teal")
    cards = [
        (40, "fp32 master weights", ["4 bytes each", "the true copy; the", "optimiser updates it"], "blue"),
        (270, "bf16 working copy", ["2 bytes each", "cast from the master", "for this step"], "green"),
        (500, "forward pass", ["matmuls in bf16,", "summed in fp32;", "softmax, norms and", "loss in fp32"], "orange"),
        (730, "backward pass", ["bf16 gradients,", "2 bytes each", "(fp16: multiply the loss", "by the scale first)"], "red"),
        (960, "update", ["unscale, check for inf,", "fp32 gradient to Adam:", "m and v, 8 bytes", "update the master"], "purple"),
    ]
    for x, title, lines, color in cards:
        b.card(x, 145, 215, 150, title, lines, color, size=12)
    for x in (255, 485, 715, 945):
        b.arrow((x, 220), (x + 15, 220), color=INK)
    b.arrow((1067, 295), (1067, 335), color=INK)
    b.arrow((1067, 335), (147, 335), color=INK, label="next step")
    b.arrow((147, 335), (147, 295), color=INK)
    b.group(20, 415, 590, 205, "Bytes per parameter", "yellow")
    b.table(40, 460, [230, 100, 100, 110], [["recipe", "weights", "grads", "state"], ["fp32 everywhere", "4", "4", "8"], ["bf16 + fp32 master", "2", "2", "12"], ["pure bf16 (risky)", "2", "2", "4"]], "yellow", size=12, row_h=30)
    b.group(630, 415, 590, 205, "Why it is not 8 bytes", "red")
    b.card(650, 460, 550, 140, "pure bf16 saved memory, then broke", ["34.7% of weights unchanged in the last step", "against 12.5% in fp32 (block 5)", "the 4-byte master copy is what keeps small", "updates from rounding away"], "red", size=12)
    return b


@board("mixture-of-experts-layer")
def moe_layer():
    b = Board(1240, 620, "A dense layer against a mixture-of-experts layer", "Same token, same attention; only the feed-forward block changes. Figures are Mixtral 8x7B from block 4")
    b.group(20, 95, 420, 330, "Dense feed-forward block", "grey")
    b.card(60, 170, 340, 110, "one big feed-forward network", ["every token uses all of it", "cost per token grows with size"], "blue", size=13)
    raw_text(b, 230, 330, "stored = used", 18, INK, weight="700")
    raw_text(b, 230, 358, "memory and compute grow together", 13, FAINT)
    b.group(460, 95, 760, 330, "Mixture-of-experts feed-forward block (8 experts, top 2)", "orange")
    b.card(480, 235, 120, 70, "token", ["vector"], "grey", size=12)
    b.card(630, 235, 120, 70, "router", ["scores each", "expert"], "yellow", size=12)
    b.arrow((600, 270), (630, 270), color=INK)
    names = ["expert 1", "expert 2", "expert 3", "expert 4", "expert 5", "expert 6", "expert 7", "expert 8"]
    chosen = {1, 5}
    for i, nme in enumerate(names):
        row, col = divmod(i, 2)
        y = 150 + i * 32
        color = "green" if i in chosen else "grey"
        b.card(800, y, 190, 26, nme + (" (chosen)" if i in chosen else ""), [], color, size=11, title_size=11)
        if i in chosen:
            b.arrow((750, 270), (800, y + 13), color="#2f9e44", width=2.4)
    b.card(1040, 235, 160, 70, "weighted sum", ["of the 2 outputs"], "purple", size=12)
    b.arrow((990, 195), (1040, 255), color=INK)
    b.arrow((990, 323), (1040, 285), color=INK)
    raw_text(b, 840, 415, "stored: 8 experts      used per token: 2", 13, INK, weight="700")
    b.card(20, 445, 1200, 150, "what this buys, in Mixtral 8x7B's own numbers (block 4)", ["46.70 billion parameters are stored, 45.10 billion of them in experts", "each token touches 2 of 8 experts: 12.88 billion parameters, 27.6 per cent", "a model with the knowledge capacity of the big number and roughly the arithmetic cost of the small one", "the price: all 46.70 billion must still sit in GPU memory"], "yellow", size=13)
    return b


@board("mixture-of-experts-worked")
def moe_worked():
    b = Board(1240, 640, "Routing by hand: 8 tokens, 4 experts, top 1", "The numbers printed by block 1")
    probs = [[0.7, 0.1, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1], [0.5, 0.3, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1], [0.4, 0.2, 0.3, 0.1], [0.2, 0.5, 0.2, 0.1], [0.2, 0.4, 0.3, 0.1], [0.1, 0.2, 0.6, 0.1]]
    choice = [0, 0, 0, 0, 0, 1, 1, 2]
    rows = [["token", "A", "B", "C", "D", "goes to"]]
    for t, (pr, c) in enumerate(zip(probs, choice)):
        rows.append([str(t)] + [f"{v:.1f}" for v in pr] + ["ABCD"[c]])
    rows.append(["mean P", "0.3875", "0.275", "0.2375", "0.1", ""])
    b.group(20, 95, 520, 515, "1. Router probabilities and each token's top expert", "blue")
    b.table(40, 140, [90, 80, 80, 80, 80, 90], rows, "blue", size=13, row_h=36)
    b.group(560, 95, 660, 250, "2. How uneven is it?", "orange")
    b.card(580, 140, 620, 70, "share of tokens f", ["A 5/8 = 0.625   B 2/8 = 0.25   C 1/8 = 0.125   D 0"], "orange", size=13)
    b.card(580, 225, 620, 100, "balance term = N x sum(f x P)", ["4 x (0.625x0.3875 + 0.25x0.275 + 0.125x0.2375 + 0)", "= 4 x 0.340625 = 1.3625", "1.0 would be perfectly even"], "red", size=13)
    b.group(560, 360, 660, 250, "3. Capacity = capacity factor x tokens / experts", "purple")
    crows = [["capacity factor", "room per expert", "tokens dropped", "slots used"], ["1.0", "2", "3 of 8", "5 of 8"], ["1.5", "3", "2 of 8", "6 of 12"], ["2.5", "5", "0 of 8", "8 of 20"]]
    b.table(580, 405, [170, 160, 150, 140], crows, "purple", size=13, row_h=36)
    raw_text(b, 890, 585, "safe capacity wastes slots; tight capacity drops tokens", 13, INK, weight="700")
    return b


@board("mixture-of-experts-models")
def moe_models():
    b = Board(1240, 620, "Stored against used: current and recent mixture-of-experts models", "Billions of parameters on a log axis. Model cards and papers named in the chapter, checked 7 and 8 October 2026; counting conventions differ a little")
    data = [
        ("Mixtral 8x7B (2023)", 46.7, 12.9),
        ("gpt-oss-120b (2025)", 117.0, 5.1),
        ("Qwen3-30B-A3B", 30.5, 3.3),
        ("Qwen3.6-35B-A3B (2026)", 35.0, 3.0),
        ("Qwen3.8-Flash-Next (2026)", 125.0, 6.0),
        ("Kolibri-1 (Oct 2026)", 78.1, 3.46),
        ("DeepSeek-V3 (2024)", 671.0, 37.0),
        ("DeepSeek-V4-Pro (2026)", 1600.0, 49.0),
        ("Kimi K3 (2026)", 2800.0, 104.0),
    ]
    import math
    x0, x1 = 330, 1040
    lo, hi = math.log10(2.0), math.log10(4000.0)

    def px(v):
        return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)

    b.group(20, 95, 1200, 500, "Blue: total parameters. Orange: parameters activated per token", "teal")
    for tick in (10, 100, 1000):
        line(b, px(tick), 135, px(tick), 565, "#ced4da", 1.0, "4 4")
        raw_text(b, px(tick), 583, f"{tick:,}B", 12, FAINT)
    for i, (name, total, active) in enumerate(data):
        y = 146 + i * 46
        raw_text(b, x0 - 12, y + 20, name, 13, INK, anchor="end", weight="700")
        rect(b, x0, y, px(total) - x0, 14, PALETTE["blue"]["fill"], PALETTE["blue"]["stroke"], 1.5, 1.0, 3)
        rect(b, x0, y + 18, px(active) - x0, 14, PALETTE["orange"]["fill"], PALETTE["orange"]["stroke"], 1.5, 1.0, 3)
        raw_text(b, px(total) + 8, y + 12, f"{total:,g}B total", 11, PALETTE["blue"]["text"], anchor="start", weight="700")
        raw_text(b, px(active) + 8, y + 30, f"{active:g}B active = {active / total:.1%}", 11, PALETTE["orange"]["text"], anchor="start", weight="700")
    return b


@board("mixture-of-experts-all-to-all")
def moe_all_to_all():
    b = Board(1240, 600, "Expert parallelism: send each token to the GPU that holds its expert", "Two ranks, four experts, top 2, 16 tokens per rank. Counts are printed by block 5 for a router that favours expert 0")
    for r, (x, experts) in enumerate(((60, "experts 0 and 1"), (700, "experts 2 and 3"))):
        b.group(x, 95, 480, 150, f"rank {r}: holds {experts}", "blue" if r == 0 else "purple")
        b.card(x + 20, 140, 440, 85, "its 16 tokens", ["each is sent to the rank of each of its 2 experts", "(32 assignments per rank)"], "blue" if r == 0 else "purple", size=12)
    b.group(60, 270, 1120, 150, "1. Dispatch: all-to-all", "orange")
    b.card(90, 315, 500, 80, "rank 0 sends 21 assignments to itself and 11 to rank 1", ["rank 1 sends 19 to rank 0 and 13 to itself"], "orange", size=12)
    b.card(640, 315, 510, 80, "experts compute where their weights live", ["rank 0 receives 40, rank 1 receives 24", "the busy rank sets the speed of the step"], "red", size=12)
    b.group(60, 435, 1120, 150, "2. Combine: all-to-all back, then weight and add", "green")
    b.card(90, 480, 500, 80, "outputs travel back along the same routes", ["counts are swapped: what was sent is now received"], "green", size=12)
    b.card(640, 480, 510, 80, "result equals one process running every expert", ["maximum error 0.00e+00 in this run", "with an even router: 27 and 37 received"], "yellow", size=12)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or name in v)
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
