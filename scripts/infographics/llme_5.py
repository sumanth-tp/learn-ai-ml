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
    b = Board(1240, 700, "Llama 3 405B: 16,384 GPUs and a 78.6 GB budget", "Layout from the paper; memory is the chapter's own arithmetic (block 2)")
    b.group(20, 95, 560, 330, "How the GPUs are carved up", "blue")
    b.card(40, 140, 520, 70, "128 data-parallel replicas", ["each sees a different slice of the batch"], "blue", size=12)
    b.card(60, 225, 480, 70, "16 pipeline stages per replica", ["126 layers, about 8 per stage"], "purple", size=12)
    b.card(80, 310, 440, 70, "8 tensor-parallel GPUs per stage", ["one server, NVLink"], "orange", size=12)
    raw_text(b, 300, 405, "8 x 16 x 128 = 16,384 GPUs   (context parallel = 1)", 13, INK, weight="700")

    b.group(600, 95, 620, 330, "Model states per GPU, 405.8B parameters", "teal")
    rows = [["ZeRO stage", "weights", "grads", "optimiser", "total GB"],
            ["0", "6.34", "6.34", "38.05", "50.73"],
            ["1", "6.34", "6.34", "0.30", "12.98"],
            ["2", "6.34", "0.05", "0.30", "6.69"],
            ["3", "0.85", "0.05", "0.30", "1.19"]]
    b.table(620, 140, [110, 110, 110, 130, 130], rows, "teal", size=13, row_h=34)
    raw_text(b, 910, 365, "bf16 weights and grads, fp32 master weights, m and v", 12, FAINT)
    raw_text(b, 910, 388, "Llama 3 shards optimiser and gradients, keeps weights: stage 2", 12, FAINT)

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


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or name in v)
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
