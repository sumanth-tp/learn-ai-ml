"""Infographics for docs/llm-engineering/01-adapting-models.

Run from the repo root:

    python3 scripts/infographics/llme_1.py            # all boards
    python3 scripts/infographics/llme_1.py loss-mask  # just one
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


def rect(b, x, y, w, h, fill, stroke, width=1.6, rx=4):
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


@board("prompt-retrieve-or-fine-tune-decision-tree")
def decision_tree():
    b = Board(1240, 740, "Diagnose first, then pick the cheapest lever", "The thresholds are the ones in the chapter's decision function")
    top = b.card(330, 95, 580, 54, "Always first: a prompt and a held-out eval set", ["measure before and after every lever"], "grey", size=12)
    root = b.card(480, 175, 280, 50, "What is actually failing?", [], "dark", size=13)
    b.arrow(top.bottom(), root.top())

    cols = [
        (30, "red", "knowledge gap", ["the model never saw the facts"], "retrieval (RAG)", ["answer from your documents"]),
        (330, "orange", "behaviour or format gap", ["knows enough, answers wrongly"], "few-shot examples in the prompt", ["cheapest fix, costs tokens per request"]),
        (630, "purple", "reasoning gap", ["facts and format are right,", "multi-step answers are wrong"], "stronger model or decomposition", ["split into steps, use tools"]),
        (930, "teal", "latency or cost", ["answers are right, the bill is not"], "shorter prompt and caching", ["try before tuning"]),
    ]
    w = 280
    boxes = []
    for x, color, title, lines, lever, lever_lines in cols:
        g = b.card(x, 270, w, 86, title, lines, color, size=12)
        l = b.card(x, 385, w, 86, lever, lever_lines, color, size=12)
        b.arrow(root.bottom(), g.top(), color=color)
        b.arrow(g.bottom(), l.top())
        boxes.append((x, color, l))

    x, color, l = boxes[0]
    b.card(x, 500, w, 80, "facts change often?", ["keep the index fresh;", "do not train facts in"], "red", size=12, dashed=True)
    b.arrow(l.bottom(), (x + w / 2, 500))

    x, color, l = boxes[1]
    d = b.card(x, 500, w, 56, "500 or more labelled examples?", [], "yellow", size=12)
    b.arrow(l.bottom(), d.top())
    y1 = b.card(x, 600, 132, 76, "yes", ["SFT with LoRA"], "green", size=12)
    n1 = b.card(x + 148, 600, 132, 76, "no", ["collect more", "examples first"], "grey", size=12)
    b.arrow(d.bottom(0.25), y1.top(), color="green")
    b.arrow(d.bottom(0.75), n1.top(), color="grey")

    x, color, l = boxes[2]
    d = b.card(x, 500, w, 56, "5,000 or more verified traces?", [], "yellow", size=12)
    b.arrow(l.bottom(), d.top())
    y2 = b.card(x, 600, 132, 76, "yes", ["tune on the", "verified traces"], "green", size=12)
    n2 = b.card(x + 148, 600, 132, 76, "no", ["stay with the", "stronger model"], "grey", size=12)
    b.arrow(d.bottom(0.25), y2.top(), color="green")
    b.arrow(d.bottom(0.75), n2.top(), color="grey")

    x, color, l = boxes[3]
    d = b.card(x, 500, w, 70, "volume at least 130,783 a month, and 500+ examples?", [], "yellow", size=12)
    b.arrow(l.bottom(), d.top())
    y3 = b.card(x, 610, 132, 70, "yes", ["tune a smaller", "model"], "green", size=12)
    n3 = b.card(x + 148, 610, 132, 70, "no", ["stay on the", "large model"], "grey", size=12)
    b.arrow(d.bottom(0.25), y3.top(), color="green")
    b.arrow(d.bottom(0.75), n3.top(), color="grey")
    return b


@board("prompt-retrieve-or-fine-tune-break-even")
def break_even():
    b = Board(1240, 640, "Where fine-tuning starts to pay", "Synthetic prices: a break-even is arithmetic, not a quote from any provider")
    b.group(20, 95, 700, 300, "Cost units per month at three volumes", "teal")
    rows = [
        ["requests / month", "long prompt", "long, 90% cached", "RAG", "tuned small"],
        ["10,000", "93", "20", "633", "1,210"],
        ["100,000", "930", "201", "930", "1,215"],
        ["1,000,000", "9,300", "2,010", "3,900", "1,258"],
    ]
    b.table(40, 140, [170, 120, 170, 100, 120], rows, "teal", size=13, row_h=40)
    b.text(380, 330, "per request: 0.009300, 0.002010, 0.003300, 0.000048", 12, FAINT)
    b.text(380, 352, "fixed per month: 0, 0, 600, 1,210", 12, FAINT)
    b.text(380, 374, "(1,210 = 3,000 / 12 training + 8 hours x 120)", 12, FAINT)

    b.group(740, 95, 480, 300, "Break-even, requests per month", "orange")
    pairs = [
        ("long prompt vs tuned small", "130,783"),
        ("long prompt, cached vs tuned", "616,718"),
        ("RAG vs tuned small", "187,577"),
        ("long prompt vs RAG", "100,000"),
    ]
    y = 140
    for label, value in pairs:
        b.card(760, y, 440, 56, value, [label], "orange", size=12, title_size=18)
        y += 64

    b.card(20, 420, 590, 100, "Why caching moves the line", ["a cached prefix is billed at 0.1 of the input price,", "so the long prompt costs less and tuning needs", "about 4.7 times the volume to pay"], "blue", size=13)
    b.card(630, 420, 590, 100, "The cost people forget", ["maintenance: 960 of the 1,210 fixed units", "are engineer time, only 250 is training"], "red", size=13)
    b.card(20, 545, 1200, 70, "Replace every price with your provider's current figure", ["multipliers differ by model and change; keep them as parameters, not constants"], "grey", size=13)
    return b


@board("preparing-data-for-fine-tuning-loss-mask")
def loss_mask():
    b = Board(1240, 660, "What the model sees and what it is graded on", "One ticket through the real SmolLM2 chat template: 52 tokens")
    x0, y0, total = 40.0, 120.0, 1160.0
    scale = total / 52
    groups = [
        (21, "system block", "grey", "<|im_start|>system ... <|im_end|>"),
        (19, "user block", "blue", "the ticket"),
        (4, "header", "grey", "assistant"),
        (8, "answer", "green", "JSON + end"),
    ]
    x = x0
    for count, title, color, sub in groups:
        w = count * scale
        c = PALETTE[color]
        rect(b, x, y0, w, 70, c["fill"], c["stroke"], 2.2)
        raw_text(b, x + w / 2, y0 + 28, title, 13, c["text"], weight="700")
        raw_text(b, x + w / 2, y0 + 48, f"{count} tokens", 12, INK)
        x += w
    xm = x0 + 44 * scale
    raw_text(b, x0 + 22 * scale, y0 - 12, "label -100: ignored by the loss", 13, FAINT, weight="700")
    raw_text(b, xm + 4 * scale, y0 - 12, "trained", 13, PALETTE["green"]["text"], weight="700")

    b.card(40, 240, 560, 150, "Answer tokens (8)", ["{\"  queue  \":  \"  account  \"}  <|im_end|>  \\n", "the end marker is trained, so generation stops;", "TRL's completion mask includes the final newline"], "green", size=12)
    b.card(620, 240, 580, 150, "Same text, two losses", ["mean over every token             4.5594", "mean over the 8 answer tokens     4.7013", "billing ticket: 4.2921 against 4.1928, the other way"], "yellow", size=12, align="left")

    b.group(40, 415, 1160, 215, "Two routes to the same mask", "purple")
    b.card(60, 460, 540, 150, "prompt-completion data", ["prompt and completion as separate columns;", "TRL builds labels for the completion only", "our hand-built mask equals TRL's for all 4 rows"], "purple", size=12)
    b.card(640, 460, 540, 150, "assistant_only_loss=True", ["stock SmolLM2 template: 0 of 45 tokens marked,", "TRL refuses with a ValueError;", "patched template with generation markers: 8 of 45"], "orange", size=12)
    return b


@board("preparing-data-for-fine-tuning-data-pipeline")
def data_pipeline():
    b = Board(1240, 700, "From raw rows to a trustworthy split", "Counts printed by blocks 4 and 5; synthetic ticket data with injected defects")
    stages = [
        ("raw rows", 105, "blue"),
        ("drop empty", 100, "blue"),
        ("outside schema", 99, "blue"),
        ("over 128 tokens", 98, "blue"),
        ("exact duplicates", 84, "orange"),
        ("conflicting labels", 83, "orange"),
        ("near duplicates", 71, "orange"),
    ]
    base_y, max_h = 330.0, 170.0
    for i, (label, n, color) in enumerate(stages):
        c = PALETTE[color]
        x = 50 + i * 160
        h = n / 105 * max_h
        rect(b, x, base_y - h, 110, h, c["fill"], c["stroke"], 2)
        raw_text(b, x + 55, base_y - h - 8, str(n), 15, c["text"], weight="700")
        raw_text(b, x + 55, base_y + 20, label, 11, INK)
    raw_text(b, 620, 118, "rows kept after each filter", 13, FAINT)

    b.group(20, 380, 600, 290, "Split by issue, not by row", "green")
    b.card(40, 425, 270, 100, "grouped split", ["45 train, 26 validation", "whole issues go to one side"], "green", size=12)
    b.card(330, 425, 270, 100, "leaking rows", ["grouped: 0 of 26", "random split: 21 of 26"], "red", size=12)
    b.card(40, 545, 560, 100, "leak = more than half the validation row's 4-grams", ["appear in training: a random split would let", "memorisation pass as generalisation"], "yellow", size=12)

    b.group(640, 380, 580, 290, "Packing, 48 examples, 512-token rows", "purple")
    rows = [["layout", "positions", "useful"],
            ["pad all to 512", "24,576", "14.5%"],
            ["pad per batch of 8", "6,320", "56.5%"],
            ["pack first-fit", "4,096", "87.2%"]]
    b.table(660, 425, [230, 160, 140], rows, "purple", size=13, row_h=40)
    b.text(930, 625, "3,570 real tokens, 8 blocks after packing", 12, FAINT)
    return b


@board("supervised-fine-tuning-with-lora-lora-params")
def lora_params():
    b = Board(1240, 680, "LoRA trains two thin matrices beside a frozen weight", "SmolLM2-135M-Instruct, 30 layers, hidden width 576")
    w = b.card(60, 130, 230, 150, "W (frozen)", ["576 x 576", "331,776 weights", "never updated"], "grey", size=12)
    a = b.card(380, 120, 230, 78, "A (trainable)", ["r x 576"], "orange", size=12)
    bb = b.card(380, 222, 230, 78, "B (trainable)", ["576 x r"], "orange", size=12)
    out = b.card(700, 130, 230, 150, "output", ["W x + (alpha / r) B A x", "alpha 16, r 8: scale 2"], "green", size=12)
    b.arrow(w.right(), out.left(), label="frozen path", label_at=0.2)
    b.arrow(a.bottom(), bb.top(), color="orange")
    b.arrow(bb.right(), out.left(0.8), color="orange", label="added")
    x_in = b.pill(130, 82, "input x", "grey", size=12, solid=True)
    b.arrow((x_in.cx, x_in.y + x_in.h), w.top(0.5))
    b.arrow((x_in.x + x_in.w, x_in.y + x_in.h / 2), a.left(), via=[(330, x_in.y + x_in.h / 2)], color="orange")
    b.card(980, 130, 240, 150, "per module, rank r", ["adds r x (in + out)", "q_proj at r 8: 9,216", "v_proj at r 8: 6,144"], "blue", size=12)

    b.group(20, 330, 700, 330, "Trainable parameters, base 134,515,008", "teal")
    rows = [["rank", "q, v", "q, k, v, o", "all linear"],
            ["1", "57,600", "115,200", "305,280"],
            ["8", "460,800", "921,600", "2,442,240"],
            ["64", "3,686,400", "7,372,800", "19,537,920"]]
    b.table(40, 375, [90, 190, 190, 190], rows, "teal", size=13, row_h=44)
    b.text(370, 590, "rank 8, all linear: 1.82% of the base", 14, "teal", weight="700")
    b.text(370, 620, "rank 8, q and v only: 0.34%", 13, "teal")

    b.group(740, 330, 480, 330, "Training state, float32 estimate", "orange")
    b.card(760, 375, 440, 80, "full fine-tuning: 2,152 MB", ["weights, gradients, Adam moments"], "red", size=12)
    b.card(760, 470, 440, 80, "LoRA rank 8: 538 MB base + 39.1 MB", ["adapter 9.8, gradients 9.8, moments 19.5"], "green", size=12)
    b.card(760, 565, 440, 70, "27% of full fine-tuning", ["activations are not counted"], "yellow", size=12)
    return b


@board("supervised-fine-tuning-with-lora-before-after")
def before_after():
    b = Board(1240, 700, "Before and after 60 LoRA steps", "23 held-out tickets from 6 issues the model never saw")
    b.group(20, 95, 640, 300, "Held-out result", "green")
    rows = [["", "prompt tokens", "valid JSON", "right queue"],
            ["base, ticket only", "47", "0 / 23", "0 / 23"],
            ["base, 6 examples", "225", "23 / 23", "13 / 23"],
            ["tuned, ticket only", "47", "23 / 23", "19 / 23"]]
    b.table(40, 140, [210, 140, 120, 120], rows, "green", size=13, row_h=44)
    b.text(340, 350, "tuned: better and a prompt 4.8 times shorter", 14, "green", weight="700")
    b.text(340, 376, "one seed, synthetic tickets, 23 cases", 12, FAINT, italic=True)

    b.group(680, 95, 540, 300, "Training loss by step", "blue")
    losses = [(10, 3.5810), (20, 0.9011), (30, 0.2312), (40, 0.0399), (50, 0.0203), (60, 0.0087)]
    base_y, max_h = 360.0, 200.0
    blue = PALETTE["blue"]
    for i, (step, loss) in enumerate(losses):
        x = 720 + i * 82
        h = max(loss / 3.581 * max_h, 3)
        rect(b, x, base_y - h, 54, h, blue["fill"], blue["stroke"], 2)
        raw_text(b, x + 27, base_y - h - 8, f"{loss:.4f}", 11, blue["text"], weight="700")
        raw_text(b, x + 27, base_y + 18, f"step {step}", 11, INK)

    b.group(20, 420, 1200, 260, "The same ticket, before and after", "orange")
    b.card(40, 465, 560, 90, "ticket", ["The price on my invoice is higher than the one I was quoted."], "grey", size=12)
    b.card(40, 575, 560, 85, "before", ["\"Hello! I'm sorry to hear that your invoice is higher\""], "red", size=12)
    b.card(640, 465, 560, 90, "after", ["{\"queue\": \"billing\"}"], "green", size=14)
    b.card(640, 575, 560, 85, "what changed", ["2,442,240 trainable parameters, a 9.8 MB adapter;", "the merged model matches adapter-on logits to 2.3e-4"], "yellow", size=12)
    return b


@board("preference-tuning-dpo-orpo-dpo-loss")
def dpo_loss_board():
    b = Board(1240, 700, "The DPO loss on one preference pair", "Toy log-probabilities from block 1; beta 0.1")
    pol = b.card(30, 110, 350, 120, "policy being trained", ["log p(chosen) = -12.0", "log p(rejected) = -20.5"], "blue", size=13)
    ref = b.card(30, 260, 350, 120, "frozen reference", ["log p(chosen) = -14.0", "log p(rejected) = -19.0"], "grey", size=13)
    rat = b.card(450, 150, 330, 190, "log-ratios", ["chosen   -12.0 - (-14.0) = +2.0", "rejected -20.5 - (-19.0) = -1.5", "gap = 3.5"], "purple", size=13, align="left")
    b.arrow(pol.right(0.5), rat.left(0.3))
    b.arrow(ref.right(0.5), rat.left(0.7))
    out = b.card(850, 110, 360, 270, "implicit rewards, beta x log-ratio", ["chosen   +0.200", "rejected -0.150", "margin    0.350", "", "loss = -log sigmoid(0.350)", "     = 0.5334", "gradient weight 0.4134"], "green", size=13, align="left")
    b.arrow(rat.right(), out.left())

    b.group(20, 420, 600, 260, "beta sets how far the policy must move", "orange")
    rows = [["beta", "margin", "loss", "gap for loss 0.1"],
            ["0.01", "0.035", "0.6758", "225.2"],
            ["0.1", "0.350", "0.5334", "22.5"],
            ["0.5", "1.750", "0.1602", "4.5"],
            ["1.0", "3.500", "0.0298", "2.3"]]
    b.table(40, 465, [100, 140, 140, 190], rows, "orange", size=13, row_h=38)
    b.group(640, 420, 580, 260, "What the numbers say", "teal")
    b.card(660, 465, 540, 90, "start of training", ["policy equals reference: margin 0, loss 0.6931 = ln 2"], "teal", size=13)
    b.card(660, 570, 540, 90, "the gradient", ["-beta x weight on the chosen log-probability,", "+beta x weight on the rejected: -0.0413 and +0.0413"], "teal", size=13)
    return b


@board("preference-tuning-dpo-orpo-methods-compare")
def methods_compare():
    b = Board(1240, 720, "Four ways to learn from preferences", "The same toy pair: chosen 8 tokens, rejected 10 tokens; beta, gamma and lambda are illustrative")
    rows = [["method", "reference model", "data it needs", "loss on the toy pair"],
            ["DPO", "yes", "chosen and rejected", "0.5334"],
            ["SimPO", "no", "chosen and rejected", "0.4375"],
            ["ORPO", "no", "chosen and rejected", "1.5415"],
            ["KTO", "yes", "one label per answer", "0.4502 and 0.4626"]]
    b.table(30, 110, [160, 220, 330, 440], rows, "blue", size=14, row_h=48)

    b.group(20, 400, 600, 290, "Reading the losses", "purple")
    b.card(40, 445, 560, 110, "DPO and KTO use the reference", ["both measure movement away from it;", "KTO needs a KL estimate, here set to 0"], "purple", size=12)
    b.card(40, 570, 560, 100, "ORPO is SFT plus a penalty", ["1.5000 SFT term + 0.1 x 0.4150 odds-ratio term", "so its scale is not comparable with the others"], "orange", size=12)
    b.group(640, 400, 580, 290, "Choosing", "green")
    b.card(660, 445, 540, 110, "have pairs and a decent SFT model", ["start with DPO: the most documented path"], "green", size=12)
    b.card(660, 570, 540, 100, "have thumbs up or down only", ["KTO takes one label per answer, no pairs"], "yellow", size=12)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or v.endswith(name))
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
