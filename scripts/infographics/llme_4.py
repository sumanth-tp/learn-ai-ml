"""Infographics for docs/llm-engineering/02-inference-and-serving, chapters 06 to 09.

Run from the repo root:

    python3 scripts/infographics/llme_4.py             # all boards
    python3 scripts/infographics/llme_4.py matrix      # boards whose name ends with the argument
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


def rect(b, x, y, w, h, fill, stroke, width=2, rx=6, opacity=1.0):
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}" fill-opacity="{opacity}"/>'
    )


@board("serving-engines-benchmark")
def engines_benchmark():
    b = Board(1200, 640, "A server that takes one request at a time",
              "OpenAI-compatible FastAPI server around SmolLM2-135M-Instruct, CPU, five runs of the benchmark client")
    b.group(20, 95, 520, 400, "The server and its client", "blue")
    c1 = b.card(45, 140, 220, 95, "benchmark client", ["1, 2 or 4 threads", "streams", "/v1/chat/completions"], "blue", size=12)
    c2 = b.card(295, 140, 225, 95, "FastAPI server", ["one SSE chunk per token", "usage in the final reply"], "purple", size=12)
    c3 = b.card(295, 260, 225, 70, "one lock", ["a single generation", "at a time"], "red", size=12)
    c4 = b.card(295, 370, 225, 70, "SmolLM2-135M", ["float32, 4 CPU threads", "24 new tokens per request"], "orange", size=12)
    b.arrow(c1.right(), c2.left())
    b.arrow(c2.bottom(), c3.top())
    b.arrow(c3.bottom(), c4.top())
    b.card(45, 260, 220, 180, "What is measured", ["TTFT: request sent to", "first token received", "TPOT: gap between tokens", "throughput: tokens per", "second, all clients"], "grey", size=12, align="left")

    b.group(560, 95, 620, 400, "Ranges across five runs", "teal")
    rows = [["clients", "mean TTFT s", "max TTFT s", "TPOT ms", "tokens/s"],
            ["1", "0.03 to 0.06", "0.03 to 0.06", "10.8 to 19.7", "48 to 86"],
            ["2", "0.18 to 0.32", "0.34 to 0.58", "11.7 to 19.0", "48 to 80"],
            ["4", "0.47 to 0.75", "0.89 to 1.50", "11.1 to 17.7", "53 to 84"]]
    b.table(575, 140, [80, 130, 130, 140, 120], rows, "teal", size=12, row_h=44)
    b.text(870, 340, "within each run, throughput at 4 clients is within about 11% of 1 client", 11, "teal", italic=True)
    b.text(870, 360, "TTFT at 4 clients is roughly 12 to 18 times the 1-client value", 11, "teal", italic=True)
    b.card(580, 385, 580, 90, "reading it", ["requests queue behind the lock: TTFT grows, TPOT does not,", "throughput stays flat. Continuous batching changes the third fact."], "yellow", size=12)

    b.card(20, 520, 1160, 100, "What a batching engine changes", ["the same OpenAI-compatible protocol, but several requests share each decode step,", "so throughput rises with concurrency instead of waiting time rising"], "green", size=13)
    return b


@board("serving-engines-matrix")
def engines_matrix():
    b = Board(1240, 700, "The feature matrix, and how a chooser reads it",
              "Checked against each engine's own documentation, 2 October 2026; a question mark is a gap in what I verified")
    cols = ["API", "JSON", "LoRA", "spec", "prefix", "multi-GPU"]
    data = [
        ("vLLM", ["Y"] * 6),
        ("SGLang", ["Y"] * 6),
        ("TGI", ["Y"] * 6),
        ("TensorRT-LLM", ["Y"] * 6),
        ("llama.cpp", ["Y", "Y", "Y", "Y", "Y", "?"]),
        ("Ollama", ["Y", "Y", "?", "?", "?", "?"]),
        ("Triton", ["?", "~", "~", "~", "~", "~"]),
    ]
    x0, y0, cw, rh = 40, 150, 78, 44
    b.group(20, 95, 760, 410, "Serving-engine feature matrix", "blue")
    for i, c in enumerate(cols):
        raw_text(b, x0 + 150 + i * cw + cw / 2, y0 - 12, c, 12, PALETTE["blue"]["text"], weight="700")
    colour = {"Y": ("green", "yes"), "?": ("grey", "not established"), "~": ("yellow", "depends on backend"), "N": ("red", "no")}
    for r, (name, cells) in enumerate(data):
        y = y0 + r * rh
        raw_text(b, x0 + 140, y + rh / 2 + 4, name, 13, INK, anchor="end", weight="700")
        for i, cell in enumerate(cells):
            col = colour[cell][0]
            rect(b, x0 + 150 + i * cw + 4, y + 4, cw - 8, rh - 8, PALETTE[col]["fill"], PALETTE[col]["stroke"], 1.6, 6)
            sym = {"Y": "✓", "?": "?", "~": "~"}[cell]
            raw_text(b, x0 + 150 + i * cw + cw / 2, y + rh / 2 + 6, sym, 17, PALETTE[col]["text"], weight="700")
    raw_text(b, x0 + 150 + 3 * cw, y0 + 7 * rh + 28, "TGI: maintenance mode  |  TensorRT-LLM: NVIDIA GPUs only  |  Triton: hosts other engines", 11, FAINT)

    b.group(800, 95, 420, 410, "The chooser's rule", "purple")
    b.card(820, 140, 380, 70, "1. state hardware and needs", ["NVIDIA, AMD, Apple, CPU or Intel GPU;", "any of the six feature columns"], "purple", size=12)
    b.card(820, 230, 380, 70, "2. fits", ["every needed cell is a yes", "(maintenance mode can be excluded)"], "green", size=12)
    b.card(820, 320, 380, 70, "3. unverified", ["no cell is a no, but one is not", "established or depends on backend"], "yellow", size=12)
    b.card(820, 410, 380, 80, "4. out", ["a needed cell is documented as", "unsupported, or the project is in", "maintenance mode and excluded"], "red", size=12)

    b.card(20, 525, 600, 150, "Default case: NVIDIA, all six needs", ["fits 3: vLLM, SGLang, TensorRT-LLM", "unverified 3: llama.cpp, Ollama, Triton", "out 1: TGI (maintenance mode)", "the chooser never ranks: fits is not best"], "green", size=13, align="left")
    b.card(640, 525, 580, 150, "Then measure", ["pick two engines that fit, serve your own model,", "and run the benchmark client from the chapter:", "TTFT, TPOT and throughput at several", "concurrency levels, warm and cold"], "orange", size=13, align="left")
    return b


@board("onnx-compression-and-pruning-pipeline")
def onnx_pipeline():
    b = Board(1200, 620, "From a PyTorch model to a runtime session",
              "Four-layer transformer encoder, d_model 256, exported with torch.onnx.export and run with ONNX Runtime 1.30.0")
    b.group(20, 95, 1160, 215, "Export, optimise, run", "blue")
    s1 = b.card(45, 145, 200, 120, "PyTorch module", ["nn.TransformerEncoder", "4 layers, eval mode", "dynamic batch and seq"], "blue", size=12)
    s2 = b.card(290, 145, 200, 120, "torch.onnx.export", ["dynamo=True,", "dynamic_shapes", "197 nodes, 15 op types"], "purple", size=12)
    s3 = b.card(535, 145, 200, 120, "graph optimisation", ["basic, extended, layout", "197 nodes become 181", "new: FusedMatMul,", "SkipLayerNormalization,", "Split"], "orange", size=11)
    s4 = b.card(770, 145, 190, 120, "InferenceSession", ["providers in priority", "order (docs example:", "CUDA, then CPU);", "this run: CPU"], "green", size=12)
    s5 = b.card(995, 145, 165, 120, "parity check", ["max difference", "vs torch about", "3e-06 on 3 shapes"], "teal", size=12)
    for a, c in ((s1, s2), (s2, s3), (s3, s4), (s4, s5)):
        b.arrow(a.right(), c.left())

    b.group(20, 330, 570, 270, "The trap", "red")
    b.card(40, 375, 530, 70, "export from an example of batch 1", ["declared dynamic, but the graph failed at batch 4", "with an InvalidArgument error from ONNX Runtime"], "red", size=12)
    b.card(40, 465, 530, 70, "export from an example of batch 2", ["runs at batches 1, 4 and 8, and at sequence", "lengths 16, 48 and 100"], "green", size=12)
    b.text(305, 565, "observed on torch 2.14.1 and onnxruntime 1.30.0, not from the documentation:", 11, FAINT, italic=True)
    b.text(305, 582, "test dynamic axes at more than one size before shipping", 11, FAINT, italic=True)

    b.group(610, 330, 570, 270, "The timing, honestly", "yellow")
    rows = [["best of 7 rounds, 5 runs", "ms"],
            ["ORT, all optimisations", "3.8 to 4.7"],
            ["ORT, no optimisation", "3.9 to 4.2"],
            ["torch eager, 3 runs", "about 2.1"],
            ["torch eager, 2 runs", "7.5 to 7.9"]]
    b.table(630, 375, [330, 200], rows, "orange", size=12, row_h=32)
    b.text(895, 560, "ORT was steady; eager flipped between two speeds on a shared laptop", 11, "orange", italic=True)
    b.text(895, 578, "no claim that either is faster", 11, "orange", italic=True)
    return b


@board("onnx-compression-and-pruning-sparsity")
def onnx_sparsity():
    b = Board(1240, 700, "What the zeros bought",
              "Digits MLP 64-512-512-10, 540 test images, dense accuracy 0.9759; one image is 0.0019")
    b.group(20, 95, 640, 330, "Global magnitude pruning, accuracy", "blue")
    rows = [["sparsity", "no fine-tune", "after 8 epochs", "zeros per layer"],
            ["0.50", "0.9759", "0.9778", "0.23 / 0.54 / 0.44"],
            ["0.80", "0.9648", "0.9815", "0.45 / 0.85 / 0.72"],
            ["0.90", "0.7685", "0.9759", "0.61 / 0.94 / 0.87"],
            ["0.95", "0.3778", "0.9481", "0.74 / 0.98 / 0.95"],
            ["0.98", "0.3148", "0.2889", "0.88 / 0.99 / 0.99"]]
    b.table(40, 140, [100, 150, 160, 200], rows, "blue", size=12, row_h=40)
    b.card(40, 385, 600, 30, "", ["98% broke it: the middle and last layers lost 99% of their weights"], "red", size=11)

    b.group(680, 95, 540, 330, "Other ways to cut", "purple")
    b.card(700, 140, 500, 65, "2:4 pattern, no fine-tuning", ["accuracy 0.9778, exactly half the weights zero", "its speedup needs Ampere-class sparse tensor cores (not run here)"], "purple", size=12)
    b.card(700, 220, 500, 65, "structured: half the first-layer units", ["0.9407 before fine-tuning, 0.9796 after,", "and the layer is physically smaller"], "green", size=12)
    b.card(700, 300, 500, 55, "simulated int8 / int4 weights", ["0.9759 and 0.9778, within one image of dense"], "teal", size=12)
    b.card(700, 370, 500, 45, "low-rank, middle layer", ["rank 16 keeps 0.9759 with 16,384 of 262,144 parameters"], "orange", size=11)

    b.group(20, 445, 1200, 235, "32 x 2048 by 2048 x 2048 matmul, one run, six runs gave these ratios", "yellow")
    items = [("dense", 0.39, "baseline", "green"), ("90% zeros in a dense tensor", 0.47, "0.94 to 1.2 times dense", "blue"),
             ("90% zeros as CSR", 14.75, "about 20 to 45 times slower", "red"), ("half the rows removed", 0.19, "about 0.4 to 0.6 times dense", "green")]
    import math
    scale = lambda ms: 40 + (math.log10(ms * 20) / math.log10(14.75 * 20)) * 440
    for i, (label, ms, note, col) in enumerate(items):
        y = 500 + i * 42
        raw_text(b, 360, y + 14, label, 12, INK, anchor="end", weight="700")
        rect(b, 380, y, scale(ms) - 40, 20, PALETTE[col]["fill"], PALETTE[col]["stroke"], 1.6, 4)
        raw_text(b, 380 + scale(ms) - 40 + 10, y + 15, f"{ms:.2f} ms  ({note})", 12, PALETTE[col]["text"], anchor="start", weight="700")
    b.text(620, 668, "zeros in a dense tensor still get multiplied; only a smaller matrix or a real sparse kernel is faster. Bars are log-scaled.", 11, FAINT, italic=True)
    return b


@board("semantic-caching-routing-and-cost-levers")
def cost_levers():
    b = Board(1240, 720, "Two cost levers, measured",
              "Prefix cache arithmetic with the Anthropic multipliers, and a semantic cache on 48 questions with MiniLM")
    b.group(20, 95, 600, 440, "Prompt cache: units of one uncached prefix", "blue")
    rows = [["requests", "uncached", "5-min cache", "1-hour cache"],
            ["1", "1.00", "1.25", "2.00"],
            ["2", "2.00", "1.35", "2.10"],
            ["5", "5.00", "1.65", "2.40"],
            ["10", "10.00", "2.15", "2.90"],
            ["50", "50.00", "6.15", "6.90"]]
    b.table(40, 140, [110, 130, 160, 170], rows, "blue", size=12, row_h=34)
    b.text(320, 365, "write 1.25 (5 min) or 2.0 (1 hour), read 0.1: break-even at request 2 and 3", 11, "blue", italic=True)
    rows2 = [["mean gap", "hit rate", "cost, 5 min TTL", "1 h TTL"],
             ["2 min", "0.92", "0.19", "0.10"],
             ["15 min", "0.29", "0.92", "0.14"],
             ["60 min", "0.08", "1.16", "0.80"],
             ["240 min", "0.02", "1.23", "1.58"]]
    b.table(40, 385, [110, 130, 170, 160], rows2, "orange", size=12, row_h=28)

    b.group(640, 95, 580, 440, "Semantic cache: threshold against hits and wrong answers", "purple")
    rows3 = [["threshold", "hit rate", "wrong share", "cost if wrong = 10"],
             ["0.55", "0.7548", "0.280", "2.380"],
             ["0.65", "0.6278", "0.206", "1.688"],
             ["0.75", "0.4430", "0.219", "1.549"],
             ["0.80", "0.3284", "0.166", "1.238"],
             ["0.85", "0.2235", "0.274", "1.408"],
             ["0.90", "0.0896", "0.233", "1.139"],
             ["0.95", "0.0000", "0.000", "1.020"]]
    b.table(660, 140, [110, 120, 140, 180], rows3, "purple", size=12, row_h=34)
    b.text(930, 430, "no cache costs 1.000; the best threshold depends on what a wrong answer costs:", 11, "purple", italic=True)
    b.text(930, 450, "wrong = 1: 0.55 at 0.477  |  2: 0.65 at 0.651  |  5: 0.80 at 0.965", 11, "purple", italic=True)
    b.text(930, 470, "wrong = 10 or 50: 0.95 at 1.020, which serves nothing from the cache", 11, "purple", italic=True)

    b.card(20, 555, 600, 145, "Sparse traffic breaks a prefix cache", ["at a 60-minute mean gap the 5-minute cache hits 8% of the time", "and costs 1.16 per request: more than not caching", "match the TTL to your inter-arrival time"], "red", size=12)
    b.card(640, 555, 580, 145, "Similarity is not equality", ["different intents reached cosine 0.903 (opens vs closes)", "the same intent can fall to 0.321", "a threshold cannot separate close wording from the answer"], "orange", size=12)
    return b


@board("semantic-caching-routing-and-cost-cascade")
def cost_cascade():
    b = Board(1200, 660, "A cascade and a router in a seeded simulation",
              "20,000 requests; cheap call 1, large call 15, router call 0.1 (relative units, not prices)")
    b.group(20, 95, 560, 270, "Cascade: ask the cheap model first", "blue")
    a = b.card(40, 145, 150, 70, "request", ["20,000 of them"], "blue", size=12)
    s = b.card(220, 145, 150, 70, "small model", ["cost 1.00", "accuracy 0.734"], "green", size=12)
    d = b.diamond(465, 180, 150, 90, "confidence\n>= t ?", "yellow", size=12)
    l = b.card(390, 270, 170, 70, "large model", ["cost 15.00", "accuracy 0.945"], "orange", size=12)
    b.arrow(a.right(), s.left())
    b.arrow(s.right(), (390, 180))
    b.arrow((465, 225), (465, 270), label="no: escalate")
    b.text(210, 300, "yes: keep the small answer", 12, "green", weight="700")
    b.text(210, 322, "oracle (large only when small is wrong):", 11, FAINT, italic=True)
    b.text(210, 340, "accuracy 0.966 at cost 4.99", 11, FAINT, italic=True)

    b.group(600, 95, 580, 270, "Cascade, by confidence threshold t", "teal")
    rows = [["t", "escalated", "accuracy", "cost"],
            ["0.2", "0.171", "0.877", "3.56"],
            ["0.4", "0.258", "0.954", "4.87"],
            ["0.5", "0.288", "0.961", "5.33"],
            ["0.6", "0.353", "0.957", "6.30"],
            ["0.7", "0.482", "0.953", "8.23"],
            ["0.8", "0.656", "0.949", "10.85"]]
    b.table(620, 140, [100, 150, 150, 150], rows, "teal", size=12, row_h=30)

    b.group(20, 385, 560, 255, "Router: guess difficulty first", "purple")
    rows2 = [["d", "to large", "accuracy", "cost"],
             ["0.3", "0.758", "0.935", "11.71"],
             ["0.4", "0.638", "0.922", "10.04"],
             ["0.5", "0.505", "0.901", "8.17"],
             ["0.6", "0.368", "0.872", "6.25"],
             ["0.7", "0.245", "0.836", "4.52"]]
    b.table(40, 430, [100, 140, 140, 140], rows2, "purple", size=12, row_h=34)

    b.card(600, 385, 580, 120, "The cascade wins by construction", ["its confidence score is built to carry information", "about whether the small answer was right: 0.961 at 5.33", "against the large model alone at 0.945 and 15.00"], "yellow", size=12)
    b.card(600, 520, 580, 120, "Real confidence is weaker", ["log probabilities, a verifier or agreement between samples", "are noisier: measure escalation rate, accuracy and cost", "on labelled traffic before trusting the saving"], "red", size=12)
    return b


@board("gpu-sizing-and-capacity-planning-formula")
def capacity_formula():
    b = Board(1240, 700, "From a model and a traffic target to a GPU count",
              "Llama 3.1 8B in 16-bit on one H100 SXM 80 GB: 20 requests/s, 1,000 + 300 tokens, 40 ms TPOT target")
    steps = [
        ("1. memory pool", ["0.90 x 80 GB - 2 GB reserve", "= 70 GB", "assumptions: 90% use, 2 GB"], "blue"),
        ("2. KV budget", ["70 GB - 16.1 GB of weights", "= 53.9 GB", "131.1 kB per token"], "purple"),
        ("3. sequences", ["53.9 GB / (131.1 kB x 1,300)", "= 316 by memory", "316 within the 40 ms target"], "green"),
        ("4. decode step", ["(weights + batch x context x KV)", "/ (3.35 TB/s x 0.6)", "= 31.7 ms at batch 316"], "orange"),
        ("5. Little's law", ["L = lambda x W", "20 x 9.55 s = 190.9 in flight", "1 decode replica"], "teal"),
        ("6. prefill", ["20 x 1,000 tokens/s needs", "3 replicas at MFU 0.4,", "30% of time on prefill"], "red"),
    ]
    cards = []
    for i, (title, lines, col) in enumerate(steps):
        x = 30 + (i % 3) * 405
        y = 120 + (i // 3) * 190
        cards.append(b.card(x, y, 370, 150, title, lines, col, size=13))
    for i in (0, 1, 3, 4):
        b.arrow(cards[i].right(), cards[i + 1].left())
    b.arrow(cards[2].bottom(), cards[3].top(), via=[(cards[2].cx, 300), (cards[3].cx, 300)])
    b.card(30, 500, 560, 170, "The answer", ["replicas = max(decode 1, prefill 3) = 3", "GPUs = 3 x tensor parallel 1 = 3", "GPU-hours per million output tokens:", "3 / (20 x 300 x 3600) x 1,000,000 = 0.139"], "green", size=13, align="left")
    b.card(620, 500, 590, 170, "Assumptions to replace by measuring", ["bandwidth efficiency 0.6, prefill MFU 0.4,", "prefill share 0.3, reserve 2 GB per GPU", "dense TFLOPS: 989.5, derived from the", "datasheet's with-sparsity figure 1,979"], "yellow", size=13, align="left")
    return b


@board("gpu-sizing-and-capacity-planning-example")
def capacity_example():
    b = Board(1240, 700, "The worked example across GPUs and models",
              "20 requests/s, 1,000 prompt + 300 output tokens, 40 ms TPOT target; datasheet figures fetched 2 October 2026")
    rows = [["case", "weights GB", "KV budget GB", "seqs by memory", "seqs in SLO", "step ms", "decode / prefill replicas", "GPUs", "GPU-h per M tokens"],
            ["8B on L4 24GB", "16.1", "3.5", "-", "-", "89.2 weights only", "TPOT target unreachable", "-", "-"],
            ["8B on A100 80GB", "16.1", "53.9", "316", "218", "40.0", "2 / 9", "9", "0.417"],
            ["8B on H100 SXM", "16.1", "53.9", "316", "316", "31.7", "1 / 3", "3", "0.139"],
            ["8B on H100 NVL", "16.1", "66.5", "390", "390", "32.0", "1 / 4", "4", "0.185"],
            ["70B on 2 H100", "141.1", "n/a", "n/a", "n/a", "n/a", "weights do not fit", "-", "-"],
            ["70B on 4 H100", "141.1", "138.9", "326", "326", "32.8", "1 / 6", "24", "1.111"],
            ["70B on 8 H100", "141.1", "418.9", "983", "983", "31.8", "1 / 3", "24", "1.111"]]
    b.table(20, 100, [190, 100, 120, 130, 110, 120, 170, 70, 150], rows, "blue", size=12, row_h=40)
    b.card(20, 445, 390, 125, "Capacity said yes, bandwidth said no", ["the L4 holds the 8B weights, but at 300 GB/s", "one step takes about 89 ms, over the 40 ms target"], "red", size=12)
    b.card(425, 445, 390, 125, "Prefill set the GPU count", ["H100 SXM: decode needs 1 replica, prefill 3.", "Longer prompts: 200 tokens 1 GPU,", "1,000 tokens 3, 4,000 tokens 11"], "orange", size=12)
    b.card(830, 445, 390, 125, "The assumptions move the answer", ["prefill share and MFU alone swing the", "8B plan between 2 and 5 GPUs:", "measure prefill and decode first"], "yellow", size=12)
    b.card(20, 590, 1200, 90, "Not in the model", ["tensor-parallel communication, peak-versus-mean traffic, headroom for a failed replica and idle capacity:", "the real cost per token is higher than this formula says"], "grey", size=12)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or v.endswith(name))
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
