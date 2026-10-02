"""Infographics for docs/llm-engineering/02-inference-and-serving, chapters 01 to 05.

Run from the repo root:

    python3 scripts/infographics/llme_3.py             # all boards
    python3 scripts/infographics/llme_3.py roofline    # boards whose name ends with the argument
"""

import math
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


def dot(b, cx, cy, r, fill, stroke, width=2):
    b.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def rect(b, x, y, w, h, fill, stroke, width=2, rx=6, opacity=1.0):
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}" fill-opacity="{opacity}"/>'
    )


CPU_PREFILL_512 = "2,837"
CPU_DECODE_1 = "76"
CPU_RATIO = "37"
CPU_B32 = "10.7"


@board("why-decoding-is-memory-bound-roofline")
def roofline():
    b = Board(1200, 650, "Decoding reads the whole model for every token",
              "Llama 3.1 8B, bf16 weights, H100 SXM figures: 3.35 TB/s and 989.5 TFLOP/s dense")
    b.group(20, 95, 560, 345, "The roofline", "teal")
    x0, y0, w, h = 85, 140, 470, 250
    rect(b, x0, y0, w, h, "#ffffff", "#ced4da", 1.5, rx=4, opacity=0.6)
    lx = lambda v: x0 + (math.log10(v) - math.log10(0.5)) / (math.log10(2048) - math.log10(0.5)) * w
    ly = lambda v: y0 + h - (math.log10(v) - math.log10(1)) / (math.log10(2000) - math.log10(1)) * h
    for t in [1, 8, 64, 512]:
        line(b, lx(t), y0, lx(t), y0 + h, "#e9ecef", 1)
        raw_text(b, lx(t), y0 + h + 17, str(t), 11, FAINT)
    for t in [1, 10, 100, 1000]:
        line(b, x0, ly(t), x0 + w, ly(t), "#e9ecef", 1)
        raw_text(b, x0 - 8, ly(t) + 4, str(t), 11, FAINT, anchor="end")
    raw_text(b, x0 + w / 2, y0 + h + 36, "FLOP per byte of weights read (= batch, in bf16)", 11, INK)
    b.parts.append(f'<text xml:space="preserve" transform="translate(40,{y0 + h / 2}) rotate(-90)" text-anchor="middle" font-family="{MONO}" font-size="11" fill="{INK}">attainable TFLOP/s</text>')
    ridge = 989.5 / 3.35
    pts = [(0.5, 3.35 * 0.5), (ridge, 989.5), (2048, 989.5)]
    d = "M" + " L".join(f"{lx(i):.1f},{ly(max(v, 1)):.1f}" for i, v in [(0.5, 1.0)] + pts[1:])
    cy = ly(3.35)
    b.parts.append(f'<path d="M{lx(1):.1f},{ly(3.35):.1f} L{lx(ridge):.1f},{ly(989.5):.1f} L{lx(2048):.1f},{ly(989.5):.1f}" fill="none" stroke="{PALETTE["orange"]["stroke"]}" stroke-width="3"/>')
    line(b, lx(ridge), ly(989.5), lx(ridge), y0 + h, PALETTE["orange"]["stroke"], 1.5, "5 4")
    raw_text(b, lx(ridge) - 6, y0 + h - 8, "ridge 295.4", 11, PALETTE["orange"]["text"], anchor="end", weight="700")
    dot(b, lx(1), ly(3.35), 7, PALETTE["blue"]["fill"], PALETTE["blue"]["stroke"])
    raw_text(b, lx(1) + 14, ly(3.35) + 20, "batch 1: 4.79 ms", 12, PALETTE["blue"]["text"], anchor="start", weight="700")
    raw_text(b, lx(1) + 14, ly(3.35) + 35, "0.3% of peak compute", 11, INK, anchor="start")
    dot(b, lx(64), ly(3.35 * 64), 6, PALETTE["blue"]["fill"], PALETTE["blue"]["stroke"])
    raw_text(b, lx(64) + 10, ly(3.35 * 64) + 16, "batch 64: still 4.79 ms", 11, PALETTE["blue"]["text"], anchor="start", weight="700")
    dot(b, lx(512), ly(989.5), 6, PALETTE["green"]["fill"], PALETTE["green"]["stroke"])
    raw_text(b, lx(512) - 4, ly(989.5) - 12, "batch 512: 8.31 ms", 11, PALETTE["green"]["text"], anchor="end", weight="700")
    raw_text(b, x0 + 14, y0 + 20, "memory-bound", 12, PALETTE["orange"]["text"], anchor="start", weight="700")
    raw_text(b, x0 + w - 12, y0 + h - 36, "compute-bound", 12, PALETTE["green"]["text"], anchor="end", weight="700")

    b.group(600, 95, 580, 345, "One decode step", "purple")
    rows = [["batch", "step ms", "tokens/s", "compute util", "bound"],
            ["1", "4.79", "209", "0.003", "memory"],
            ["8", "4.79", "1,669", "0.027", "memory"],
            ["32", "4.79", "6,675", "0.108", "memory"],
            ["64", "4.79", "13,350", "0.217", "memory"],
            ["128", "4.79", "26,699", "0.433", "memory"],
            ["295", "4.79", "61,533", "0.999", "memory"],
            ["512", "8.31", "61,611", "1.000", "compute"]]
    b.table(620, 135, [70, 100, 110, 150, 110], rows, "purple", size=13, row_h=34)
    b.text(890, 428, "Block 1 of the chapter prints every row", 11, FAINT, italic=True)

    b.card(20, 465, 370, 165, "Why 4.79 ms", ["8.03 B parameters x 2 bytes", "= 16.06 GB of weights", "16.06 GB / 3.35 TB/s = 4.79 ms", "the same for any batch below 295"], "orange", size=12)
    b.card(415, 465, 370, 165, "Prefill is a big batch", ["prompt 16 or 128 tokens: 4.79 ms", "prompt 512 tokens: 8.31 ms", "prompt 2,048 tokens: 33.24 ms", "compute-bound beyond about 295"], "blue", size=12)
    b.card(810, 465, 370, 165, "Shrink the bytes, move the ridge", ["ridge batch by weight format:", "bf16 295   int8 148   int4 74", "fewer bytes also shortens the", "batch-1 step in proportion"], "green", size=12)
    return b


@board("why-decoding-is-memory-bound-metrics")
def metrics():
    b = Board(1200, 700, "What users feel and what operators pay for",
              "One request: 512-token prompt, 256 new tokens, Llama 3.1 8B on the chapter's H100 figures")
    b.group(20, 95, 1160, 150, "One request on a timeline", "blue")
    p = rect(b, 60, 140, 190, 60, PALETTE["orange"]["fill"], PALETTE["orange"]["stroke"])
    raw_text(b, 155, 165, "prefill, one pass", 13, PALETTE["orange"]["text"], weight="700")
    raw_text(b, 155, 184, "8.31 ms", 12, INK)
    for i in range(5):
        rect(b, 275 + i * 90, 150, 76, 40, PALETTE["green"]["fill"], PALETTE["green"]["stroke"])
        raw_text(b, 313 + i * 90, 175, "decode", 12, PALETTE["green"]["text"], weight="700")
    raw_text(b, 740, 175, "...  255 more steps", 13, INK, anchor="start")
    line(b, 60, 214, 250, 214, PALETTE["orange"]["stroke"], 2)
    raw_text(b, 155, 232, "TTFT", 12, PALETTE["orange"]["text"], weight="700")
    line(b, 275, 214, 365, 214, PALETTE["green"]["stroke"], 2)
    raw_text(b, 320, 232, "TPOT", 12, PALETTE["green"]["text"], weight="700")
    b.card(900, 135, 250, 90, "latency =", ["TTFT + TPOT x (tokens - 1)", "throughput = output tokens", "per second, all users"], "grey", size=11)

    b.group(20, 265, 640, 270, "Latency and throughput against batch", "teal")
    rows = [["batch", "TPOT ms", "latency s", "tokens/s", "vs batch 1"],
            ["1", "4.82", "1.237", "207", "1.0x"],
            ["8", "4.99", "1.282", "1,598", "7.7x"],
            ["32", "5.60", "1.435", "5,708", "27.6x"],
            ["64", "6.40", "1.639", "9,994", "48.3x"],
            ["128", "8.00", "2.048", "15,999", "77.3x"]]
    b.table(40, 310, [80, 120, 130, 130, 140], rows, "teal", size=13, row_h=34)
    b.text(340, 528, "KV cache: 131,072 bytes per token, average context 640", 11, FAINT, italic=True)

    b.group(680, 265, 500, 270, "Why not 128 times?", "red")
    b.card(700, 310, 460, 95, "each sequence brings its own cache", ["batch 128 reads 128 x 640 x 128 KiB", "= 10.7 GB extra on top of 16.06 GB", "of weights, every single step"], "red", size=12)
    b.card(700, 420, 460, 95, "the price of 77x throughput", ["a user waits 2.048 s instead of 1.237 s", "(66% longer): the cache term", "is the subject of the next chapter"], "orange", size=12)

    b.group(20, 555, 1160, 125, "The same shape on a CPU (SmolLM2-135M, fp32, one run on this machine; ratios vary run to run)", "purple")
    b.card(40, 595, 360, 70, f"prefill, 512 tokens", [f"{CPU_PREFILL_512} tokens per second"], "blue", size=12)
    b.card(420, 595, 360, 70, "decode, batch 1", [f"{CPU_DECODE_1} tokens per second ({CPU_RATIO}x slower)"], "orange", size=12)
    b.card(800, 595, 360, 70, "decode, batch 32", [f"{CPU_B32}x the batch-1 rate"], "green", size=12)
    return b


@board("kv-cache-and-paged-attention-size")
def kv_size():
    b = Board(1200, 690, "What the KV cache costs per token",
              "Sizes read from each model's own config.json, cache stored in bf16")
    b.group(20, 95, 520, 300, "The formula", "blue")
    b.card(40, 135, 480, 80, "bytes per token =", ["2 (key and value) x layers x KV heads", "x head dimension x bytes per element"], "blue", size=12)
    b.card(40, 230, 480, 150, "Llama 3.1 8B", ["2 x 32 x 8 x 128 x 2 = 131,072 bytes", "= 128 KiB per token", "with all 32 heads: 524,288 bytes, 4x more", "SmolLM2-135M, 100 tokens, fp32:", "measured 4,608,000 = formula 4,608,000"], "green", size=12)

    b.group(560, 95, 620, 300, "KiB per token, one sequence", "teal")
    data = [("gpt2", "MHA", 36.0), ("falcon-7b", "MQA", 8.0), ("Mistral-7B", "GQA", 128.0), ("Llama 3.1 8B", "GQA", 128.0),
            ("Qwen2.5-7B", "GQA", 56.0), ("Llama 3.1 70B", "GQA", 320.0), ("SmolLM2-135M", "GQA", 22.5), ("DeepSeek-V2-Lite", "MLA", 30.4)]
    colors = {"MHA": "red", "GQA": "blue", "MQA": "green", "MLA": "purple"}
    for i, (name, kind, kib) in enumerate(data):
        y = 135 + i * 32
        raw_text(b, 690, y + 17, name, 12, INK, anchor="end")
        c = PALETTE[colors[kind]]
        rect(b, 700, y + 3, kib / 320 * 330, 22, c["fill"], c["stroke"], 2, rx=4)
        raw_text(b, 700 + kib / 320 * 330 + 8, y + 19, f"{kib:g}  {kind}", 12, c["text"], anchor="start", weight="700")

    b.group(20, 415, 1160, 255, "Four ways to store keys and values", "orange")
    b.card(40, 455, 265, 195, "MHA", ["every query head has", "its own K and V", "2 n_h d_h l elements", "gpt2: 36 KiB per token"], "red", size=12)
    b.card(325, 455, 265, 195, "GQA", ["query heads share K and V", "in n_g groups", "2 n_g d_h l elements", "Llama 3.1 8B: 8 groups of 32"], "blue", size=12)
    b.card(610, 455, 265, 195, "MQA", ["one K and V for all heads", "2 d_h l elements", "falcon-7b: 8 KiB per token", "the smallest, quality risk"], "green", size=12)
    b.card(895, 455, 265, 195, "MLA", ["cache a latent vector plus", "a small rotary key", "(d_c + d_h^R) l elements", "V2-Lite: (512 + 64) x 27"], "purple", size=12)
    return b


@board("kv-cache-and-paged-attention-paging")
def kv_paging():
    b = Board(1200, 700, "Paging the cache like virtual memory",
              "Llama 3.1 8B, 40 GB of cache = 305,175 token slots, seeded mix of 4,000 requests")
    b.group(20, 95, 590, 330, "Reserve the maximum, or take blocks on demand", "red")
    rows = [["allocator", "sequences", "waste"],
            ["contiguous, reserve 4096", "74", "91.8%"],
            ["paged, block 1", "916", "0.1%"],
            ["paged, block 8", "907", "1.1%"],
            ["paged, block 16", "895", "2.3%"],
            ["paged, block 32", "867", "4.5%"],
            ["paged, block 128", "765", "15.3%"],
            ["paged, block 512", "497", "45.1%"]]
    b.table(40, 135, [270, 150, 130], rows, "red", size=13, row_h=33)
    b.text(315, 415, "small blocks waste little, huge blocks bring the waste back", 11, FAINT, italic=True)

    b.group(630, 95, 550, 330, "A block table, block size 4", "blue")
    for i, (label, n) in enumerate([("sequence a: 10 tokens", 3), ("sequence b: 5 tokens", 2)]):
        y = 145 + i * 120
        raw_text(b, 650, y, label, 13, PALETTE["blue"]["text"], anchor="start", weight="700")
        ids = [[0, 1, 2], [3, 4]][i]
        for k, blk in enumerate(ids):
            x = 650 + k * 100
            rect(b, x, y + 15, 84, 50, PALETTE["blue"]["fill"], PALETTE["blue"]["stroke"], 2)
            raw_text(b, x + 42, y + 36, f"block {blk}", 12, PALETTE["blue"]["text"], weight="700")
            filled = 4 if k < n - 1 else (10 - 8 if i == 0 else 1)
            raw_text(b, x + 42, y + 54, f"{filled} of 4 used", 11, INK)
    b.text(905, 398, "5 blocks hold 15 tokens, block ids need not be adjacent", 11, FAINT, italic=True)

    b.group(20, 445, 560, 225, "Sharing a prompt, copy on write", "green")
    b.card(40, 485, 520, 70, "4 samples, 500-token prompt, 64 new tokens each", ["blocks of 16 tokens"], "green", size=12)
    b.card(40, 570, 250, 80, "with sharing", ["51 blocks"], "green", size=12, title_size=18)
    b.card(310, 570, 250, 80, "no sharing", ["144 blocks"], "red", size=12, title_size=18)

    b.group(600, 445, 580, 225, "What the PagedAttention paper reports", "purple")
    b.card(620, 485, 540, 70, "token states in the previous systems' cache", ["20.4% to 38.2% of the memory"], "purple", size=12)
    b.card(620, 570, 255, 80, "default block size", ["16 tokens"], "purple", size=12, title_size=16)
    b.card(895, 570, 265, 80, "throughput", ["2 to 4 times the others"], "purple", size=12, title_size=16)
    return b


@board("continuous-batching-and-scheduling-static-vs-continuous")
def batching_compare():
    b = Board(1200, 720, "Static against continuous batching",
              "A schematic of four slots, then the simulated trace: 600 requests, Llama 3.1 8B step-time model")
    b.group(20, 95, 560, 270, "Static: the batch waits for its longest request", "red")
    b.group(600, 95, 580, 270, "Continuous: a slot is refilled the moment it frees", "green")
    lens = [3, 7, 4, 9]
    for panel, x0 in [("static", 95), ("continuous", 685)]:
        unit = 46
        for slot in range(4):
            y = 150 + slot * 48
            raw_text(b, x0 - 6, y + 25, f"slot {slot + 1}", 11, FAINT, anchor="end")
            line(b, x0, y + 40, x0 + 9 * unit + 10, y + 40, "#e9ecef", 1)
        if panel == "static":
            for slot, n in enumerate(lens):
                y = 150 + slot * 48
                rect(b, x0, y + 4, n * unit, 30, PALETTE["blue"]["fill"], PALETTE["blue"]["stroke"], 2)
                raw_text(b, x0 + n * unit / 2, y + 24, f"request {chr(65 + slot)}", 11, PALETTE["blue"]["text"], weight="700")
                if n < 9:
                    rect(b, x0 + n * unit, y + 4, (9 - n) * unit, 30, "#f1f3f5", "#adb5bd", 1.5)
                    raw_text(b, x0 + n * unit + (9 - n) * unit / 2, y + 24, "idle, waiting", 11, FAINT)
        else:
            plan = [[("A", 3, "blue"), ("E", 4, "purple"), ("I", 2, "orange")],
                    [("B", 7, "blue"), ("F", 2, "purple")],
                    [("C", 4, "blue"), ("G", 3, "purple"), ("J", 2, "orange")],
                    [("D", 9, "blue")]]
            for slot, items in enumerate(plan):
                y = 150 + slot * 48
                cx = x0
                for name, n, col in items:
                    rect(b, cx, y + 4, n * unit - 3, 30, PALETTE[col]["fill"], PALETTE[col]["stroke"], 2)
                    raw_text(b, cx + n * unit / 2 - 1, y + 24, f"request {name}" if n > 2 else name, 11, PALETTE[col]["text"], weight="700")
                    cx += n * unit
        raw_text(b, x0, 350, "time  ->", 11, FAINT, anchor="start")

    b.group(20, 385, 1160, 315, "The same 600 requests, simulated (latency in s, TTFT in ms, rate in requests per second)", "teal")
    rows = [["policy", "5/s: tok/s", "latency", "TTFT", "30/s: tok/s", "latency", "TTFT"],
            ["static, batch 16", "791", "5.98", "3,155", "860", "53.31", "50,274"],
            ["static, batch 64", "809", "4.62", "1,691", "1,969", "19.52", "14,777"],
            ["continuous, batch 64", "815", "0.83", "8.2", "4,324", "0.91", "9.6"]]
    b.table(40, 430, [270, 140, 120, 130, 150, 120, 130], rows, "teal", size=14, row_h=40)
    b.card(40, 610, 540, 70, "light load: same throughput, different wait", ["throughput is the offered load, but a static", "request waits for the batch: 0.83 s against 5.98 s"], "green", size=12)
    b.card(600, 610, 560, 70, "heavy load: static cannot keep up", ["30 per second asks for about 5,040 tokens/s, far above", "860 or 1,969: the queue grows, 53 s mean latency at batch 16"], "red", size=12)
    return b


@board("continuous-batching-and-scheduling-scheduling")
def batching_scheduling():
    b = Board(1200, 760, "Queues, chunks and prefixes",
              "Continuous batching, max batch 64; the rate sweep and policies are block 1, the prefix cache is block 2")
    b.group(20, 95, 560, 300, "Arrival rate sweep: the knee near 60 per second", "orange")
    rows = [["req/s", "tok/s", "TTFT mean", "TTFT p99", "in system"],
            ["10", "1,601", "8.6", "19.6", "8.0"],
            ["30", "4,324", "9.6", "24.0", "23.3"],
            ["50", "6,252", "12.5", "72.4", "37.0"],
            ["60", "6,937", "127.8", "395.7", "47.5"],
            ["70", "7,113", "532.8", "1,239.7", "66.2"]]
    b.table(40, 140, [80, 100, 120, 110, 100], rows, "orange", size=13, row_h=36)
    b.text(300, 378, "in system = rate x mean latency (Little's law): 23.3 against 23.4 at 30 per second", 11, FAINT, italic=True)

    b.group(600, 95, 580, 300, "Policies at 58 requests per second", "purple")
    rows = [["policy", "TTFT mean", "TTFT p99", "gap p99"],
            ["first come first served", "68.6", "254.7", "14.95"],
            ["shortest prompt first", "51.8", "751.9", "15.44"],
            ["chunked, 512 tokens", "41.3", "201.5", "9.38"],
            ["chunked, 256 tokens", "23.2", "151.3", "5.89"]]
    b.table(620, 140, [240, 110, 110, 90], rows, "purple", size=13, row_h=38)
    b.text(890, 378, "all times in ms; gap = time between a user's successive tokens", 11, FAINT, italic=True)

    b.group(20, 415, 560, 325, "A long prefill stalls everyone's decode", "blue")
    rect(b, 45, 465, 510, 56, PALETTE["blue"]["fill"], PALETTE["blue"]["stroke"], 2)
    raw_text(b, 300, 488, "one step: a 2,048-token prompt plus the decodes", 12, PALETTE["blue"]["text"], weight="700")
    raw_text(b, 300, 508, "33.24 ms for the prompt alone (chapter 1): every decode waits", 12, INK)
    for k in range(4):
        rect(b, 45 + k * 130, 565, 120, 46, PALETTE["green"]["fill"], PALETTE["green"]["stroke"], 2)
        raw_text(b, 105 + k * 130, 585, "512-token chunk", 11, PALETTE["green"]["text"], weight="700")
        raw_text(b, 105 + k * 130, 601, "+ all decodes", 11, INK)
    raw_text(b, 300, 546, "chunked prefill splits it, decodes ride in every step", 12, PALETTE["green"]["text"], weight="700")
    b.text(300, 640, "p99 inter-token gap at 58 per second: 14.95 ms unchunked,", 12, INK)
    b.text(300, 660, "9.38 ms with a 512-token budget, 5.89 ms with 256", 12, INK)
    b.text(300, 700, "the total prefill arithmetic is unchanged, only spread over steps", 11, FAINT, italic=True)

    b.group(600, 415, 580, 325, "Prefix cache: share of prompt tokens served from it", "teal")
    rows = [["system prompts", "32 blocks", "256", "4,096"],
            ["1", "47.1%", "78.4%", "78.4%"],
            ["4", "12.5%", "77.5%", "77.7%"],
            ["16", "2.8%", "37.3%", "76.1%"],
            ["64", "1.1%", "9.9%", "68.7%"]]
    b.table(620, 460, [200, 130, 110, 110], rows, "teal", size=13, row_h=38)
    b.card(620, 665, 540, 60, "one changed token at the start: 0.0% reuse", ["the whole prefix hash chain changes"], "red", size=12)
    return b


@board("quantisation-for-inference-formats")
def quant_formats():
    b = Board(1200, 760, "Fewer bits per weight, and where the scale lives",
              "Formats, scale granularity and the first experiments (block 1)")
    b.group(20, 95, 570, 300, "Number formats", "blue")
    rows = [["format", "bits", "how the value is stored"],
            ["bf16 / fp16", "16", "floating point, no scale needed"],
            ["FP8 E4M3", "8", "4-bit exponent, 3-bit mantissa"],
            ["FP8 E5M2", "8", "5-bit exponent, 2-bit mantissa"],
            ["INT8", "8", "integer code times a scale"],
            ["INT4", "4", "integer code times a scale"],
            ["NF4", "4", "16 levels from normal quantiles"]]
    b.table(40, 140, [130, 70, 340], rows, "blue", size=13, row_h=33)
    b.text(305, 388, "FP8 as defined in the FP8 paper; NF4 levels as built in block 1", 11, FAINT, italic=True)

    b.group(610, 95, 570, 300, "Where the scale lives", "orange")
    for i, (label, sub, count) in enumerate([("one scale per tensor", "an outlier sets the grid", 1),
                                              ("one scale per row", "per output channel", 4),
                                              ("one scale per group", "32 to 128 weights share one", 16)]):
        y = 140 + i * 82
        raw_text(b, 630, y + 14, label, 13, PALETTE["orange"]["text"], anchor="start", weight="700")
        raw_text(b, 630, y + 33, sub, 11, INK, anchor="start")
        for k in range(count if count <= 16 else 16):
            w = 280 / count
            rect(b, 880 + k * w, y + 4, w - 3, 38, PALETTE["orange"]["fill"], PALETTE["orange"]["stroke"], 1.5, rx=3)
    b.text(895, 380, "storage = bits + 16 / group: 4-bit codes, groups of 64 = 4.25 bits", 11, FAINT, italic=True)

    b.group(20, 415, 570, 330, "An outlier of 40 among 4,096 N(0,1) weights", "red")
    rows = [["scheme", "8-bit error", "4-bit error"],
            ["absmax, one scale", "0.008233", "0.983967"],
            ["absmax, groups of 256", "0.000538", "0.076620"],
            ["absmax, groups of 64", "0.000155", "0.026326"],
            ["zero point, groups of 64", "0.000069", "0.018636"]]
    b.table(40, 460, [260, 140, 140], rows, "red", size=13, row_h=38)
    b.card(40, 665, 530, 60, "error measured on the 4,095 ordinary weights", ["one outlier at 4 bits wipes out the rest: 0.98"], "red", size=12)

    b.group(610, 415, 570, 330, "Where to put 16 levels, 4-bit blocks of 64", "green")
    rows = [["levels", "error / variance"],
            ["15 even (integer absmax)", "0.0118"],
            ["16 even", "0.0102"],
            ["16 from normal quantiles", "0.0085"]]
    b.table(630, 460, [320, 200], rows, "green", size=13, row_h=38)
    b.card(630, 625, 530, 100, "if weights are bell shaped", ["so levels packed near zero beat even spacing:", "the idea behind NF4 in the QLoRA paper", "(levels here are NF4-style, not the exact table)"], "green", size=12)
    return b


@board("quantisation-for-inference-quality")
def quant_quality():
    b = Board(1200, 800, "What quantising a real 135M model costs",
              "SmolLM2-135M-Instruct, perplexity on 4,096 WikiText-2 test tokens, fp32 = 18.07")
    b.group(20, 95, 590, 310, "Round to nearest, every decoder layer", "teal")
    rows = [["bits", "one row", "g192", "g64", "g32"],
            ["8", "18.12", "18.11", "18.13", "18.10"],
            ["6", "18.86", "18.54", "18.41", "18.32"],
            ["5", "21.37", "19.96", "19.43", "18.88"],
            ["4", "37.80", "28.23", "24.85", "23.07"],
            ["3", "7921.37", "698.95", "178.79", "102.26"]]
    b.table(40, 140, [70, 120, 110, 110, 110], rows, "teal", size=14, row_h=38)
    b.text(315, 392, "4 bits with groups of 32 = 4.50 bits per weight and perplexity 23.07", 11, FAINT, italic=True)

    b.group(630, 95, 550, 310, "4 bits, groups of 64: use the activations", "purple")
    rows = [["method", "perplexity", "output error"],
            ["round to nearest", "24.85", "0.0121"],
            ["AWQ-style scaling", "21.84", "0.0077"],
            ["GPTQ-style", "22.38", "0.0038"]]
    b.table(650, 140, [220, 150, 150], rows, "purple", size=13, row_h=38)
    b.card(650, 305, 510, 80, "lowest layer error is not the best perplexity", ["GPTQ-style halves AWQ-style's output error", "yet its perplexity is higher: 22.38 against 21.84"], "purple", size=12)

    b.group(20, 425, 590, 355, "A trap: GPTQ-style on v_proj only", "red")
    rows = [["calibration", "perplexity", "first-token error"],
            ["round to nearest", "18.85", "(reference)"],
            ["windows of 1024", "23.17", "1.70"],
            ["windows of 64", "18.92", "0.34"]]
    b.table(40, 470, [220, 140, 170], rows, "red", size=13, row_h=38)
    b.card(40, 640, 550, 120, "the first token of a window is the attention sink", ["with long windows it is 1 row in 1,024, so the", "fit gives it up: 1.70 relative error at that token.", "Short windows put it back in the fit."], "red", size=12)

    b.group(630, 425, 550, 355, "Integer 8 is not free: dynamic int8 on CPU", "orange")
    rows = [["variant", "file MB", "perplexity"],
            ["fp32", "538", "18.07"],
            ["dynamic int8, decoder layers", "221", "26.94"],
            ["dynamic int8, every Linear", "249", "27.66"]]
    b.table(650, 470, [270, 100, 130], rows, "orange", size=13, row_h=38)
    b.card(650, 640, 510, 120, "weight-only per-row int8 costs nothing", ["perplexity 18.12 in the first table, against", "26.94 when activations are also quantised per", "tensor, and the forward pass got slower here"], "orange", size=12)
    return b


@board("quantisation-for-inference-kv")
def quant_kv():
    b = Board(1200, 560, "Quantising the KV cache: keys per channel, values per token",
              "Fake-quantised keys and values in every layer of SmolLM2-135M, groups of 32 tokens per channel; fp32 perplexity 18.07")
    b.group(20, 95, 700, 300, "Perplexity by bit width and scheme", "blue")
    rows = [["bits", "tokens / tokens", "keys by channel", "values by channel"],
            ["8", "18.07", "18.07", "18.07"],
            ["4", "23.07", "19.19", "23.46"],
            ["3", "75.70", "25.57", "75.05"],
            ["2", "1381.39", "1062.27", "3421.69"]]
    b.table(40, 140, [90, 190, 190, 190], rows, "blue", size=14, row_h=44)
    b.text(370, 366, "column 1: keys and values per token; column 2: keys per channel, values per token;\ncolumn 3: keys per token, values per channel", 10, FAINT, italic=True)
    b.group(740, 95, 440, 300, "What this shows", "green")
    b.card(760, 140, 400, 110, "keys have channel structure", ["per-channel keys at 3 bits: 25.57,", "per-token keys at 3 bits: 75.70.", "KIVI's finding reproduced on a small model"], "green", size=12)
    b.card(760, 265, 400, 110, "values do not", ["per-channel values at 4 bits: 23.46,", "per-token values: 23.07.", "KIVI uses per-token for values"], "orange", size=12)
    b.card(20, 415, 1160, 120, "At 2 bits this simple scheme collapses", ["a simplified simulation: whole sequences fake-quantised in every layer,", "none of the extra engineering of the KIVI paper, so these rows are not a measurement of KIVI itself"], "grey", size=12)
    return b


@board("speculative-decoding-draft-verify")
def spec_draft_verify():
    b = Board(1200, 760, "Draft cheaply, verify in one pass",
              "A small model guesses several tokens, the large model checks them all at once and keeps the right prefix")
    b.group(20, 95, 1160, 250, "One step with draft length 4", "blue")
    toks = [("the", "ok"), ("cat", "ok"), ("sat", "ok"), ("on", "bad")]
    raw_text(b, 60, 140, "draft model proposes, one cheap step each:", 13, PALETTE["blue"]["text"], anchor="start", weight="700")
    for i, (t, st) in enumerate(toks):
        col = "green" if st == "ok" else "red"
        rect(b, 60 + i * 120, 155, 105, 46, PALETTE[col]["fill"], PALETTE[col]["stroke"], 2)
        raw_text(b, 112 + i * 120, 184, t, 14, PALETTE[col]["text"], weight="700")
    raw_text(b, 60, 235, "target model checks all four in a single forward pass:", 13, PALETTE["blue"]["text"], anchor="start", weight="700")
    for i, (t, st) in enumerate(toks):
        col = "green" if st == "ok" else "red"
        rect(b, 60 + i * 120, 250, 105, 46, PALETTE[col]["fill"], PALETTE[col]["stroke"], 2)
        raw_text(b, 112 + i * 120, 279, "keep" if st == "ok" else "reject", 13, PALETTE[col]["text"], weight="700")
    rect(b, 545, 250, 175, 46, PALETTE["purple"]["fill"], PALETTE["purple"]["stroke"], 2)
    raw_text(b, 632, 279, "+1 from the target", 13, PALETTE["purple"]["text"], weight="700")
    b.card(740, 150, 410, 90, "this step produced 4 tokens", ["3 accepted + 1 from the target,", "for the price of one target pass and 4 draft steps"], "green", size=12)
    b.card(740, 255, 410, 70, "output distribution is unchanged", ["accept with probability min(1, p / q), else resample"], "purple", size=12)

    b.group(20, 365, 570, 380, "Tokens per target step: formula, then simulation", "teal")
    rows = [["alpha", "gamma", "formula", "simulated"],
            ["0.5", "4", "1.938", "1.935"],
            ["0.7", "4", "2.773", "2.777"],
            ["0.8", "2", "2.440", "2.444"],
            ["0.8", "4", "3.362", "3.366"],
            ["0.8", "8", "4.329", "4.335"],
            ["0.9", "4", "4.095", "4.096"]]
    b.table(40, 410, [110, 110, 150, 150], rows, "teal", size=14, row_h=38)
    b.text(305, 700, "E = (1 - alpha^(gamma + 1)) / (1 - alpha);", 12, INK)
    b.text(305, 720, "speedup = E / (gamma c + 1) when verification is free", 12, INK)

    b.group(610, 365, 570, 380, "Is the accepted output really from p?", "orange")
    rows = [["token", "p", "q", "speculative", "draft alone"],
            ["0", "0.400", "0.150", "0.402", "0.150"],
            ["1", "0.250", "0.300", "0.250", "0.301"],
            ["2", "0.150", "0.200", "0.149", "0.199"],
            ["4", "0.070", "0.250", "0.069", "0.251"]]
    b.table(630, 410, [80, 90, 90, 130, 130], rows, "orange", size=13, row_h=38)
    b.card(630, 615, 530, 110, "400,000 trials", ["distance to p: 0.0020 speculative, 0.3003 draft alone", "acceptance 0.6983 measured, 0.7000 = sum of min(p, q)"], "orange", size=12)
    return b


@board("speculative-decoding-reality")
def spec_reality():
    b = Board(1200, 800, "When it pays and when it does not",
              "SmolLM2-1.7B target, SmolLM2-135M draft, CPU, one run (timings vary); roofline rows use the chapter 1 H100 figures")
    b.group(20, 95, 570, 330, "Measured acceptance and cost, gamma 4", "blue")
    rows = [["measure", "value"],
            ["acceptance alpha", "0.769"],
            ["tokens per target pass", "3.20 (formula 3.17)"],
            ["draft step / target step, c", "0.130"],
            ["verify 5 tokens / 1 step, v", "3.28"],
            ["speedup if v were 1", "2.08"],
            ["speedup with measured v", "0.83"]]
    b.table(40, 140, [280, 250], rows, "blue", size=13, row_h=38)

    b.group(610, 95, 570, 330, "Real wall-clock, 48 new tokens, same output", "green")
    rows = [["task and method", "tok/s"],
            ["open text, plain", "10.8"],
            ["open text, draft model", "7.5"],
            ["open text, prompt lookup", "7.4"],
            ["copy from prompt, plain", "11.0"],
            ["copy from prompt, draft model", "15.7"],
            ["copy from prompt, prompt lookup", "16.7"]]
    b.table(630, 140, [360, 170], rows, "green", size=13, row_h=38)

    b.group(20, 445, 1160, 335, "Batching eats the gain: roofline, alpha 0.8, gamma 4, 8B target, 0.5B draft", "purple")
    rows = [["batch", "1", "32", "64", "128", "256", "512"],
            ["plain step, ms", "4.79", "4.79", "4.79", "4.79", "4.79", "8.31"],
            ["verify step, ms", "4.79", "4.79", "5.19", "10.39", "20.77", "41.55"],
            ["speedup per sequence", "2.69", "2.69", "2.52", "1.39", "0.73", "0.64"]]
    b.table(40, 490, [260, 140, 140, 140, 140, 140, 140], rows, "purple", size=14, row_h=42)
    b.card(40, 675, 560, 85, "break-even near a batch of 184", ["verifying 5 tokens per sequence reaches the ridge", "at a batch of about 59, and keeps getting dearer"], "purple", size=12)
    b.card(620, 675, 540, 85, "also true on the CPU above", ["verifying 5 tokens cost 3.28 single steps there,", "so the speedup was below 1 despite alpha 0.769"], "red", size=12)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or v.endswith(name))
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
