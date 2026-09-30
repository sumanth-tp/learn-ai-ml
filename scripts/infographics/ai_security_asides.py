"""Two brief Module 3 source visuals omitted from the initial board batch.

Source frames: AI Security course rQE3w8Qjx98, 4:06:04 and 4:19:00–4:19:55.
Run from any directory: python3 scripts/infographics/ai_security_asides.py
"""
from pathlib import Path
from board import Board

OUT = Path(__file__).resolve().parents[2] / "static/img/ai-security"


def kv_cache():
    b = Board(1280, 1040, "Long context needs a larger KV cache",
              "Why memory budgets matter · visual aside at 4:19")
    b.group(25, 95, 1230, 310, "RETAIN KEYS AND VALUES FROM EARLIER TOKENS", "orange")
    layer = b.card(50, 160, 205, 170, "Each attention layer", ["Layer 1", "...", "Layer N"], "purple", size=15)
    for i, name in enumerate(["Token 1", "Token 2", "Token 3", "... Token T"]):
        x = 325 + i * 220
        t = b.card(x, 160, 185, 55, name, [], "blue", title_size=16)
        kv = b.card(x, 280, 185, 70, "Keys + values", ["Reused by later tokens"], "orange", size=11)
        b.arrow(t.bottom(), kv.top(), color="orange")
    b.arrow(layer.right(), (305, 245), color="purple")
    b.text(640, 385, "More tokens → more retained K/V vectors → more device memory", 16, "orange", "700")
    b.group(25, 455, 380, 390, "MHA · separate K/V", "blue")
    b.group(450, 455, 380, 390, "GQA · shared K/V groups", "green")
    b.group(875, 455, 380, 390, "MLA · compressed latent", "purple")
    for i in range(4):
        q = b.card(45, 530 + i * 70, 100, 45, f"Q{i + 1}", [], "blue", size=12)
        kv = b.card(235, 530 + i * 70, 150, 45, f"K{i + 1} / V{i + 1}", [], "blue", size=12)
        b.arrow(q.right(), kv.left(), color="blue")
    shared = b.card(655, 620, 155, 100, "Shared K/V", ["One group"], "green", size=12)
    for i in range(4):
        q = b.card(470, 530 + i * 70, 100, 45, f"Q{i + 1}", [], "green", size=12)
        b.arrow(q.right(), shared.left(), color="green")
    heads = b.card(910, 535, 310, 65, "Query heads Q1 ... Qn", [], "purple", title_size=16)
    latent = b.card(950, 715, 230, 85, "Compressed latent", ["Compact shared representation"], "purple", size=12)
    b.arrow(heads.bottom(), latent.top(), color="purple")
    b.card(80, 880, 1120, 90, "Cache size depends on the architecture",
           ["For a standard K/V cache: 2 × tokens × layers × KV heads × head dimension × bytes per element.",
            "The screen's 32 query heads / 8 KV groups example gives 8/32 = 1/4 of the MHA cache, all else equal."],
           "grey", size=13, title_size=17)
    b.text(640, 1008, "The source's ‘roughly halves’ wording does not match that head-count example; the notes correct it.", 13, "grey")
    return b


def temporal():
    b = Board(1280, 820, "Temporal memory: retrieve by meaning and time",
              "A board shown briefly at 4:06 and 4:11, without a separate taught chapter")
    query = b.card(30, 130, 250, 100, "Query", ["Optional date constraints"], "purple", size=14)
    store = b.cylinder(30, 340, 250, 135, "Memory store", ["Timestamped records"], "purple", size=14)
    b.group(330, 95, 925, 445, "RETRIEVAL PIPELINE", "teal")
    time = b.card(355, 265, 235, 110, "Time-range filter", ["timestamp BETWEEN", "start AND end"], "teal", size=14)
    semantic = b.card(680, 160, 235, 100, "Semantic score", ["Embedding similarity"], "green", size=14)
    recency = b.card(680, 360, 235, 100, "Recency score", ["Exponential decay"], "red", size=14)
    rank = b.card(1005, 245, 220, 145, "Combined ranking", ["w1 × semantic", "+ w2 × recency", "Keep top-k"], "orange", size=14)
    b.arrow(query.right(), time.left(), via=[(310, 180), (310, 320)], color="blue")
    b.arrow(store.right(), time.left(), via=[(310, 407), (310, 320)], color="blue")
    b.arrow(time.right(), semantic.left(), via=[(635, 320), (635, 210)], color="green")
    b.arrow(time.right(), recency.left(), via=[(635, 320), (635, 410)], color="red")
    b.arrow(semantic.right(), rank.left(), via=[(960, 210), (960, 317)], color="green")
    b.arrow(recency.right(), rank.left(), via=[(960, 410), (960, 317)], color="red")
    b.group(330, 585, 925, 165, "TIMELINE BRANCH", "pink")
    construct = b.card(390, 645, 360, 75, "Group by time windows", [], "pink", title_size=17)
    timeline = b.card(855, 645, 340, 75, "Ordered event timeline", [], "pink", title_size=17)
    b.arrow(store.bottom(), construct.left(), via=[(155, 682)], color="pink")
    b.arrow(construct.right(), timeline.left(), color="pink")
    b.text(640, 790, "Blended ranking can favour recent relevant facts; the timeline preserves their sequence.", 14, "grey")
    return b


if __name__ == "__main__":
    for name, make in [("m3-kv-cache", kv_cache), ("m3-temporal", temporal)]:
        print(make().save(OUT / f"{name}.svg"))
