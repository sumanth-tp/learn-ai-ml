"""Infographics for docs/senior/01-system-design-cases, cases 1 to 3.

Run from the repo root:

    python3 scripts/infographics/senior_1.py            # all boards
    python3 scripts/infographics/senior_1.py funnel     # boards whose name contains the word
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "senior"
BOARDS = {}
NAMES = {}


def board(name):
    def deco(fn):
        BOARDS[fn.__name__] = fn
        NAMES[fn.__name__] = name
        return fn
    return deco


@board("enterprise-document-qa-architecture")
def qa_architecture():
    b = Board(1240, 700, "Enterprise document Q&A: two paths", "2 million documents, 30 million chunks, 20,000 employees; sizes from the chapter estimator")
    b.group(20, 95, 1200, 235, "Ingestion path (offline)", "blue")
    c1 = b.card(40, 145, 190, 90, "connectors", ["copy each file's", "permissions with it"], "blue", size=12)
    c2 = b.card(265, 145, 190, 90, "parse and chunk", ["15 chunks per document", "350 tokens each"], "blue", size=12)
    c3 = b.card(490, 145, 190, 90, "embed", ["10.50 billion tokens", "once, again on a new model"], "blue", size=12)
    b.arrow(c1.right(), c2.left())
    b.arrow(c2.right(), c3.left())
    v = b.cylinder(730, 140, 150, 100, "vector index", ["int8 30.7 GB", "graph 3.8 GB"], "purple", size=11)
    k = b.cylinder(900, 140, 150, 100, "keyword index", ["BM25 postings"], "purple", size=11)
    d = b.cylinder(1070, 140, 130, 100, "chunk store", ["text 42.0 GB", "metadata 6.0 GB"], "purple", size=11)
    b.arrow(c3.right(), v.left(), label="vectors")
    b.card(40, 255, 640, 55, "ACL rides with every chunk", ["a vector index without permissions is a leak waiting for a clever question"], "red", size=12)
    b.card(730, 255, 470, 55, "float32 vectors would need 122.9 GB", ["int8 quantisation cuts it to a quarter"], "grey", size=12)

    b.group(20, 350, 1200, 250, "Query path (online)", "green")
    q1 = b.card(40, 400, 135, 90, "auth", ["user, groups,", "date"], "green", size=11)
    q2 = b.card(200, 400, 145, 90, "retrieve", ["BM25 top 50", "dense top 50"], "green", size=11)
    q3 = b.card(370, 400, 135, 90, "fuse", ["reciprocal rank", "fusion, k = 60"], "green", size=11)
    q4 = b.card(530, 400, 135, 90, "ACL filter", ["inside the search,", "not after it"], "red", size=11)
    q5 = b.card(690, 400, 135, 90, "rerank", ["top 20 scored by", "a cross-encoder"], "orange", size=11)
    q6 = b.card(850, 400, 150, 90, "pack context", ["token budget,", "best at both ends"], "orange", size=11)
    q7 = b.card(1025, 400, 175, 90, "LLM answer", ["cited, checked", "against the chunks"], "purple", size=11)
    for a, c in [(q1, q2), (q2, q3), (q3, q4), (q4, q5), (q5, q6), (q6, q7)]:
        b.arrow(a.right(), c.left())
    b.card(40, 515, 600, 70, "load is small, memory is not", ["42,000 questions a day, 5.2 per second at peak:", "the index size, not the traffic, sets the bill's floor"], "yellow", size=12)
    b.card(660, 515, 540, 70, "cost per question", ["6 chunks: 2,560 input tokens, $0.0129 at the assumed prices", "monthly $11,947; 3 chunks $9,037; 10 chunks $15,828"], "grey", size=12)
    b.text(620, 640, "prices are placeholders in the estimator, not any provider's list", 12, "#868e96", italic=True)
    return b


@board("enterprise-document-qa-funnel")
def qa_funnel():
    b = Board(1240, 660, "The retrieval funnel, measured", "100 SciFact questions, 1,500 documents: share of questions with a relevant document in the top k")
    b.group(20, 95, 620, 270, "Three retrievers", "teal")
    rows = [["retriever", "top 1", "top 5", "top 10", "top 20", "top 50"],
            ["BM25", "0.71", "0.91", "0.93", "0.93", "0.95"],
            ["dense (MiniLM)", "0.75", "0.86", "0.90", "0.92", "0.98"],
            ["hybrid (RRF)", "0.77", "0.92", "0.94", "0.98", "0.99"]]
    b.table(40, 140, [170, 80, 80, 80, 80, 80], rows, "teal", size=13, row_h=36)
    b.card(40, 295, 580, 55, "each retriever misses different questions", ["fusing the two ranked lists recovers most of them by depth 20"], "teal", size=12)

    b.group(660, 95, 560, 270, "Rerank the hybrid shortlist (cross-encoder)", "orange")
    rows = [["after rerank", "top 1", "top 3", "top 5"],
            ["hybrid, no rerank", "0.77", "0.88", "0.92"],
            ["hybrid + rerank 10", "0.74", "0.86", "0.91"],
            ["hybrid + rerank 20", "0.74", "0.82", "0.89"]]
    b.table(680, 140, [200, 100, 100, 100], rows, "orange", size=13, row_h=36)
    b.card(680, 285, 520, 70, "this small reranker did not help here", ["ms-marco MiniLM, scientific claims, 100 questions:", "measure on your own questions first"], "red", size=12)

    b.group(20, 385, 620, 245, "Permissions: filter before or after?", "purple")
    b.card(40, 435, 285, 110, "post-filter", ["retrieve 10, drop unauthorised", "3.7 results left on average", "hit rate at 5: 0.94"], "purple", size=12)
    b.card(335, 435, 285, 110, "pre-filter", ["restrict the search itself,", "then take 5", "hit rate at 5: 0.97"], "green", size=12)
    b.card(40, 560, 580, 55, "user sees 30% of documents plus the evidence for their question", [], "grey", size=12)

    b.group(660, 385, 560, 245, "Chunks in the prompt cost money", "yellow")
    rows = [["chunks", "input tokens", "per month"],
            ["3", "1,510", "$9,037"],
            ["6", "2,560", "$11,947"],
            ["10", "3,960", "$15,828"]]
    b.table(680, 435, [140, 190, 190], rows, "yellow", size=13, row_h=36)
    b.card(680, 585, 520, 38, "assumed prices, 42,000 questions a day", [], "grey", size=11)
    return b


@board("real-time-fraud-scoring-architecture")
def fraud_architecture():
    b = Board(1240, 700, "Real-time fraud scoring: decide now, learn later", "5,000 transactions per second at peak, 100 ms to decide")
    b.group(20, 95, 1200, 270, "Decision path (synchronous), p99 allowance in ms", "red")
    s1 = b.card(40, 150, 150, 90, "gateway in", ["network 12"], "red", size=12)
    s2 = b.card(225, 150, 200, 90, "feature fetch", ["8 lookups in parallel", "allowance 25"], "orange", size=12)
    s3 = b.card(460, 150, 150, 90, "model", ["scores the", "transaction: 14"], "purple", size=12)
    s4 = b.card(645, 150, 150, 90, "rules and cut-offs", ["policy: 3"], "blue", size=12)
    s5 = b.card(830, 150, 150, 90, "decision", ["approve, step up,", "decline: out 12"], "green", size=12)
    for a, c in [(s1, s2), (s2, s3), (s3, s4), (s4, s5)]:
        b.arrow(a.right(), c.left())
    b.card(1005, 150, 195, 90, "sum of p99s: 66", ["limit 100, so 34 ms", "of headroom"], "grey", size=12)
    b.card(225, 270, 350, 75, "fetch slower than the timeout?", ["decide with default features and rules,", "mark the decision degraded"], "yellow", size=12)
    b.arrow(s2.bottom(0.5), (400, 270), color="red", dashed=True)
    b.card(610, 270, 590, 75, "an approved fraud is paid for in full", ["a wrongly declined customer is paid for in friction:", "the cut-off comes from the two costs, not from 0.5"], "grey", size=12)

    b.group(20, 385, 1200, 275, "Learning path (asynchronous)", "blue")
    a1 = b.card(40, 440, 190, 90, "event stream", ["every transaction", "and every outcome"], "blue", size=12)
    a2 = b.card(265, 440, 190, 90, "stream features", ["velocity windows", "written to online store"], "teal", size=12)
    a3 = b.card(490, 440, 190, 90, "labels arrive late", ["chargebacks take", "weeks; reviews are biased"], "orange", size=12)
    a4 = b.card(715, 440, 230, 90, "training set", ["point-in-time join:", "features as served then"], "purple", size=12)
    a5 = b.card(980, 440, 220, 90, "retrain, shadow, canary", ["compare on live traffic", "before it decides"], "green", size=12)
    b.arrow(a1.right(), a2.left())
    b.arrow(a2.right(), a3.left())
    b.arrow(a3.right(), a4.left())
    b.arrow(a4.right(), a5.left())
    b.card(40, 560, 1160, 80, "the same feature code serves training and serving", ["a velocity count 5 minutes stale scored average precision 0.4853 against 0.8222 fresh:", "skew, not the algorithm, is the usual cause of a model that worked offline and fails live"], "red", size=12)
    return b


@board("real-time-fraud-scoring-latency-and-drift")
def fraud_latency():
    b = Board(1280, 700, "Where the milliseconds and the accuracy go", "20,000 simulated decisions and a 60-day simulated transaction history")
    b.group(20, 95, 700, 300, "Feature fetch design, end to end (ms)", "orange")
    rows = [["design", "p50", "p99", "p99.9", ">100 ms"],
            ["8 lookups in series", "56.0", "104.2", "134.0", "1.39%"],
            ["8 in parallel", "26.8", "57.9", "93.2", "0.07%"],
            ["parallel + hedge", "26.6", "43.7", "50.3", "0.00%"],
            ["+ 20 ms timeout", "26.6", "42.0", "48.0", "0.00%"]]
    b.table(40, 140, [250, 100, 100, 110, 110], rows, "orange", size=13, row_h=36)
    b.card(40, 335, 660, 45, "hedging sent 4.93% extra calls; the timeout degraded 5.49% of decisions", [], "grey", size=12)

    b.group(740, 95, 520, 300, "One slow lookup in n", "red")
    rows = [["lookups", "chance one is in its own slowest 1%"],
            ["1", "1.0%"], ["8", "7.7%"], ["20", "18.2%"], ["100", "63.4%"]]
    b.table(760, 140, [100, 380], rows, "red", size=13, row_h=36)
    b.card(760, 335, 480, 45, "the 100-server case matches the Tail at Scale paper", [], "grey", size=12)

    b.group(20, 415, 620, 265, "Why a random split flatters the model", "purple")
    rows = [["validation", "ROC AUC", "avg precision"],
            ["random 70/30", "0.9755", "0.8438"],
            ["train days 0-39, test 40-59", "0.7243", "0.5151"],
            ["test 50-59, labels at once", "0.9304", "0.6349"],
            ["test 50-59, 14-day delay", "0.7317", "0.5104"]]
    b.table(40, 460, [280, 130, 150], rows, "purple", size=12, row_h=34)
    b.card(40, 640, 580, 32, "half the attacks switch style on day 40", [], "grey", size=11)

    b.group(660, 415, 600, 265, "Stale velocity features", "teal")
    rows = [["train lag", "serve lag", "avg precision"],
            ["0 s", "0 s", "0.8222"], ["0 s", "300 s", "0.4853"],
            ["0 s", "1,800 s", "0.4048"], ["300 s", "300 s", "0.6848"]]
    b.table(680, 460, [170, 170, 220], rows, "teal", size=12, row_h=34)
    b.card(680, 640, 560, 32, "serve what you trained on, or retrain on what you serve", [], "grey", size=11)
    return b


@board("search-and-recommendation-ranking-funnel")
def ranking_funnel():
    b = Board(1240, 730, "Ranking is a funnel with a budget at every stage", "assumed costs: 2 microseconds per item light, 60 heavy, 3 ms retrieval; 20,000 requests per second")
    stages = [
        (640, "catalogue: 50,000,000 items", ["all of them through the heavy ranker:", "3,000 CPU seconds per request"], "grey"),
        (540, "retrieve 1,000", ["nearest neighbours on embeddings: 3.0 ms"], "blue"),
        (440, "light ranker: 1,000 scored", ["2 microseconds each: 2.0 ms, keep 200"], "teal"),
        (340, "heavy ranker: 200 scored", ["60 microseconds each: 12.0 ms"], "purple"),
        (240, "rules, then 10 shown", ["diversity, freshness"], "green"),
    ]
    y = 100
    prev = None
    for w, title, lines, col in stages:
        cur = b.card(340 - w / 2, y, w, 70, title, lines, col, size=12)
        if prev is not None:
            b.arrow(prev.bottom(), cur.top())
        prev = cur
        y += 98
    b.card(20, 590, 640, 105, "17.0 CPU ms per request, 680 cores", ["3.0 + 2.0 + 12.0 ms at 20,000 requests per second", "and half-loaded cores", "if the heavy ranker kept 800: 53.0 ms, 2,120 cores"], "yellow", size=12)

    b.group(700, 95, 520, 300, "Simulation: precision at 10", "orange")
    rows = [["retrieve", "keep 50", "keep 100", "keep 200"],
            ["100", "0.227", "0.238", "-"],
            ["300", "0.307", "0.317", "0.321"],
            ["1000", "0.289", "0.330", "0.345"],
            ["2000", "0.288", "0.326", "0.351"]]
    b.table(715, 140, [100, 130, 130, 130], rows, "orange", size=13, row_h=34)
    b.card(715, 325, 490, 55, "depth pays up to a plateau", ["1000 to 2000 retrieved adds 0.006 at keep 200"], "grey", size=12)

    b.group(700, 415, 520, 290, "Retrieval ceiling for a perfect ranker", "red")
    rows = [["retrieve", "best possible precision at 10"],
            ["100", "0.576"], ["300", "0.954"], ["1000", "1.000"], ["2000", "1.000"]]
    b.table(715, 460, [110, 380], rows, "red", size=13, row_h=34)
    b.card(715, 650, 490, 40, "what retrieval drops, no ranker can bring back", [], "grey", size=12)
    return b


@board("search-and-recommendation-ranking-position-bias")
def ranking_bias():
    b = Board(1140, 600, "Clicks are not relevance", "6,000 simulated queries of 20 items each; the old ranker decided the order that was clicked")
    b.group(20, 95, 520, 240, "Attention falls with position", "blue")
    rows = [["position", "examined", "click rate, random order"],
            ["1", "1.00", "0.308"], ["2", "0.57", "-"], ["5", "0.28", "-"],
            ["10", "0.16", "-"], ["20", "0.09", "0.026"]]
    b.table(40, 140, [110, 140, 240], rows, "blue", size=13, row_h=32)

    b.group(560, 95, 560, 240, "Ranker quality on fresh slates (NDCG at 5)", "green")
    rows = [["trained on", "NDCG@5"],
            ["the old ranker itself", "0.7176"],
            ["clicks at face value", "0.9229"],
            ["clicks weighted by 1 / P(examined)", "0.9331"],
            ["clicks from randomly ordered slates", "0.9776"],
            ["perfect ordering", "1.0000"]]
    b.table(580, 140, [350, 170], rows, "green", size=13, row_h=32)

    b.card(20, 360, 520, 100, "why the face-value click is biased", ["items near the top get seen, so they get clicked,", "so the model learns the old ranker's taste"], "red", size=12)
    b.card(560, 360, 560, 100, "two ways out", ["weight each click by 1 / P(examined): helps, noisy", "if the propensities are guessed wrong", "log a slice of randomised traffic: cleanest, costs clicks"], "orange", size=12)
    b.card(20, 480, 1100, 90, "the weights here use the true examination model", ["real systems estimate it, from randomised swaps or a click model, and carry that error into the ranker"], "grey", size=12)
    return b


def main(names):
    todo = [k for k in BOARDS if not names or any(n in NAMES[k] for n in names)]
    for key in todo:
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
