"""Infographics for docs/mlops/platform.

Run from the repo root:

    python3 scripts/infographics/plat_1.py                 # all boards
    python3 scripts/infographics/plat_1.py scheduling      # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "plat"
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


def rect(b, x, y, w, h, fill, stroke, width=2, rx=6):
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


@board("plat-ab-testing-power")
def ab_power():
    b = Board(1180, 640, "How many users, and how much CUPED saves", "10% baseline, alpha 0.05, power 0.80, two-sided z-test")
    b.group(20, 95, 560, 330, "Users per arm for each effect you want to see", "blue")
    blue = PALETTE["blue"]
    rows = [("MDE 0.020", 3841), ("MDE 0.010", 14751), ("MDE 0.005", 57763)]
    for i, (name, n) in enumerate(rows):
        y = 150 + i * 70
        raw_text(b, 40, y + 22, name, 13, INK, "start", "700")
        w = 330 * n / 57763
        rect(b, 160, y, max(w, 6), 34, blue["fill"], blue["stroke"])
        raw_text(b, 160 + max(w, 6) + 10, y + 22, f"{n:,}", 14, blue["text"], "start", "700")
    b.card(40, 355, 520, 56, "halve the effect, quadruple the users", ["3,841 -> 14,751 -> 57,763 (about x4 each step)"], "orange", size=12)

    b.group(600, 95, 560, 330, "Does the formula keep its promise? 4,000 simulated tests", "green")
    rows2 = [["check", "result"],
             ["n = 14,751 per arm, true lift +0.01", "power 0.800"],
             ["n = 14,751 per arm, no lift", "false positives 0.051"],
             ["half the traffic, 7,375 per arm", "power 0.495"]]
    b.table(620, 145, [320, 200], rows2, "green", size=13, row_h=44)
    b.card(620, 335, 520, 75, "a coin flip is not a test", ["underpowered tests miss real wins and still look like evidence"], "red", size=12)

    b.group(20, 445, 1140, 175, "CUPED keeps (1 - rho squared) of the variance", "purple")
    cards = [("rho 0.3", "0.91 of variance", "9% fewer users"),
             ("rho 0.5", "0.75 of variance", "25% fewer users"),
             ("rho 0.7", "0.51 of variance", "49% fewer users"),
             ("rho 0.9", "0.19 of variance", "81% fewer users")]
    for i, (a, bb, c) in enumerate(cards):
        b.card(40 + i * 205, 490, 190, 90, a, [bb, c], "purple", size=12)
    b.card(870, 490, 270, 90, "measured at rho 0.7", ["ratio 0.516, theory 0.510", "power 0.383 -> 0.643"], "green", size=12)
    return b


@board("plat-ab-testing-peeking")
def ab_peeking():
    b = Board(1200, 640, "Four ways an experiment goes wrong", "Peeking, a broken split, shared capacity, and what each check says")
    b.group(20, 95, 700, 300, "Twenty looks at a no-effect test (3,000 simulated runs each)", "red")
    rows = [["stopping rule", "false pos.", "power", "mean n"],
            ["fixed horizon, one look", "0.054", "0.799", "2000"],
            ["naive, test at every look", "0.255", "0.889", "733"],
            ["Bonferroni, alpha/20", "0.018", "0.516", "1160"],
            ["O'Brien-Fleming shape", "0.055", "0.784", "1360"],
            ["always-valid mSPRT", "0.020", "0.555", "1109"]]
    b.table(40, 140, [290, 130, 110, 120], rows, "red", size=13, row_h=36)
    b.text(370, 372, "mean n is the average users per arm at which a detected effect stopped", 11, "red", italic=True)

    b.group(740, 95, 440, 300, "Sample ratio mismatch, 50/50 intended", "orange")
    b.card(760, 140, 400, 62, "50,120 vs 49,880", ["p = 0.448: fine"], "green", size=12)
    b.card(760, 212, 400, 62, "10,000 vs 9,650", ["p = 0.0125: borderline, find out why"], "yellow", size=12)
    b.card(760, 284, 400, 62, "50,900 vs 49,100", ["p = 1.25e-08: SRM, do not trust the result"], "red", size=12)
    b.text(960, 372, "flag SRM below p = 0.001, investigate below 0.05", 11, "orange", italic=True)

    b.group(20, 415, 1160, 205, "Interference: 400 markets, 200 users, 50 bookable slots each, treatment lifts appetite by 60%", "purple")
    b.card(45, 470, 330, 90, "user-level A/B test", ["booking lift +0.1105", "treated users take slots from control"], "orange", size=12)
    b.card(425, 470, 330, 90, "everyone treated vs nobody", ["booking lift +0.0500", "the capacity is the cap"], "green", size=12)
    b.card(805, 470, 340, 90, "the test overstates it by 2.2x", ["randomise whole markets", "and accept the lost power"], "red", size=12)
    b.arrow((375, 515), (425, 515))
    b.arrow((755, 515), (805, 515))
    b.text(600, 595, "shared capacity makes users depend on each other's treatment, which breaks the independence the test assumes", 11, "purple", italic=True)
    return b


@board("plat-kubernetes-scheduling")
def k8s_scheduling():
    b = Board(1200, 680, "Why GPUs sit idle", "5 preprocessing pods (3 CPU) arrive, then 6 GPU pods (4 CPU, 1 GPU)")
    b.group(20, 95, 1160, 130, "Cluster", "grey")
    b.card(45, 135, 340, 70, "cpu-1", ["8 CPU, 0 GPU"], "blue", size=12)
    b.card(430, 135, 340, 70, "gpu-a", ["16 CPU, 2 GPU"], "orange", size=12)
    b.card(815, 135, 340, 70, "gpu-b", ["16 CPU, 4 GPU"], "orange", size=12)

    b.group(20, 245, 1160, 300, "Outcomes (block 2)", "teal")
    rows = [["GPU nodes", "strategy", "preprocessing placed", "GPU pods placed", "Pending", "GPUs idle"],
            ["open", "spread", "5 of 5: cpu-1 1, gpu-a 2, gpu-b 2", "4 of 6", "2", "2"],
            ["open", "bin-pack", "5 of 5: cpu-1 2, gpu-a 3", "5 of 6", "1", "1"],
            ["tainted", "spread", "2 of 5: cpu-1 2", "6 of 6", "0", "0"],
            ["tainted", "bin-pack", "2 of 5: cpu-1 2", "6 of 6", "0", "0"]]
    b.table(40, 290, [120, 120, 420, 190, 130, 130], rows, "teal", size=13, row_h=46)
    b.text(600, 530, "tainted rows: the 3 other preprocessing pods wait for CPU nodes", 11, "teal", italic=True)

    b.card(20, 565, 560, 95, "stranding", ["CPU-only pods that land on GPU nodes use the CPU", "that GPU pods also need, so GPUs go unused"], "red", size=12)
    b.card(620, 565, 560, 95, "the cure and its price", ["taint GPU nodes, tolerate in GPU pods", "then size the CPU pool for everything else"], "green", size=12)
    return b


@board("plat-kubernetes-autoscaling")
def k8s_autoscaling():
    b = Board(1200, 720, "The autoscaler formula under a spike", "target 70%, tolerance 0.1, sync every tick, scale-down window 20 ticks, new pods ready after 2 ticks")
    b.card(20, 95, 1160, 70, "desired = ceil( current x metric / target )", ["tick 4: ceil(2 x 3.15 / 0.70) = ceil(9.0) = 9 replicas"], "blue", size=13, title_size=18)

    b.group(20, 185, 760, 330, "Replicas over 44 ticks", "orange")
    x0, y0, w, h = 80, 235, 660, 220
    base = y0 + h
    line(b, x0, base, x0 + w, base)
    line(b, x0, y0, x0, base)
    tick_w = w / 44
    ty = lambda v: base - v / 10 * h
    for v in (2, 9):
        line(b, x0, ty(v), x0 + w, ty(v), "#ced4da", 1, "4 4")
        raw_text(b, x0 - 8, ty(v) + 4, str(v), 12, INK, "end", "700")
    orange, blue = PALETTE["orange"], PALETTE["blue"]
    segs = [(0, 4, 2), (4, 37, 9), (37, 44, 2)]
    for a, z, v in segs:
        line(b, x0 + a * tick_w, ty(v), x0 + z * tick_w, ty(v), orange["stroke"], 3.5)
    for a, z, v in [(0, 6, 2), (6, 38, 9), (38, 44, 2)]:
        line(b, x0 + a * tick_w, ty(v) + 7, x0 + z * tick_w, ty(v) + 7, blue["stroke"], 2.5, "6 4")
    line(b, x0 + 4 * tick_w, y0, x0 + 4 * tick_w, base, "#e03131", 1.4, "3 3")
    line(b, x0 + 18 * tick_w, y0, x0 + 18 * tick_w, base, "#2f9e44", 1.4, "3 3")
    raw_text(b, x0 + 4 * tick_w, y0 - 6, "tick 4: load x4.5", 11, "#c92a2a")
    raw_text(b, x0 + 18 * tick_w + 4, y0 - 6, "tick 18: load drops", 11, "#2b8a3e", "start")
    for t in (0, 10, 20, 30, 40):
        raw_text(b, x0 + t * tick_w, base + 18, str(t), 11, FAINT)
    raw_text(b, x0 + w / 2, base + 36, "tick (one sync period)", 11, FAINT)
    raw_text(b, 150, 285, "solid: replicas", 11, orange["text"], "start", "700")
    raw_text(b, 150, 302, "dashed: ready replicas", 11, blue["text"], "start", "700")

    b.group(800, 185, 380, 330, "What the numbers say", "green")
    b.card(820, 230, 340, 70, "spike at tick 4", ["2 ready pods read 315%", "asks for 9; ready at tick 6"], "green", size=12)
    b.card(820, 315, 340, 70, "load falls at tick 18", ["utilisation 16%", "window still holds the 9s"], "yellow", size=12)
    b.card(820, 400, 340, 70, "back to 2 at tick 37", ["20 ticks after the last 9", "= the 300 s default window"], "orange", size=12)

    b.group(20, 535, 1160, 165, "Jobs: 8 indexes, 3 at a time, 20% pod failure (80% in the last row)", "purple")
    rows = [["backoffLimit", "failure rate", "result", "indexes done", "failed pods", "rounds"],
            ["4", "20%", "Complete", "8 of 8", "3", "4"],
            ["1", "20%", "Failed (BackoffLimitExceeded)", "3 of 8", "3", "2"],
            ["4", "80%", "Failed (BackoffLimitExceeded)", "0 of 8", "6", "2"]]
    b.table(40, 575, [150, 150, 340, 170, 170, 130], rows, "purple", size=12, row_h=24)
    return b


@board("plat-iac-plan-apply")
def iac_plan_apply():
    b = Board(1200, 700, "Plan and apply", "Configuration, state and reality, and the plan the chapter's engine prints")
    b.card(30, 100, 300, 100, "Configuration", ["your files: what you want"], "blue", size=12, title_size=16)
    b.card(450, 100, 300, 100, "State", ["what the tool recorded"], "purple", size=12, title_size=16)
    b.card(870, 100, 300, 100, "Reality", ["what the cloud really holds"], "green", size=12, title_size=16)
    b.arrow((330, 150), (450, 150), label="plan compares", label_dy=-14)
    b.arrow((870, 150), (750, 150), label="refresh reads", label_dy=-14)
    b.text(600, 232, "apply changes reality in dependency order, then records the result in state", 12, "purple", italic=True)

    b.group(20, 255, 700, 300, "Step 3: rename the bucket, resize the endpoint, add an alarm", "teal")
    rows = [["symbol", "address", "change"],
            ["-/+", "bucket.artifacts", "name ml-artifacts-prod -> ml-artifacts-prod-eu"],
            ["~", "endpoint.ranker", "also bucket_name; replicas 3 -> 5; image 1.0 -> 1.1"],
            ["+", "alarm.latency", "new"]]
    b.table(40, 295, [80, 170, 410], rows, "teal", size=12, row_h=40)
    b.card(40, 470, 660, 70, "Plan: 2 to add, 1 to change, 1 to destroy.", ["a replacement counts once as an add and once as a destroy"], "yellow", size=12, title_size=15)

    b.group(740, 255, 440, 300, "Order and idempotence", "orange")
    b.card(760, 300, 400, 54, "wave 1: bucket.artifacts, network.main", [], "orange", size=12)
    b.card(760, 364, 400, 54, "wave 2: cluster.gpu", [], "orange", size=12)
    b.card(760, 428, 400, 54, "wave 3: endpoint.ranker", [], "orange", size=12)
    b.card(760, 492, 400, 46, "second plan, nothing changed: No changes.", [], "green", size=12)

    b.card(20, 575, 1160, 90, "read every -/+ as a deletion", ["a bucket name cannot change in place, so the old bucket and its contents are destroyed; prevent_destroy makes the plan fail instead"], "red", size=12)
    return b


@board("plat-iac-state-drift")
def iac_drift():
    b = Board(1200, 700, "Drift, guards and locking", "A console edit, the lifecycle rules that answer it, and two writers sharing one state")
    b.group(20, 95, 740, 330, "Someone sets node_count to 4 in the console", "orange")
    b.card(40, 140, 340, 70, "refresh-only report", ["cluster.gpu  node_count: 2 -> 4"], "yellow", size=12)
    b.card(400, 140, 340, 70, "normal plan, code says 2", ["~ cluster.gpu  node_count: 4 -> 2", "Plan: 0 to add, 1 to change, 0 to destroy."], "orange", size=12)
    b.card(40, 235, 340, 80, "accept the drift", ["apply the refresh-only plan,", "then change the code to 4"], "green", size=12)
    b.card(400, 235, 340, 80, "overwrite it", ["apply: back to 2", "right for a temporary fix"], "blue", size=12)
    b.card(40, 335, 700, 70, "ignore_changes = [node_count]", ["an autoscaler owns the value: the plan says No changes."], "purple", size=12)

    b.group(780, 95, 400, 330, "prevent_destroy", "red")
    b.card(800, 140, 360, 140, "bucket renamed again", ["plan fails before any apply:", "prevent_destroy blocks the", "replacement forced by name"], "red", size=12)
    b.card(800, 295, 360, 110, "what it does not do", ["it cannot stop you deleting", "the resource block entirely"], "grey", size=12)

    b.group(20, 445, 1160, 235, "Two engineers apply from the same starting state (block 2)", "teal")
    b.card(45, 490, 540, 110, "no lock", ["A writes cluster.gpu, then B writes endpoint.ranker", "resources recorded: endpoint.ranker only, serial 1", "the next plan wants to create cluster.gpu again"], "red", size=12)
    b.card(615, 490, 540, 110, "with a lock file", ["A acquires: True. B while A holds it: False", "B waits, reads A's state, then writes", "resources recorded: both, serial 2"], "green", size=12)
    b.text(600, 650, "a remote backend with locking gives the whole team this guarantee", 12, "teal", italic=True)
    return b


@board("plat-cloud-ml-map")
def cloud_map():
    b = Board(1340, 600, "Where the same job lives on four clouds", "Feature names as the official documentation pages use them (read October 2026); 'not found' means the pages I read did not show it")
    nf = "not found on the pages read"
    rows = [["stage", "SageMaker AI", "Vertex (Agent Platform)", "Azure ML and Foundry", "Bedrock"],
            ["train your own", "training jobs, HyperPod, Autopilot", "custom training, AutoML", "train in the cloud, AutoML, hyperparameter tuning", "model customisation is linked from the overview; not read"],
            ["pipelines and registry", "Model Building Pipelines, Model Registry", "Pipelines (Kubeflow Pipelines, TFX), Model Registry", "pipelines, designer, versioned model registry", nf],
            ["online serving", "real-time, serverless and asynchronous endpoints", "online prediction", "managed online endpoints with traffic split; Foundry model and agent endpoints", "bedrock-runtime APIs (Converse, Invoke, Messages, Responses)"],
            ["batch", "Batch Transform, S3 in and out", "batch inference from Cloud Storage or BigQuery", "batch endpoints, jobs on compute clusters", "batch inference, S3 JSONL, no tool calling"],
            ["monitoring", "Model Monitor: drift and quality", "Model Monitoring: drift and skew", "Event Grid events including data drift; Foundry tracing and evaluations", nf],
            ["generative AI", "JumpStart pretrained models", "Model Garden, more than 200 models; Agent Studio", "model catalog; Foundry: 10,000+ models, Agent Service, content filters", "100+ models, Knowledge Bases, Guardrails, AgentCore"]]
    b.table(20, 100, [150, 245, 265, 345, 305], rows, "blue", size=12)
    b.card(20, 450, 650, 130, "same lifecycle, different names", ["SageMaker AI, Vertex and Azure ML all cover train, register,", "serve, batch and monitor; Bedrock is the managed GenAI layer.", "Which cloud holds your data and identity often decides"], "green", size=12)
    b.card(690, 450, 630, 130, "read the gaps carefully", ["a blank cell is a page I did not find a feature on,", "not a statement that it does not exist", "re-check the product page before deciding"], "orange", size=12)
    return b


@board("plat-cloud-ml-cost-and-names")
def cloud_cost():
    b = Board(1240, 700, "Cost shape and a moving map", "Placeholder rate, so only the shape is meaningful; the dated renames below are from the provider pages")
    b.group(20, 95, 700, 360, "Online or batch? 0.2 s per request, 1 unit per hour", "blue")
    rows = [["requests / month", "instances", "always-on", "pay-per-use", "batch"],
            ["100,000", "1", "730", "11", "6"],
            ["1,000,000", "1", "730", "111", "56"],
            ["10,000,000", "2", "1460", "1111", "556"],
            ["30,000,000", "4", "2920", "3333", "1667"],
            ["100,000,000", "13", "9490", "11111", "5556"]]
    b.table(40, 140, [190, 110, 130, 140, 100], rows, "blue", size=13, row_h=36)
    b.text(370, 395, "pay-per-use costs 2x per busy second here; it matches one always-on instance", 11, "blue", italic=True)
    b.text(370, 413, "at 6,570,000 requests a month (50% busy). Batch has no idle time.", 11, "blue", italic=True)

    b.group(740, 95, 480, 360, "Names and dates (from the provider pages)", "purple")
    b.card(760, 140, 440, 66, "SageMaker", ["renamed Amazon SageMaker AI on 3 December 2024"], "purple", size=12)
    b.card(760, 216, 440, 66, "Vertex AI", ["now Gemini Enterprise Agent Platform", "(Google Cloud blog dated 23 April 2026)"], "purple", size=12)
    b.card(760, 292, 440, 66, "Azure AI Foundry", ["now Microsoft Foundry", "prompt flow retires 20 April 2027"], "purple", size=12)
    b.card(760, 368, 440, 66, "Bedrock Agents", ["now Agents Classic, closed to new customers;", "AgentCore is the pointer"], "purple", size=12)

    b.card(20, 480, 1200, 80, "the lesson", ["a feature checklist rarely separates the big platforms; cost shape, data location, identity and the dated product changes do"], "yellow", size=13)
    b.card(20, 580, 1200, 90, "before you pick one", ["price your own traffic with the provider's current price page,", "read the feature's page for the limits that matter to you,", "and check whether the product is being renamed or retired"], "red", size=12)
    return b


@board("plat-batch-inference-idempotent")
def batch_idempotent():
    b = Board(1200, 700, "A batch job you can run twice", "Seven daily partitions of 2,000 rows each, scored with DuckDB; every number is printed by block 2")
    b.group(20, 95, 560, 215, "Append versus overwrite, run twice", "red")
    b.card(40, 140, 255, 150, "append a new file", ["28,000 rows written", "14,000 distinct ids", "14,000 duplicates"], "red", size=12)
    b.card(305, 140, 255, 150, "overwrite the partition", ["14,000 rows", "14,000 distinct ids", "same checksum both runs"], "green", size=12)

    b.group(600, 95, 580, 215, "Backfill with a ledger, a worker dies on 09-04", "orange")
    rows = [["run", "ran", "skipped", "failed"],
            ["first", "6", "0", "2026-09-04"],
            ["second", "1", "6", "none"],
            ["third", "0", "7", "none"]]
    b.table(620, 140, [110, 90, 120, 220], rows, "orange", size=13, row_h=36)
    b.text(890, 300, "a day is skipped when its input fingerprint matches the ledger", 11, "orange", italic=True)

    b.group(20, 330, 560, 190, "Late data: 300 rows arrive for 09-02", "purple")
    b.card(40, 375, 520, 125, "only 2026-09-02 reruns", ["ran 1 day, skipped 6", "table: 14,300 rows, 14,300 distinct ids", "the other six partitions are untouched"], "purple", size=12)

    b.group(600, 330, 580, 190, "A crash halfway through writing", "teal")
    b.card(620, 375, 265, 125, "temp file, then rename", ["old file stays whole:", "2,000 good rows"], "green", size=12)
    b.card(905, 375, 255, 125, "write in place", ["half a file is left:", "unreadable"], "red", size=12)

    b.card(20, 545, 1160, 120, "the rule that makes all of this work", ["the output path is a function of the input partition, never of the clock or a random name;", "write somewhere private and publish by renaming; record done only after the rename;", "then retries, restarts and backfills are the same operation"], "yellow", size=13)
    return b


@board("plat-batch-inference-window")
def batch_window():
    b = Board(1200, 700, "Making the nightly batch fit", "Block 1 shows what batching and sorting save; block 3 shows what skew costs")
    b.group(20, 95, 560, 240, "Batch size changes calls, not answers (100,000 rows)", "blue")
    rows = [["batch size", "model calls", "identical scores"],
            ["1,000", "100", "True"],
            ["10,000", "10", "True"],
            ["100,000", "1", "True"],
            ["row by row", "100,000", "-"]]
    b.table(40, 140, [170, 170, 190], rows, "blue", size=13, row_h=36)

    b.group(600, 95, 580, 240, "Padding waste, 20,000 sequences, batches of 32", "green")
    blue, green = PALETTE["blue"], PALETTE["green"]
    rect(b, 620, 158, 540 * 0.253, 36, blue["fill"], blue["stroke"])
    rect(b, 620, 158, 540, 36, "none", INK, 1.2, 4)
    raw_text(b, 620, 150, "arrival order: 25.3% useful", 13, blue["text"], "start", "700")
    rect(b, 620, 230, 540 * 0.995, 36, green["fill"], green["stroke"])
    rect(b, 620, 230, 540, 36, "none", INK, 1.2, 4)
    raw_text(b, 620, 222, "sorted by length: 99.5% useful", 13, green["text"], "start", "700")
    raw_text(b, 890, 296, "5,861,600 vs 1,488,128 padded tokens: 74.6% of the compute saved", 12, INK)
    raw_text(b, 890, 315, "keep the row id, sort, score, then restore the order", 11, FAINT)

    b.group(20, 355, 1160, 215, "50 million rows, 2,000 rows/s per worker, 2 hour window, one partition holds 30%", "orange")
    rows2 = [["workers", "as partitioned", "fits", "chunked at 5M rows", "fits"],
             ["2", "3.47 h", "no", "3.47 h", "no"],
             ["4", "2.08 h", "no", "1.74 h", "yes"],
             ["8", "2.08 h", "no", "0.87 h", "yes"]]
    b.table(40, 400, [140, 220, 100, 260, 100], rows2, "orange", size=13, row_h=36)
    b.card(900, 400, 260, 144, "more workers cannot beat the biggest partition", ["8 workers still take 2.08 h", "chunking fixes it"], "red", size=12)

    b.card(20, 590, 1160, 90, "longest-first simulation on 4 workers", ["25 partitions, one with 30%: 2.08 h (83% utilisation); cut into 27 tasks of at most 5M rows: 1.82 h; perfect split 1.74 h"], "yellow", size=12)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or v.endswith(name))
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
