from pathlib import Path

from board import Board


OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "dm"


def representations():
    board = Board(1160, 470, "Choose the representation for the work", "The same event can be stored for transactions, exchange or analysis")
    board.card(30, 125, 330, 165, "Row-oriented record", ["keep an event's fields together", "good for point reads and updates", "10 GB lecture example"], "blue", size=17)
    board.card(415, 125, 330, 165, "JSON or CSV exchange", ["move records between systems", "schema and types need checking", "nested JSON is semi-structured"], "purple", size=17)
    board.card(800, 125, 330, 165, "Columnar Parquet", ["read selected fields in groups", "compress values of one column", "2 GB lecture example"], "green", size=17)
    board.card(245, 340, 670, 80, "Two of 50 equally sized columns = 4%", ["ideal projection: 2 GB × 0.04 = 0.08 GB; actual reads vary"], "yellow", size=17)
    return board


def quality_dimensions():
    board = Board(1150, 500, "Data quality has six different questions", "The same row can pass one check and fail another")
    entries = [
        ("Accuracy", "Does it match reality?", "blue"),
        ("Completeness", "Are required values present?", "teal"),
        ("Consistency", "Do sources agree?", "purple"),
        ("Timeliness", "Is it fresh enough?", "orange"),
        ("Validity", "Does it obey the rules?", "green"),
        ("Uniqueness", "Should this key repeat?", "yellow"),
    ]
    for index, (title, detail, colour) in enumerate(entries):
        x = 30 + (index % 3) * 375
        y = 120 + (index // 3) * 145
        board.card(x, y, 340, 105, title, [detail], colour, size=17)
    board.card(180, 415, 790, 70, "950 of 1,000 values present = 95% completeness; a 99% rule fails", [], "blue", size=14)
    return board


def storage_architectures():
    board = Board(1160, 520, "Store and refine data for its consumers", "Warehouse, lake and lakehouse are design choices, not a quality ranking")
    board.card(30, 125, 345, 135, "Warehouse", ["curated tables", "SQL and business reports", "schema checked before loading"], "blue", size=17)
    board.card(405, 125, 345, 135, "Lake", ["raw and varied files", "flexible analysis", "needs catalogue and governance"], "purple", size=17)
    board.card(780, 125, 345, 135, "Lakehouse", ["files with table metadata", "snapshots and transactions", "SQL plus ML workloads"], "green", size=17)
    bronze = board.card(115, 345, 260, 100, "Bronze", ["retained source data"], "orange")
    silver = board.card(450, 345, 260, 100, "Silver", ["validated detail"], "teal")
    gold = board.card(785, 345, 260, 100, "Gold", ["curated product"], "yellow")
    board.arrow(bronze.right(), silver.left())
    board.arrow(silver.right(), gold.left())
    return board


def pipeline_flow():
    board = Board(1160, 480, "A reliable pipeline is a dependency graph", "The source's arrival-rate example describes average work in flight")
    source = board.card(30, 125, 245, 100, "Source events", ["100 events each second"], "blue")
    extract = board.card(325, 125, 245, 100, "Extract and check", ["stable event IDs", "reject bad records"], "teal")
    transform = board.card(620, 125, 245, 100, "Transform", ["idempotent output", "partition by time"], "purple")
    publish = board.card(915, 125, 215, 100, "Publish", ["consumer contract"], "green")
    for left, right in [(source, extract), (extract, transform), (transform, publish)]:
        board.arrow(left.right(), right.left())
    board.card(225, 315, 710, 100, "Little's Law: L = λW", ["100 events/s × 0.2 s = 20 events in flight", "stable long-run averages required"], "yellow", size=17)
    return board


def dataops_reliability():
    board = Board(1160, 485, "Operate data as a service", "Reproducible infrastructure and a measurable consumer promise")
    board.card(30, 130, 335, 160, "Define", ["versioned infrastructure plan", "pipeline code and data contract", "review changes before apply"], "blue", size=17)
    board.card(410, 130, 335, 160, "Verify", ["tests on representative data", "replay and failure exercise", "observe freshness and errors"], "purple", size=17)
    board.card(790, 130, 335, 160, "Respond", ["owner and recovery runbook", "consumer status", "correct and backfill"], "green", size=17)
    board.card(230, 340, 700, 90, "A 365-day, 99.9% availability target", ["8.76 hours equivalent downtime; 99.99% allows 0.876 hours"], "yellow", size=17)
    return board


def ml_lifecycle():
    board = Board(1160, 510, "ML delivery is an iterative loop", "Hold the test set aside before using validation feedback to choose a model")
    entries = [
        ("Frame", "business outcome", "blue"),
        ("Understand", "source and labels", "teal"),
        ("Prepare", "time-safe features", "purple"),
        ("Model", "train on one split", "orange"),
        ("Evaluate", "validate, then test", "green"),
        ("Deploy", "watch real outcomes", "yellow"),
    ]
    for index, (title, detail, colour) in enumerate(entries):
        x = 30 + index * 188
        board.card(x, 145, 170, 125, title, [detail], colour, size=16)
        if index:
            board.arrow((x - 18, 205), (x, 205))
    board.card(235, 355, 690, 95, "10,000 examples at 70 / 15 / 15", ["7,000 train · 1,500 validation · 1,500 test", "5-fold CV holds out 2,000 per fold"], "blue", size=17)
    return board


def ingestion_modes():
    board = Board(1160, 500, "Bring source changes in with identity", "Batch, stream and CDC differ in how change is observed")
    board.card(30, 130, 335, 155, "Batch", ["bounded file or query", "schedule and partition", "good for replay"], "blue", size=17)
    board.card(410, 130, 335, 155, "Event stream", ["producer sends events", "low-latency processing", "ordering and retry policy"], "purple", size=17)
    board.card(790, 130, 335, 155, "Database CDC", ["read committed changes", "insert · update · delete", "retain source position"], "green", size=17)
    board.card(225, 350, 710, 90, "Lecture sample", ["5,000 of 2,000,000 rows = 0.0025 = 0.25%", "selection method determines representativeness"], "yellow", size=17)
    return board


def profiling_validation():
    board = Board(1160, 500, "Profile, validate, then investigate change", "One rule failure and one drift score answer different questions")
    profile = board.card(30, 140, 335, 140, "Profile", ["types · nulls · ranges", "10,000 rows; 1,500 nulls", "15% missing"], "blue", size=17)
    validate = board.card(410, 140, 335, 140, "Validate", ["allowed null rate ≤ 5%", "15% exceeds threshold", "rule fails"], "orange", size=17)
    drift = board.card(790, 140, 335, 140, "Monitor shift", ["compare fixed reference bins", "PSI needs context", "investigate before retraining"], "purple", size=17)
    board.arrow(profile.right(), validate.left())
    board.arrow(validate.right(), drift.left())
    board.card(225, 355, 710, 90, "A failed rule needs an action", ["quarantine or block according to consumer cost", "retain the failed rows and an owner"], "green", size=17)
    return board


def analytics_engineering():
    board = Board(1160, 510, "Analytical meaning is built in layers", "A Type 2 customer dimension preserves each historical address version")
    board.card(30, 130, 335, 115, "Stage", ["rename and type source fields", "keep source identity"], "blue")
    board.card(410, 130, 335, 115, "Intermediate", ["join and validate grain", "reusable business entities"], "purple")
    board.card(790, 130, 335, 115, "Mart or metric", ["facts, dimensions, definitions", "consumer-ready output"], "green")
    for x in (365, 745):
        board.arrow((x, 187), (x + 45, 187))
    board.card(185, 335, 790, 105, "Type 2 history: original + three address changes", ["four rows with non-overlapping validity windows", "join by customer ID and event time"], "yellow", size=17)
    return board


def feature_preparation():
    board = Board(1160, 455, "Prepare a feature with training-only statistics", "The lecture's standardisation example")
    value = board.card(50, 145, 245, 110, "Input", ["x = 80"], "blue", size=20)
    stats = board.card(355, 145, 330, 110, "Training statistics", ["mean μ = 70", "standard deviation σ = 5"], "purple", size=18)
    output = board.card(745, 145, 365, 110, "Standardised", ["z = (80 − 70) / 5", "= 2.0"], "green", size=19)
    board.arrow(value.right(), stats.left())
    board.arrow(stats.right(), output.left())
    board.card(235, 325, 690, 85, "Reuse μ and σ at validation and serving", ["fitting again on later data changes feature meaning"], "yellow", size=17)
    return board


def feature_time():
    board = Board(1160, 485, "Historical features need an as-of join", "A day-4 prediction can use the day-3 value, not the day-5 value")
    board.card(35, 145, 250, 115, "Day 1", ["value 4", "historical"], "blue")
    board.card(325, 145, 250, 115, "Day 3", ["value 8", "as-of day 4"], "green")
    board.card(615, 145, 250, 115, "Day 4", ["prediction cutoff", "use value 8"], "yellow")
    board.card(905, 145, 220, 115, "Day 5", ["value 12", "future leakage"], "orange")
    board.card(235, 340, 690, 85, "Offline history and online latest differ", ["apply the same entity key, transformation and freshness policy"], "purple", size=17)
    return board


def orchestration():
    board = Board(1160, 500, "Orchestrate dependencies and recovery", "A schedule starts a run; tasks still need safe retry and publish behaviour")
    source = board.card(30, 130, 245, 100, "Hourly trigger", ["cron 0 * * * *", "24 runs per UTC day"], "blue")
    wait = board.card(325, 130, 245, 100, "Wait for source", ["sensor or event", "bounded timeout"], "teal")
    build = board.card(620, 130, 245, 100, "Build and test", ["idempotent task", "record logical date"], "purple")
    publish = board.card(915, 130, 215, 100, "Publish", ["approved output"], "green")
    for left, right in [(source, wait), (wait, build), (build, publish)]:
        board.arrow(left.right(), right.left())
    board.card(230, 330, 700, 95, "Exponential retry example", ["base 2 s → waits 2, 4, 8 s", "14 s total before execution and queue time"], "yellow", size=17)
    return board


def experiment_metadata():
    board = Board(1160, 510, "A score needs its provenance", "Run 2 has the highest reported F1, but evaluation design still needs review")
    board.card(30, 130, 335, 135, "Run 1", ["F1 = 0.71", "code · data · config"], "blue", size=18)
    board.card(410, 130, 335, 135, "Run 2", ["F1 = 0.76", "highest recorded score"], "green", size=18)
    board.card(790, 130, 335, 135, "Run 3", ["F1 = 0.74", "different candidate"], "purple", size=18)
    board.card(170, 345, 365, 105, "Lineage", ["source → dataset → feature", "→ run → model version"], "teal", size=17)
    board.card(625, 345, 365, 105, "Release decision", ["leakage · uncertainty · latency", "review before promotion"], "yellow", size=17)
    return board


def distributed_processing():
    board = Board(1160, 530, "Partition locally, shuffle only when needed", "A grouping key can make one partition much heavier than its neighbours")
    source = board.card(30, 135, 245, 115, "10 GiB input", ["128 MiB target", "80 size-based pieces"], "blue")
    narrow = board.card(325, 135, 245, 115, "Map or filter", ["narrow transformation", "mostly local work"], "teal")
    shuffle = board.card(620, 135, 245, 115, "Group by key", ["shuffle records", "network and disk"], "orange")
    reduce = board.card(915, 135, 215, 115, "Aggregate", ["one result per key"], "green")
    for left, right in [(source, narrow), (narrow, shuffle), (shuffle, reduce)]:
        board.arrow(left.right(), right.left())
    board.card(220, 345, 720, 110, "Skew can dominate elapsed time", ["one key with 50% of data holds 5 GiB", "80 tasks do not mean 80 concurrent workers"], "yellow", size=17)
    return board


def llm_pipelines():
    board = Board(1160, 525, "A RAG knowledge base has a data lifecycle", "A vector score selects candidates only under a defined retrieval policy")
    load = board.card(30, 135, 245, 110, "Load and clean", ["source ID and version", "deduplicate and permit"], "blue")
    chunk = board.card(325, 135, 245, 110, "Chunk", ["stable chunk IDs", "context and overlap"], "teal")
    embed = board.card(620, 135, 245, 110, "Embed and index", ["model version", "vector plus metadata"], "purple")
    search = board.card(915, 135, 215, 110, "Retrieve", ["filter, rank, cite"], "green")
    for left, right in [(load, chunk), (chunk, embed), (embed, search)]:
        board.arrow(left.right(), right.left())
    board.card(210, 340, 740, 115, "Lecture cosine example", ["q = [1,0,1,1], d = [1,1,1,0]", "dot = 2; norms = √3 each; cosine = 2/3 = 0.667"], "yellow", size=17)
    return board


def privacy_governance():
    board = Board(1160, 530, "Protect data through its full lifecycle", "A group size is one privacy property, not a complete release decision")
    board.card(30, 135, 335, 145, "Collect", ["purpose and minimisation", "owner and lawful basis", "access policy"], "blue", size=17)
    board.card(410, 135, 335, 145, "Transform", ["mask or tokenise", "encrypt and retain safely", "audit use"], "purple", size=17)
    board.card(790, 135, 335, 145, "Release", ["test linkage risk", "review recipients", "track deletion"], "green", size=17)
    board.card(125, 355, 420, 105, "Smallest age/ZIP group = 4", ["k = 4", "1/4 is a uniform-guess illustration"], "yellow", size=17)
    board.card(615, 355, 420, 105, "Differential privacy", ["define unit and sensitivity", "choose mechanism and ε budget"], "teal", size=17)
    return board


def data_observability():
    board = Board(1160, 530, "Observe data where consumers rely on it", "Five useful dimensions support diagnosis; the response follows the data contract")
    entries = [
        ("Freshness", "source and publish age", "blue"),
        ("Volume", "expected row count", "teal"),
        ("Schema", "fields and types", "purple"),
        ("Distribution", "nulls and value mix", "orange"),
        ("Lineage", "upstream and impact", "green"),
    ]
    for index, (title, detail, colour) in enumerate(entries):
        board.card(30 + index * 226, 145, 205, 120, title, [detail], colour, size=16)
    board.card(225, 350, 710, 105, "Lecture freshness example", ["approved data age 90 min > 60 min limit", "30 min breach → alert and assess consumer impact"], "yellow", size=17)
    return board


def question_bank():
    board = Board(1160, 520, "Practise the whole data lifecycle", "Twenty-six unique source questions; the comprehensive bank repeats sixteen of them")
    board.card(30, 140, 335, 150, "Foundations", ["architecture · batch and stream", "lifecycle · quality · lineage", "questions 1–10"], "blue", size=16)
    board.card(410, 140, 335, 150, "Production", ["orchestration · features", "privacy · observability", "questions 11–16"], "purple", size=16)
    board.card(790, 140, 335, 150, "Numericals", ["ten worked calculations", "units and assumptions matter", "questions 17–26"], "green", size=16)
    board.card(190, 350, 780, 100, "Check the premise before using a shortcut", ["80 pieces ≠ 80 simultaneous workers; 0.667 cosine ≠ automatic retrieval", "k = 4 ≠ a universal 25% identity-risk bound"], "yellow", size=16)
    return board


def midsem_practice():
    board = Board(1160, 530, "Three scenarios, three design reviews", "The 2026 mid-semester paper connects quality, architecture and consistency")
    board.card(30, 130, 335, 160, "HealthPredict", ["rural share 10% vs 40%", "missing EHR fields 30%", "drift and false negatives"], "blue", size=17)
    board.card(410, 130, 335, 160, "StreamFlow", ["clickstream 500 GB/day", "four-layer data design", "historical feature timing"], "purple", size=17)
    board.card(790, 130, 335, 160, "PayFast", ["sub-second fraud checks", "replay and audit history", "ACID core, eventual views"], "green", size=17)
    board.card(210, 360, 740, 95, "Mark the trade-off and its assumption", ["architecture choice depends on retention, workload and failure cost", "state what is protected and what is eventually consistent"], "yellow", size=16)
    return board


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in [
        ("representations", representations),
        ("quality-dimensions", quality_dimensions),
        ("storage-architectures", storage_architectures),
        ("pipeline-flow", pipeline_flow),
        ("dataops-reliability", dataops_reliability),
        ("ml-lifecycle", ml_lifecycle),
        ("ingestion-modes", ingestion_modes),
        ("profiling-validation", profiling_validation),
        ("analytics-engineering", analytics_engineering),
        ("feature-preparation", feature_preparation),
        ("feature-time", feature_time),
        ("orchestration", orchestration),
        ("experiment-metadata", experiment_metadata),
        ("distributed-processing", distributed_processing),
        ("llm-pipelines", llm_pipelines),
        ("privacy-governance", privacy_governance),
        ("data-observability", data_observability),
        ("question-bank", question_bank),
        ("midsem-practice", midsem_practice),
    ]:
        make().save(OUT / f"{name}.svg")


if __name__ == "__main__":
    main()
