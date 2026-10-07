"""Infographics for docs/genai/23-capstone.md.

Run from the repo root:

    python3 scripts/infographics/capstone_1.py            # all boards
    python3 scripts/infographics/capstone_1.py repo       # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "capstone"
BOARDS = {}
NAMES = {}


def board(name):
    def deco(fn):
        BOARDS[fn.__name__] = fn
        NAMES[fn.__name__] = name
        return fn

    return deco


@board("research-copilot-architecture")
def architecture():
    b = Board(1200, 720, "Research Copilot: how a question travels", "Offline ingestion on the left, query time on the right, every answer checked and logged")

    b.group(20, 90, 360, 420, "Ingestion (run once, offline)", "orange")
    sources = b.card(40, 130, 150, 120, "5 source kinds", ["pdf", "web page", "csv", "text", "youtube"], "orange", size=11)
    loaders = b.card(215, 130, 145, 50, "loaders.py", ["one Document per page / row"], "orange", size=10)
    chunk = b.card(215, 195, 145, 50, "chunking.py", ["1000 chars, 200 overlap"], "orange", size=10)
    metadata = b.card(40, 275, 320, 62, "every chunk gets metadata", ["source  kind  chunk_id  citation = source#chunk_id"], "yellow", size=11)
    store = b.cylinder(135, 375, 130, 110, "Chroma", ["copilot_db/", "17 chunks for the", "sample library"], "purple", size=10)
    b.arrow(sources.right(), loaders.left())
    b.arrow(loaders.bottom(), chunk.top())
    b.arrow(chunk.bottom(), (285, 275))
    b.arrow(metadata.bottom(), store.top())

    b.group(400, 90, 780, 420, "Query time", "blue")
    question = b.card(420, 130, 120, 60, "question", ["copilot ask  or the app"], "blue", size=11)
    router = b.diamond(640, 160, 150, 70, "router.py", "yellow", size=12)
    b.arrow(question.right(), (565, 160))

    docs = b.card(430, 240, 235, 175, "documents route", ["retrieval.py: MMR + BM25,", "fused by reciprocal rank", "guardrails.py: quarantine", "rag.py: prompt, then", "GroundedAnswer schema", "verify_citations()"], "green", size=10, bullets=True)
    live = b.card(680, 240, 235, 175, "live route", ["agent.py: create_agent", "tools: search_library,", "web_search, calculator", "recursion_limit caps steps", "calculator parses with ast,", "never eval()"], "teal", size=10, bullets=True)
    chat = b.card(930, 240, 230, 175, "chitchat route", ["one plain model call", "no retrieval, no tools"], "grey", size=10)
    b.arrow(router.bottom(), (548, 240), label="documents")
    b.arrow((715, 160), (797, 240), label="live")
    b.arrow(router.right(), (1045, 240), via=[(1045, 160)], label="chitchat")
    b.arrow(store.right(), (430, 330), via=[(400, 330)], dashed=True)

    out = b.card(430, 440, 730, 55, "Answer: text, citations (source + quote), confidence, route, used_web", [], "purple", size=11)
    b.arrow(docs.bottom(), (548, 440))
    b.arrow(live.bottom(), (797, 440))
    b.arrow(chat.bottom(), (1045, 440))

    b.group(20, 530, 1160, 170, "After every answer", "grey")
    b.card(40, 570, 330, 100, "observability.py", ["logs/queries.jsonl: question, route,", "citations, tokens, cost, latency"], "grey", size=11)
    b.card(400, 570, 360, 100, "evaluation.py + eval/golden_set.jsonl", ["30 questions, 5 unanswerable:", "refusals, faithfulness, relevancy,", "context precision and recall"], "teal", size=11)
    b.card(790, 570, 370, 100, "tests/  (75 offline tests)", ["pytest -q runs with no API key and with", "deprecation warnings turned into errors"], "green", size=11)
    return b


@board("research-copilot-repo-map")
def repo_map():
    b = Board(1200, 760, "The repository, file by file", "Which file you write in which milestone, and what it is for")
    rows = [
        ["M", "File", "What it does"],
        ["1", "ingestion/loaders.py", "pdf, web, csv, txt, youtube -> Document objects"],
        ["1", "ingestion/chunking.py", "split text; add source, kind, chunk_id, citation"],
        ["2", "index.py", "build, reload and list the Chroma collection"],
        ["3", "prompts.py, guardrails.py, rag.py", "grounded prompt, injection quarantine, answer chain"],
        ["4", "schemas.py, rag.py", "GroundedAnswer vs Answer; verify_citations()"],
        ["5", "retrieval.py", "MMR + BM25, merged by reciprocal rank fusion"],
        ["6", "tools.py", "search_library, web_search, calculator"],
        ["7", "agent.py", "create_agent plus a step limit"],
        ["8", "router.py", "documents, live or chitchat"],
        ["9", "evaluation.py + eval/", "metrics and the 30-question golden set"],
        ["10", "pipeline.py, observability.py, cli.py, app/", "Copilot.ask(), JSONL logs, command line, Streamlit"],
        ["0", "config.py, models.py, offline.py, text_utils.py", "settings, model factory, offline stand-ins, helpers"],
        ["all", "tests/", "75 offline tests, one file per concern"],
    ]
    b.table(30, 100, [60, 360, 700], rows, header_color="blue", size=12, row_h=34)
    b.card(30, 600, 1130, 120, "How to read this", ["Build the files in the order of the milestone column. Each milestone ends with a test you can run:", "pytest -q tests/test_chunking.py after M1, test_retrieval.py after M5, and so on.", "Milestone 0 is shared plumbing that you write once at the start; tests/ grows with every milestone."], "yellow", size=12, align="left")
    return b


@board("research-copilot-api-changes")
def api_changes():
    b = Board(1200, 760, "What broke in the old capstone, and what replaces it", "Checked against langchain 1.4.3, langchain-core 1.6.6, langgraph 1.2.12, ragas 0.4.3, youtube-transcript-api 1.2.4 on 5 October 2026")
    rows = [
        ["Old code", "What happens now", "Replacement used"],
        ["langchain.text_splitter", "ModuleNotFoundError", "langchain_text_splitters"],
        ["langchain.retrievers.EnsembleRetriever", "ModuleNotFoundError (legacy copy in langchain_classic)", "HybridRetriever in retrieval.py"],
        ["create_react_agent, AgentExecutor, hub.pull", "AttributeError (removed)", "langchain.agents.create_agent"],
        ["langchain.schema.runnable", "ModuleNotFoundError", "langchain_core.runnables"],
        ["YouTubeTranscriptApi.get_transcript(id)", "method no longer exists", "YouTubeTranscriptApi().fetch(id)"],
        ["from ragas import evaluate", "ModuleNotFoundError inside ragas", "four metrics in evaluation.py"],
        ["langchain_community loaders and tools", "warns: package is being sunset", "pypdf, bs4, csv, ddgs used directly"],
        ["eval() behind a character whitelist", "still runs, still unsafe", "ast-based safe_eval()"],
        ["one Answer schema with defaults", "fragile with strict structured output", "GroundedAnswer + Answer"],
    ]
    b.table(30, 100, [400, 400, 330], rows, header_color="red", size=12, row_h=40)
    b.card(30, 540, 560, 190, "Why not keep the old imports?", ["langchain 1.x removed those module paths.", "langchain_classic keeps some legacy code,", "but it is a compatibility package, not the", "place new projects should start."], "yellow", size=12, bullets=False, align="left")
    b.card(620, 540, 540, 190, "How this was checked", ["Every old import was tried against the installed", "versions, with warnings enabled.", "Then the new repository ran its 75 tests with", "DeprecationWarning turned into an error."], "green", size=12, bullets=False, align="left")
    return b


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = [name for name in BOARDS if not argv or any(a in NAMES[name] for a in argv)]
    for name in wanted:
        path = OUT / f"{NAMES[name]}.svg"
        BOARDS[name]().save(path)
        print("wrote", path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
