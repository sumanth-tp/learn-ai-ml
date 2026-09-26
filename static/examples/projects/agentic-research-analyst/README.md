# research-analyst

A multi-agent research analyst built on LangGraph. It plans a research question
into sub-questions, fans out parallel researcher subgraphs (Corrective RAG over an
internal index with web fallback), writes a report, runs a critic/revision loop,
checks every claim against its cited source (Self-RAG style) and returns a cited
Markdown + JSON report.

Everything runs offline by default: the LLM, embeddings and web search are
deterministic fakes behind the same interfaces as the real providers.

## Quick start (offline, no keys)

```bash
uv sync
make demo          # seed index -> write a report to out/report.md -> eval gate
make test          # 57 tests, no network
make lint
```

## Real providers

```bash
cp .env.example .env
# set OPENAI_API_KEY (and optionally TAVILY_API_KEY), then:
make demo-live
# or: uv run research-analyst --live run "your question" --out out/r.md
```

Without `TAVILY_API_KEY`, live mode uses the bundled stub web corpus and logs a warning.

## Commands

| Command | What it does |
| --- | --- |
| `research-analyst seed [--force]` | Build the internal vector index from `corpus/internal/*.md` |
| `research-analyst run "Q" [--thread-id ID] [--out f.md]` | Research a question, stream progress to stderr |
| `research-analyst resume ID` | Resume an interrupted run from its SQLite checkpoint |
| `research-analyst status ID` | Show whether a run is complete or where it stopped |
| `research-analyst delete ID` | Erase every checkpoint of one run |
| `research-analyst purge --days 30` | Delete runs older than the retention window |
| `research-analyst eval` | Run the eval set and the regression gate (exit 1 on regression) |
| `research-analyst serve` | Start the FastAPI server on :8000 |

Add `--live` before the sub-command to use real providers.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/healthz` | Liveness |
| POST | `/v1/reports` | Run to completion, return the report JSON (idempotent per `thread_id`) |
| POST | `/v1/reports/stream` | Server-sent events: progress, then `event: report` |
| GET | `/v1/reports/{thread_id}` | Status and report |
| POST | `/v1/reports/{thread_id}/resume` | Resume an interrupted run |
| DELETE | `/v1/reports/{thread_id}` | Erase a run |

Set `RA_API_KEY` to require an `X-API-Key` header.

## Docker

```bash
make docker-build
docker compose up --build -d      # API on :8000, data in the analyst-data volume
docker compose --profile jobs run --rm eval
```

## Tracing

Set `RA_LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...`. Every run is tagged
with the mode and version, and carries the thread id and question as metadata.

## Data

`src/research_analyst/corpus/` holds a small **synthetic** corpus: fictional
internal memos and a stub "web" of fictional `.example` domains, including a
syndicated duplicate, a content-farm page and a prompt-injection document to
exercise dedup, quality scoring and the injection tripwire. Replace
`corpus/internal/` with your own markdown and run `research-analyst seed --force`.
