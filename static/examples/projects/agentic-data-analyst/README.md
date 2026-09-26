# Agentic data analyst

A LangGraph agent that answers business questions over a DuckDB warehouse: it retrieves
the relevant tables, plans, writes SQL with structured output, validates it with sqlglot,
estimates its cost, asks a human before expensive queries, runs it read-only with a
timeout and row cap, corrects itself on errors, explains the result and draws a chart in
a sandboxed subprocess. It remembers follow-ups, caches question-to-SQL pairs, persists
every step with a checkpointer (time travel included), streams progress, and is scored by
an execution-accuracy eval with a regression gate.

Everything runs offline by default (a deterministic scripted model and hashing
embeddings). Set `ANALYST_LLM_MODE=real` and `OPENAI_API_KEY` to use a real model.

## Quick start

```bash
make demo          # uv sync, seed the warehouse, guided tour, eval gate
make test          # 116 tests, no keys, no network
make run           # API + web UI on http://127.0.0.1:8000
docker compose up --build   # the same, in a hardened container
```

## CLI

```bash
uv run analyst seed [--force]
uv run analyst ask "Total revenue by year" --thread t1
uv run analyst ask "now only for 2024" --thread t1          # follow-up with memory
uv run analyst ask "List every product paired with every customer" --auto-reject
uv run analyst chat                                         # interactive, prompts for approval
uv run analyst history t1                                   # every checkpoint of a thread
uv run analyst replay t1 <checkpoint-id> --sql "SELECT ..." # time travel with a fix
uv run analyst eval                                         # golden set + regression gate
uv run analyst eval --refresh                               # recompute expected results
uv run analyst eval --update-baseline                       # accept a new baseline
uv run --env-file .env analyst --real ask "..."             # real model
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Web UI |
| GET | `/healthz`, `/readyz` | Liveness, readiness (warehouse reachable) |
| GET | `/metrics` | Prometheus metrics |
| POST | `/v1/threads/{id}/ask` | `{"question": ...}` → Server-Sent Events |
| POST | `/v1/threads/{id}/approval` | `{"approved": true, "reviewer": "..."}` → SSE |
| GET | `/v1/threads/{id}` | Current state and conversation history |
| GET | `/v1/threads/{id}/history` | Checkpoints, newest first |
| POST | `/v1/threads/{id}/fork` | `{"checkpoint_id": ..., "sql": ...}` replay or fork |

Set `ANALYST_API_KEY` to require an `X-API-Key` header.

## Configuration

All settings are environment variables with the `ANALYST_` prefix; see `.env.example` and
`src/data_analyst/config.py`. Provider keys (`OPENAI_API_KEY`) and LangSmith variables
(`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`) keep their standard names.

## Layout

```
src/data_analyst/
  config.py            settings
  warehouse/catalog.py semantic layer: tables, PII flags, joins, metrics
  warehouse/seed.py    deterministic sample data, masked views
  validator.py         sqlglot policy checks, LIMIT enforcement
  executor.py          read-only DuckDB, timeout, row cap, EXPLAIN cost, DLP
  retrieval.py         schema retrieval, semantic cache
  llm.py               model interfaces: real (LangChain) and offline
  prompts.py, schemas.py
  sandbox/             chart code gate, subprocess runner with audit hook
  graph/               LangGraph state, nodes, builder
  service.py           graph + checkpointer + events + time travel
  evals.py             execution accuracy, regression gate
  cli.py, api/         entry points
evals/golden.jsonl     question / SQL / expected result
evals/baseline.json    regression thresholds
tests/                 unit, integration, API, CLI, regression
```
