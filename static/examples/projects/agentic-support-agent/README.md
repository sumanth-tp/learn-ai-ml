# support-agent

A production-shaped customer-support agent for a fictional online shop
(Larkspur & Co), built with LangGraph. It looks up orders, opens returns,
issues refunds and answers policy questions. Large refunds pause for a human
reviewer, the same refund can never be paid twice, and every conversation is
durable, streamable, replayable and traced.

Everything runs offline with deterministic fakes (no API keys, no network).
Set `OPENAI_API_KEY` and `FAKE_LLM=false` to use a real model.

## Quick start

```bash
# prerequisites: Python 3.12, uv >= 0.8, (optional) Docker
make install        # uv sync --frozen
make test           # 65 tests, offline
make eval           # offline eval + regression gate
make demo           # scripted walkthrough (real model if OPENAI_API_KEY is set)
make run            # API + console on http://localhost:8000 (SQLite)
make up             # the whole system on Postgres via docker compose
```

Open http://localhost:8000 for the chat and review-queue console.

## Configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FAKE_LLM` | `false` | `true` uses the offline scripted model and local embeddings |
| `LLM_PROVIDER` / `LLM_MODEL` | `openai` / `gpt-4o-mini` | any `init_chat_model` provider |
| `OPENAI_API_KEY` | none | required when `FAKE_LLM=false` and provider is openai |
| `DATABASE_URL` | `sqlite:///./data/support.db` | orders, refunds, threads, approvals |
| `CHECKPOINT_BACKEND` | `sqlite` | `memory`, `sqlite` or `postgres` |
| `POSTGRES_URL` | none | checkpointer and store when backend is postgres |
| `REFUND_APPROVAL_THRESHOLD` | `100` | refunds above this need a reviewer |
| `REFUND_GATEWAY` | `stub` | `stub` or `http` (real provider at `REFUND_API_URL`) |
| `MAX_STEPS` / `MAX_TOKENS_PER_REQUEST` | `8` / `8000` | per-request budget |
| `API_TOKEN` / `REVIEWER_TOKEN` | dev values | service and reviewer bearer tokens |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | off | LangSmith tracing, PII anonymised |

## API

| Method | Path | Who | Purpose |
| --- | --- | --- | --- |
| POST | `/v1/chat` | customer | SSE stream: metadata, token, update, interrupt, message, done |
| GET | `/v1/threads/{id}/history` | owner | messages, summary, pending approvals |
| GET | `/v1/approvals` | reviewer | refunds waiting for a decision |
| POST | `/v1/threads/{id}/resume` | reviewer | approve or reject; 409 if nothing is pending |
| GET | `/v1/threads/{id}/checkpoints` | reviewer | time-travel: list checkpoints |
| POST | `/v1/threads/{id}/replay` | reviewer | re-run from a checkpoint |
| POST | `/v1/threads/{id}/fork` | reviewer | branch from a checkpoint with a new message |
| GET | `/healthz`, `/readyz`, `/metrics` | ops | liveness, readiness, Prometheus |

Customer calls send `Authorization: Bearer $API_TOKEN` and `X-User-Id`.
Reviewer calls send `Authorization: Bearer $REVIEWER_TOKEN` and `X-Reviewer`.

```bash
curl -N -X POST localhost:8000/v1/chat \
  -H 'Authorization: Bearer dev-customer-token' -H 'X-User-Id: cust_001' \
  -H 'Content-Type: application/json' \
  -d '{"message": "I want a refund for ORD-1002", "thread_id": "demo_1"}'
```

## Layout

```text
src/support_agent/
  config.py         settings from env
  db.py, seed.py    SQLAlchemy schema and idempotent seed data
  services/         orders, refunds (idempotent ledger + gateways), FAQ retriever
  tools.py          LangChain tools, ToolNode, retry/permission wrapper
  intent.py         LLM classifier with keyword fallback
  guardrails.py     injection blocking, output scrubbing
  memory.py         LangGraph Store long-term memory
  graph.py          the StateGraph
  runner.py         thread ownership, locking, streaming, review queue, time travel
  persistence.py    checkpointer and store per backend
  tracing.py        LangSmith with PII anonymiser
  api/app.py        FastAPI + SSE, api/static/index.html console
  evaluation.py     offline eval and regression gate
  fakes.py          deterministic offline chat model
evals/              dataset.jsonl, thresholds.json
tests/              unit, integration, API, persistence, eval
```
