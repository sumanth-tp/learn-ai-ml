# MCP agent host

A LangGraph agent that connects to several MCP servers at once and works across
them safely. It ships with three servers of its own:

| Server   | Transport       | Offers                                                        |
| -------- | --------------- | ------------------------------------------------------------- |
| notes    | stdio           | notes CRUD, `search`, run-time tools via `list_changed`       |
| calendar | streamable HTTP | events, idempotent `create_event`, sampling in `summarise_day` |
| docs     | streamable HTTP | BM25 `search`, `docs://` resources, a citation prompt         |

The host adds namespacing (`server__tool`), allow/deny policy, human approval for
destructive tools (LangGraph `interrupt`), timeouts with MCP cancellation,
reconnection with backoff, graceful degradation, output truncation, prompt-injection
spotlighting and cross-server blocking, sampling, SQLite persistence, SSE streaming,
OpenTelemetry spans for every MCP call, and a tool-selection eval.

## Setup

```bash
uv sync                      # Python 3.12, installs the locked dependencies
cp .env.example .env         # optional; set OPENAI_API_KEY for the real model
```

## Run

```bash
make demo        # spawns the HTTP servers, runs a scripted five-turn session
make servers     # terminal 1: calendar on :8101, docs on :8102
make run         # terminal 2: API + web UI on http://127.0.0.1:8000
make chat        # or a terminal chat instead of the web UI
make up          # everything in Docker (host, calendar, docs)
```

Without `OPENAI_API_KEY` the host uses an offline keyword-router model, so every
command works without a key or network access to a provider.

## Test and evaluate

```bash
make test        # 53 tests, offline, about 4 seconds
make lint
make eval        # tool-selection accuracy with a regression gate
```

## API

| Method | Path                         | Purpose                               |
| ------ | ---------------------------- | ------------------------------------- |
| GET    | `/healthz`                   | ok / degraded, and which servers      |
| GET    | `/servers`                   | status, tools, collisions             |
| GET    | `/prompts`                   | prompts offered by every server       |
| GET    | `/threads`, `/threads/{id}`  | conversations and history             |
| DELETE | `/threads/{id}`              | erase a conversation                  |
| POST   | `/threads/{id}/messages`     | `{"text": ...}`, SSE stream           |
| POST   | `/threads/{id}/prompts`      | start a turn from an MCP prompt       |
| POST   | `/threads/{id}/approval`     | `{"approve": [tool_call_ids]}`, SSE   |

Set `HOST_API_KEY` to require `Authorization: Bearer <key>`.
