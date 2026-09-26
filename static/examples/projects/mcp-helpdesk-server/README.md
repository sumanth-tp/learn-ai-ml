# Helpdesk MCP server

A production-grade remote MCP server for an internal IT helpdesk, built on
FastMCP 4 and the MCP Python SDK 2. Tools, resources and prompts; Streamable
HTTP with JWT auth plus a stdio mode; multi-tenant isolation and roles;
confirmation for destructive actions; async SQLAlchemy with Alembic
migrations; pagination, idempotency, rate limits, timeouts; structured logs,
Prometheus metrics and health probes.

## Quick start (offline, no keys)

```bash
uv sync                 # Python 3.12, installs from uv.lock
make test               # 84 tests, no network
make demo               # full scenario over real HTTP on a random port
```

## Run it

```bash
cp .env.example .env
make run                # migrate + seed + serve on http://127.0.0.1:8000/mcp
make token USER_ID=sam TENANT=acme ROLE=agent   # dev bearer token
```

Everything in containers, Postgres included:

```bash
docker compose up --build            # set HELPDESK_HOST_PORT=8123 if 8000 is taken
```

stdio for Claude Desktop: see `deploy/claude_desktop_config.json`.

## Test with the MCP Inspector

```bash
make run                             # terminal 1
make token USER_ID=sam ROLE=agent    # copy the token
npx @modelcontextprotocol/inspector  # terminal 2, opens a browser
```

In the Inspector choose transport **Streamable HTTP**, URL
`http://127.0.0.1:8000/mcp`, and add the header
`Authorization: Bearer <token>`. List tools, call `search_tickets`, read
`helpdesk://tickets/1`, get the `triage_ticket` prompt.

## Commands

| Command | What it does |
| --- | --- |
| `make install` | `uv sync --frozen` |
| `make lint` | ruff check + format check |
| `make test` | pytest, fully offline |
| `make run` / `make run-stdio` | serve over HTTP / stdio |
| `make demo` | end-to-end scenario, 4 users, real HTTP |
| `make eval` / `make eval-live` | triage eval and gate (fake / real model) |
| `make schema-snapshot` | accept current tool schemas as the contract |
| `make up` / `make down` | docker compose stack |

## Real model

```bash
export HELPDESK_LLM_PROVIDER=openai OPENAI_API_KEY=sk-...
make eval-live && make demo
```

Optional LangSmith tracing: `LANGSMITH_TRACING=true LANGSMITH_API_KEY=...`.

## Layout

`src/helpdesk_mcp/server.py` wires everything; `repository.py` is the only
module that talks SQL and applies tenant scoping; `middleware.py` holds
observability, rate limiting, role visibility and timeouts; `confirm.py`
handles elicitation across protocol versions; `migrations/` is Alembic.
