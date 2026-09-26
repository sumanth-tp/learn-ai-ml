# mcp-gateway

An enterprise MCP gateway: one authenticated, policy-enforcing front door that
aggregates many upstream MCP servers (stdio and streamable HTTP) for many users.

Features: JWT identity, YAML policy-as-code with argument constraints, a secret
broker that injects per-upstream credentials (no token passthrough), a
hash-chained audit log with PII redaction, per-user and per-tool rate limits and
daily quotas, tool pinning (rug-pull detection), prompt-injection scanning of
descriptions and outputs, output size caps, a TTL cache for read-only tools, a
circuit breaker per upstream, health/readiness/metrics endpoints and an admin CLI.

## Quick start (offline, no API keys)

```bash
make install        # uv sync + copy .env.example to .env
make test           # 91 tests, fully offline
make demo           # starts the real stack and walks through allow/deny/rug-pull
make stack          # run upstreams + gateway on :8080 until Ctrl+C
make up             # the same with docker compose
```

Connect a client:

```bash
TOKEN=$(uv run mcp-gateway-admin mint-token --sub ana --groups employees)
# then use http://127.0.0.1:8080/mcp with "Authorization: Bearer $TOKEN"
```

## Admin CLI

```bash
uv run mcp-gateway-admin upstreams --probe
uv run mcp-gateway-admin policies
uv run mcp-gateway-admin check --user sam --groups support --tool payments_refund \
  --args '{"amount": 500, "currency": "EUR", "idempotency_key": "k-12345678"}'
uv run mcp-gateway-admin denials --limit 20
uv run mcp-gateway-admin pins [--approve TOOL]
uv run mcp-gateway-admin verify-audit
```

## Layout

- `src/mcp_gateway/gateway.py` assembles FastMCP, providers, middleware and routes
- `src/mcp_gateway/middleware.py` the enforcement pipeline
- `src/mcp_gateway/policy.py` the policy engine; `config/policy.yaml` the rules
- `src/mcp_gateway/upstreams.py` upstream specs and credentialed client factories
- `src/mcp_gateway/demo_upstreams/` docs (stdio), payments and tickets (HTTP)
- `evals/` labelled injection cases and golden policy decisions (`make eval`)

Configuration is by environment variables (`GATEWAY_*`); see `.env.example`.
