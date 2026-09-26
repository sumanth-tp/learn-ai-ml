.PHONY: install test lint format run run-stdio migrate seed token demo eval eval-live schema-snapshot inspector docker-build up down

install:          ## install runtime + dev dependencies from the lockfile
	uv sync --frozen

test:             ## full offline test suite (no keys, no network)
	uv run pytest -q

lint:             ## lint and format check
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check . --fix

migrate:          ## apply migrations to HELPDESK_DATABASE_URL
	uv run helpdesk-mcp migrate

seed:             ## demo tenants, KB articles, tickets (idempotent)
	uv run helpdesk-mcp seed

run: migrate seed ## serve Streamable HTTP on http://127.0.0.1:8000/mcp
	uv run helpdesk-mcp serve

run-stdio: migrate seed  ## serve over stdio (what Claude Desktop launches)
	uv run helpdesk-mcp serve --transport stdio

token:            ## print a dev token: make token USER_ID=sam TENANT=acme ROLE=agent
	@uv run helpdesk-mcp mint-token --user $(or $(USER_ID),sam) --tenant $(or $(TENANT),acme) --role $(or $(ROLE),agent)

demo:             ## end-to-end scenario over real HTTP, offline by default
	uv run helpdesk-mcp demo

eval:             ## triage eval + regression gate with the configured model (fake by default)
	uv run helpdesk-mcp eval

eval-live:        ## same eval against the real model (needs OPENAI_API_KEY)
	HELPDESK_LLM_PROVIDER=openai uv run helpdesk-mcp eval

schema-snapshot:  ## accept the current tool schemas as the new contract
	uv run helpdesk-mcp schema-snapshot

inspector:        ## open the MCP Inspector against the local HTTP server
	npx @modelcontextprotocol/inspector

docker-build:
	docker build -t helpdesk-mcp:dev .

up:               ## Postgres + migrations + seed + server, one command
	docker compose up --build

down:
	docker compose down -v
