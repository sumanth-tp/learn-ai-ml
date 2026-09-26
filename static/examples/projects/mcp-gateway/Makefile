.PHONY: install test lint format eval run stack demo demo-llm up down admin docker clean

install:            ## create .venv with runtime + dev dependencies
	uv sync
	@test -f .env || cp .env.example .env

test:               ## full offline test suite
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check . --fix

eval:               ## scanner recall/FPR gate + policy golden decisions
	uv run python -m mcp_gateway.evals

run: stack          ## alias

stack:              ## upstreams + gateway locally, one command (Ctrl+C stops all)
	uv run mcp-gateway stack

demo:               ## scripted end-to-end walkthrough, offline
	uv run mcp-gateway demo

demo-llm:           ## same, with a real LLM judging new tool descriptions
	uv run mcp-gateway demo --llm

up:                 ## docker compose stack
	docker compose up --build -d
	@echo "gateway: http://localhost:8080/mcp  (make down to stop)"

down:
	docker compose down

admin:              ## e.g. make admin ARGS="denials --limit 10"
	uv run mcp-gateway-admin $(ARGS)

docker:
	docker build -t mcp-gateway:local .

clean:
	rm -rf .venv .pytest_cache .ruff_cache var
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
