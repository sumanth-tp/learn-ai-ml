.PHONY: install test lint format run servers up down demo eval chat docker

install:            ## install the locked dependencies
	uv sync

test:               ## offline test suite: no keys, no network
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check . --fix

servers:            ## run the two HTTP MCP servers in the foreground
	uv run calendar-server & uv run docs-server & wait

run:                ## API + web UI on :8000 (start `make servers` first)
	uv run mcp-host serve

chat:               ## terminal chat (start `make servers` first)
	uv run mcp-host chat

demo:               ## one command: spawn servers, run a scripted session (real model if a key is set)
	uv run mcp-host demo

eval:               ## tool-selection eval with its regression gate
	uv run mcp-host eval

up:                 ## the whole system in containers
	docker compose up --build

down:
	docker compose down

docker:
	docker build -t mcp-agent-host .
