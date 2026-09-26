# Offline by default: FAKE_LLM=true unless you export OPENAI_API_KEY.
OFFLINE := $(if $(OPENAI_API_KEY),,--offline)

.PHONY: install test lint format run demo eval eval-live chat seed up down logs docker-build clean

install:            ## create .venv and install all dependencies from uv.lock
	uv sync --frozen

test:               ## unit + integration tests, fully offline
	uv run pytest -q

lint:               ## ruff lint and format check (what CI runs)
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

seed:
	uv run support-agent seed $(OFFLINE)

run:                ## API + console on http://localhost:8000 (SQLite persistence)
	uv run support-agent serve $(OFFLINE)

chat:               ## chat in the terminal as cust_001
	uv run support-agent chat $(OFFLINE)

demo:               ## scripted end-to-end walkthrough (real model if OPENAI_API_KEY is set)
	uv run support-agent demo $(OFFLINE)

eval:               ## offline eval + regression gate (fails the build below thresholds)
	uv run support-agent eval --offline

eval-live:          ## same dataset against the real model
	uv run support-agent eval

up:                 ## the whole system: Postgres + API, one command
	docker compose up --build -d
	@echo "open http://localhost:8000"

down:
	docker compose down

logs:
	docker compose logs -f app

docker-build:
	docker build -t support-agent:local .

clean:
	rm -rf .venv .pytest_cache .ruff_cache data/*.db data/*.db-*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
