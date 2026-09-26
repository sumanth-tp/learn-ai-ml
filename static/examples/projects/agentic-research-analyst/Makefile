.PHONY: install seed test lint fmt run demo demo-live eval docker-build up down clean

Q ?= Should we pilot sodium-ion batteries instead of LFP for our 2027 grid-storage sites?

install:
	uv sync

seed:
	uv run research-analyst seed

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check . --fix

run:
	uv run research-analyst serve --host 0.0.0.0 --port 8000

eval:
	uv run research-analyst eval --out data/eval-results.json

# One command, fully offline: install, seed the index, write a report, run the eval gate.
demo: install seed
	uv run research-analyst run "$(Q)" --out out/report.md
	uv run research-analyst eval --out data/eval-results.json

# Same flow against real providers (needs OPENAI_API_KEY; TAVILY_API_KEY optional).
demo-live: install
	uv run research-analyst --live seed
	uv run research-analyst --live run "$(Q)" --out out/report-live.md

docker-build:
	docker build -t research-analyst:local .

up:
	docker compose up --build -d

down:
	docker compose down

clean:
	rm -rf .venv .pytest_cache .ruff_cache data out
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
