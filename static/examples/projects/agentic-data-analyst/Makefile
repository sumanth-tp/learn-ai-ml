.PHONY: install seed test lint format eval baseline run demo docker-build up down clean

install:  ## install runtime + dev dependencies from the lockfile
	uv sync --frozen

seed:  ## build the sample warehouse (deterministic)
	uv run analyst seed --force

test:  ## unit + integration + regression tests, fully offline
	uv run pytest -q

lint:  ## ruff lint and format check
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

eval:  ## golden-set eval + regression gate (exit 1 on regression)
	uv run analyst eval

baseline:  ## accept the current eval as the new baseline (review the diff!)
	uv run analyst eval --update-baseline

run:  ## API + web UI on http://127.0.0.1:8000
	uv run analyst serve

demo: install  ## one command: seed, guided end-to-end tour, eval gate
	uv run analyst seed
	uv run analyst demo

docker-build:
	docker build -t agentic-data-analyst:local .

up:  ## API + UI in Docker on http://localhost:8000
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf .venv .pytest_cache .ruff_cache data/*.sqlite data/*.png evals/report.json chart.png
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
