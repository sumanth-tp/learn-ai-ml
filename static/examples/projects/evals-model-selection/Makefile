.PHONY: install test lint format run demo demo-offline serve docker clean

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

# The one command: benchmark -> candidates -> judge -> calibration -> stats -> report, offline.
run demo-offline:
	MODELSEL_PROFILE=offline MODELSEL_ALLOW_PRIVATE=true uv run modelsel run --include-private

# Real providers: needs OPENAI_API_KEY, ANTHROPIC_API_KEY and a running Ollama.
demo:
	MODELSEL_PROFILE=real uv run modelsel run

serve:
	uv run modelsel serve --port 8000

docker:
	docker build -t modelsel:local .

clean:
	rm -rf .venv .pytest_cache .ruff_cache var reports
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
