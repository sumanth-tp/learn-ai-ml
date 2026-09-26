.PHONY: install test lint format run demo workers redteam regress docker-build docker-up clean

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check . --fix

run:
	uv run agentmon serve

workers:
	uv run agentmon workers

demo:
	uv run agentmon demo

redteam:
	uv run agentmon redteam

regress:
	uv run agentmon regress

docker-build:
	docker build -t agentmon:local .

docker-up:
	docker compose up --build

clean:
	rm -rf .venv .pytest_cache .ruff_cache out var
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
