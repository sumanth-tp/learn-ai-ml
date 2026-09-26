.PHONY: install lint format test run demo e2e baseline experiments noise serve up down docker clean

install:            ## install dependencies (dev included) with uv
	uv sync

lint:               ## ruff lint + format check + judge prompt lock
	uv run ruff check .
	uv run ruff format --check .
	uv run ragate judge-lock --check

format:
	uv run ruff format .
	uv run ruff check --fix .

test:               ## offline test suite (no keys, no network)
	uv run pytest -q

e2e:                ## the whole gate offline: index, dataset checks, eval, compare, report
	uv run ragate e2e

run: e2e            ## alias: run the system end to end

baseline:           ## re-baseline (do this on main after a promote, or after a judge change)
	uv run ragate baseline --force

experiments:        ## chunk size / k / reranker / model grid
	uv run ragate experiments

noise:              ## measure judge noise (use real keys for a meaningful band)
	uv run ragate noise --repeats 5

serve:              ## API + dashboard on http://127.0.0.1:8000
	uv run ragate serve

demo:               ## real models: needs OPENAI_API_KEY in .env
	RAGATE_PROVIDER=openai RAGATE_JUDGE_PROVIDER=openai RAGATE_EMBEDDING_PROVIDER=openai \
	RAGATE_STATE_DIR=.ragate-demo uv run ragate eval --out reports/demo-run.json --judge-backend deepeval

up:                 ## docker compose: run the gate, then serve the dashboard
	docker compose up --build

down:
	docker compose down

docker:
	docker build -t ragate:local .

clean:
	rm -rf .ragate .ragate-demo reports .pytest_cache .ruff_cache
