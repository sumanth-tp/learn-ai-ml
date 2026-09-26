# modelsel: a model-selection lab for support-ticket triage

Choose the right LLM for a real task with a custom benchmark, a calibrated LLM
judge, and statistics you can defend. Runs fully offline with deterministic fake
models; switch one environment variable to run OpenAI, Anthropic and Ollama.

## Quick start

```bash
uv sync                 # Python 3.12, installs everything
make test               # 60 tests, no keys, no network
make run                # the whole pipeline offline, writes reports/latest.html
```

`make run` does: build the 150-item benchmark (if missing) -> run 5 candidates on
3 tasks -> judge the replies -> calibrate the judge against human labels ->
paired statistics -> contamination checks -> gates, Pareto frontier, weighted
matrix -> Markdown and HTML report -> SQLite run store.

## Real providers

```bash
cp .env.example .env    # add OPENAI_API_KEY, ANTHROPIC_API_KEY
ollama pull llama3.1:8b # for the local candidate
MODELSEL_PROFILE=real uv run modelsel run
```

Before a release decision, also score the held-out private split:

```bash
MODELSEL_ALLOW_PRIVATE=true uv run modelsel run --include-private
```

## Commands

| Command | What it does |
| --- | --- |
| `modelsel build-data` | Regenerate `data/benchmark`, `data/private` and `data/human_labels.jsonl` |
| `modelsel run [--include-private] [--models a,b]` | Full selection run and report |
| `modelsel calibrate` | Judge calibration only; exit code 1 if the judge is not trusted |
| `modelsel sample-size --delta 0.03 --sd 0.15 [--discordant 0.1] [--n 80]` | Items needed / minimum detectable effect |
| `modelsel report RUN_ID [--format md\|html]` | Re-render a stored run |
| `modelsel serve` | HTTP API on :8000 (`POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/report`) |

## Adding a new model

Add an entry under `[models."provider:name"]` in `data/models.toml` with its
prices and RPM, append it to a profile's `candidates`, and re-run. Every old
call is a cache hit, so you pay only for the new model.

## Docker

```bash
docker build -t modelsel:local .
docker compose up api                       # API on :8000
docker compose --profile batch run --rm run # one offline run
```

## Layout

See the project page for a file-by-file walkthrough. `src/modelsel/` holds the
code, `data/` the benchmark and catalogue, `tests/` the offline test suite.
