# agentmon: production evaluation and monitoring for an LLM agent

A LangGraph banking assistant ("Penny", four tools over a fake core-banking backend)
wrapped in everything a production team needs to know whether it is still good:

* **Tracing**: OpenTelemetry spans (GenAI semantic conventions) for every run, node, LLM
  call and tool call, exported to SQLite (offline) and optionally to OTLP or LangSmith.
* **Online evaluation**: free heuristics on 100% of traffic, then sampled (random,
  stratified by intent, always on errors, guardrail hits and thumbs-down) LLM judges for
  helpfulness, groundedness and policy, run by async workers with retries, timeouts,
  a dead-letter state and a daily budget (a cascading evaluator).
* **Agent evals**: tool-call accuracy, trajectory matching (exact, in_order, unordered,
  superset, subset), step efficiency, loop detection, task completion.
* **Safety**: a red-team suite (direct and indirect prompt injection, jailbreaks, PII
  exfiltration, toxic requests, scope drift) measured as attack success rate, with and
  without guardrails, plus over-refusal on a benign set.
* **Monitoring**: windowed metrics, threshold, burn-rate and drift (PSI/KL) alerts with
  hysteresis, root-cause triage, and a static HTML dashboard.
* **Feedback loop**: failing and flagged traces go to a review queue; approved ones join
  the golden set that the offline regression gate runs on.

Everything runs offline with deterministic fakes behind the same LangChain interfaces
as the real models.

## Quick start

```bash
uv sync                 # Python 3.12, installs dev tools too
make test               # ~100 tests, no keys, no network
make demo               # simulate a week, regression on day 4; writes out/dashboard.html
make run                # API on http://127.0.0.1:8000 (see /docs)
```

`make demo` prints the alerts as they fire, the root cause (prompt v2 deployed Thursday
09:00), the red-team attack success rate with guardrails off and on, the review-queue
outcome, and the regression gate for v1 (pass) and v2 (fail).

## Real models

```bash
cp .env.example .env
# set AGENTMON_LLM_PROVIDER=openai and OPENAI_API_KEY=sk-...
uv run agentmon chat "What's the balance on ACC-1001?"
```

Any OpenAI-compatible server works with `AGENTMON_LLM_BASE_URL` (for example Ollama at
`http://localhost:11434/v1`). LangSmith tracing: `LANGSMITH_TRACING=true`,
`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`.

## Commands

| Command | What it does |
| --- | --- |
| `agentmon serve` | HTTP API with in-process eval workers |
| `agentmon workers` | eval workers only (for a separate container) |
| `agentmon chat "..."` | one message through the agent, then its evals |
| `agentmon demo` | the simulated week and the whole pipeline |
| `agentmon redteam [--no-guardrails]` | ASR report; exit 1 if the safety gate fails |
| `agentmon regress [--prompt-version v2]` | offline regression gate; exit 1 on failure |
| `agentmon alerts` / `agentmon dashboard` | replay alert rules / write the dashboard |

## Docker

```bash
docker compose up --build                     # api :8000 + worker, shared SQLite volume
docker compose --profile demo run --rm demo   # the simulated week, output in ./out
docker compose --profile tracing up           # adds Jaeger (UI :16686); set
                                              # AGENTMON_OTLP_ENDPOINT=http://jaeger:4318/v1/traces
```

## Layout

See `src/agentmon/`: `agent/` (graph, tools, guardrails, backend), `evals/` (heuristics,
judges, sampling, pipeline, workers, trajectory, regression), `safety/redteam.py`,
`monitoring/` (metrics, drift, alerts, rootcause, dashboard), `feedback/review.py`,
`simulation.py`, `demo.py`, `api.py`, `cli.py`. Data in `data/`, rules in `config/`.
