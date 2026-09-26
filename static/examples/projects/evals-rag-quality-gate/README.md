# ragate: RAG evaluation system and CI release gate

An offline evaluation system for a small but real RAG assistant (the fictional Fernhill
Analytics employee handbook), wired into CI as a release gate.

- **App under test:** 15-document handbook, sentence-aware chunking, FAISS + BM25 hybrid
  retrieval (RRF), lexical or LLM reranker, a cited generator with a refusal contract,
  an input guard and PII redaction. FastAPI `/ask` endpoint.
- **Golden set:** 40 human-reviewed items stratified into factoid, multi-hop,
  unanswerable and adversarial; synthetic generation into a review CSV; immutable
  versions with a manifest and sha256; leakage and contamination checks.
- **Metrics:** recall@k, precision@k, hit@k, MRR, nDCG, contextual precision/recall;
  faithfulness, answer relevancy, G-Eval correctness with a rubric (DeepEval), citation
  validity/precision/recall; the RAG triad; refusal and false-refusal rates, PII
  leakage; p50/p95 latency, tokens and cost per query.
- **Judge practice:** versioned prompts with a lock file, temperature 0, SQLite cache,
  measured noise band.
- **Gate:** per-metric direction, tolerance, noise, floors and ceilings; paired
  bootstrap CIs; markdown report; exit 0 promote, 1 block, 2 error.
- **Experiments:** chunk size, k, hybrid, reranker and model swaps in one table.

Everything runs offline with deterministic fakes. Real providers are one env var away.

## Quick start

```bash
uv sync                 # Python 3.12, installs dev tools too
make test               # 84 tests, no keys, no network
make e2e                # index -> dataset checks -> eval -> gate vs baseline -> reports/gate.md
make experiments        # reports/experiments.md
make serve              # http://127.0.0.1:8000 (dashboard), /docs (OpenAPI)
docker compose up --build   # the gate once, then the API over the same run store
```

## With real models

```bash
cp .env.example .env    # set OPENAI_API_KEY
make demo               # real generator, embeddings and judge; DeepEval metric backend
uv run ragate noise --repeats 5        # measure the real judge's noise band
uv run ragate baseline --out baselines/baseline.real.json   # a baseline for the real judge
```

Runs with a different judge are never compared with each other (the gate exits 2).

## Commands

| Command | What it does |
| --- | --- |
| `ragate ingest` | build or reuse the index for a config |
| `ragate ask "..."` | ask one question |
| `ragate dataset check` | quality, contamination and leakage checks |
| `ragate dataset synth` | synthetic candidates into `data/golden/review/pending.csv` |
| `ragate dataset freeze --review CSV --version v2 --base v1` | approved rows into a new version |
| `ragate eval [--config] [--out]` | evaluate and store a run |
| `ragate baseline` | write `baselines/baseline.json` |
| `ragate gate --candidate run.json` | compare, report, exit code |
| `ragate e2e` | everything above in one go |
| `ragate experiments` | the experiment grid |
| `ragate noise --repeats N` | judge noise band into `baselines/noise.json` |
| `ragate judge-lock [--check]` | judge prompt version lock |
| `ragate serve` | API and dashboard |

See the course page for the full walkthrough, design decisions and runbook.
