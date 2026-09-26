"""HTTP API and a small dashboard over the run store.

GET  /health             liveness plus which providers are active
POST /ask                ask the RAG assistant (the app under test)
GET  /runs               recent eval runs with headline metrics
GET  /runs/{run_id}      one run, every item
POST /gate               compare two stored runs with the gate rules
GET  /metrics            Prometheus metrics
GET  /                   HTML dashboard: runs and gate decisions
"""

from __future__ import annotations

import html
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from ragate import __version__
from ragate.config import load_pipeline_config
from ragate.evaluation.gate import compare, load_gate_config, load_noise
from ragate.evaluation.report import fmt, gate_report
from ragate.evaluation.store import RunStore
from ragate.log import get_logger
from ragate.rag.pipeline import RagPipeline, build_pipeline
from ragate.settings import Settings, get_settings
from ragate.tracing import tracing_status

log = get_logger(__name__)

ASK_LATENCY = Histogram("ragate_ask_latency_seconds", "End-to-end /ask latency")
ASK_TOTAL = Counter("ragate_ask_total", "Questions answered", ["outcome"])
ASK_TOKENS = Counter("ragate_ask_tokens_total", "LLM tokens used by /ask", ["kind"])
ASK_COST = Counter("ragate_ask_cost_usd_total", "Estimated LLM spend by /ask")

DASH_METRICS = ["recall_at_k", "faithfulness", "correctness", "refusal_rate", "pii_leak_rate",
                "latency_p95_ms", "cost_per_query_usd"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    refused: bool
    sources: list[str]
    latency_ms: float
    cost_usd: float


class GateRequest(BaseModel):
    baseline_run: str
    candidate_run: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    state: dict[str, object] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        config = load_pipeline_config(settings.config_dir / "pipeline.yaml")
        state["pipeline"] = build_pipeline(config, settings)
        state["runs"] = RunStore(settings.runs_db)
        log.info("api_ready", config=config.name, provider=settings.provider)
        yield

    app = FastAPI(title="ragate", version=__version__, lifespan=lifespan)

    def runs() -> RunStore:
        return state["runs"]  # type: ignore[return-value]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "provider": settings.provider,
                "judge_provider": settings.judge_provider, "tracing": tracing_status()}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        pipeline: RagPipeline = state["pipeline"]  # type: ignore[assignment]
        start = time.perf_counter()
        try:
            ans = pipeline.ask(req.question)
        except Exception as exc:
            ASK_TOTAL.labels("error").inc()
            log.error("ask_failed", error=repr(exc))
            raise HTTPException(status_code=503, detail="the assistant is unavailable") from exc
        elapsed = time.perf_counter() - start
        ASK_LATENCY.observe(elapsed)
        ASK_TOTAL.labels("refused" if ans.refused else "answered").inc()
        ASK_TOKENS.labels("input").inc(ans.usage.input_tokens)
        ASK_TOKENS.labels("output").inc(ans.usage.output_tokens)
        ASK_COST.inc(ans.cost_usd)
        return AskResponse(answer=ans.answer, citations=ans.citations, refused=ans.refused,
                           sources=[c.chunk.chunk_id for c in ans.contexts],
                           latency_ms=round(elapsed * 1000, 2), cost_usd=ans.cost_usd)

    @app.get("/runs")
    def list_runs(limit: int = 50) -> list[dict]:
        return runs().list(limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run = runs().get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run.model_dump()

    @app.post("/gate", response_class=PlainTextResponse)
    def gate(req: GateRequest) -> str:
        base, cand = runs().get(req.baseline_run), runs().get(req.candidate_run)
        if base is None or cand is None:
            raise HTTPException(status_code=404, detail="run not found")
        cfg = load_gate_config(settings.config_dir / "gate.yaml")
        result = compare(base, cand, cfg, load_noise(settings.baselines_dir / "noise.json",
                                                      cand.judge.model_id))
        report = gate_report(result, base, cand)
        runs().record_decision(base.run_id, cand.run_id, result.decision, report)
        return report

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        rows = []
        for r in runs().list(50):
            cells = "".join(f"<td>{fmt(r['aggregates'].get(m), m)}</td>" for m in DASH_METRICS)
            rows.append(f"<tr><td><a href='/runs/{html.escape(r['run_id'])}'>"
                        f"{html.escape(r['run_id'])}</a></td><td>{html.escape(r['created_at'])}"
                        f"</td><td>{html.escape(r['dataset_version'])}</td>{cells}</tr>")
        decisions = "".join(
            f"<tr><td>{html.escape(d['created_at'])}</td><td>{html.escape(d['baseline'])}</td>"
            f"<td>{html.escape(d['candidate'])}</td><td>{html.escape(d['decision'])}</td></tr>"
            for d in runs().decisions()
        )
        head = "".join(f"<th>{m}</th>" for m in DASH_METRICS)
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>ragate runs</title>
<style>body{{font-family:system-ui;margin:2rem;background:#fbfaf7;color:#222}}
table{{border-collapse:collapse;margin-bottom:2rem}}td,th{{border:1px solid #ccc;
padding:.3rem .6rem;font-size:.9rem}}th{{background:#eee}}</style></head><body>
<h1>ragate: eval runs</h1><table><tr><th>run</th><th>created</th><th>dataset</th>{head}</tr>
{''.join(rows)}</table><h2>Gate decisions</h2><table><tr><th>when</th><th>baseline</th>
<th>candidate</th><th>decision</th></tr>{decisions}</table></body></html>"""

    return app
