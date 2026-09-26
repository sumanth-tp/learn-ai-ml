"""HTTP API: chat, feedback, traces, metrics, alerts, the review queue and the
dashboard. Eval workers run in-process by default (AGENTMON_INPROCESS_WORKERS=false
when a separate worker container is used)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agentmon.agent.service import ChatResponse
from agentmon.config import Settings
from agentmon.feedback.review import ReviewQueue
from agentmon.logging_setup import configure_logging
from agentmon.models import Feedback
from agentmon.monitoring.alerts import load_rules, run_alerts
from agentmon.monitoring.dashboard import render_dashboard
from agentmon.monitoring.metrics import TraceFrame, aggregate
from agentmon.runtime import Runtime, build_runtime


class ChatRequest(BaseModel):
    user_id: str = Field(pattern=r"^CUST-\d+$")
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    request_id: str | None = Field(None, max_length=64, description="Idempotency key")


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: Literal[-1, 1]
    correction: str | None = Field(None, max_length=2000)


class ReviewDecision(BaseModel):
    reviewer: str = Field(min_length=1, max_length=64)
    note: str = ""


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_json)
    rt = runtime or build_runtime(settings)
    queue = ReviewQueue(
        rt.store, rt.clock, settings.golden_seed_path, settings.golden_production_path
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        stop = asyncio.Event()
        task = None
        if settings.inprocess_workers:
            task = asyncio.create_task(rt.workers.run_forever(stop))
        yield
        stop.set()
        if task is not None:
            await task

    app = FastAPI(title="agentmon", version="0.1.0", lifespan=lifespan)
    app.state.runtime = rt

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "prompt_version": rt.service.prompt_version,
            "provider": settings.llm_provider,
            "eval_jobs": rt.store.job_counts(),
        }

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        return await run_in_threadpool(
            rt.service.handle, req.user_id, req.message, req.session_id, req.request_id
        )

    @app.post("/v1/feedback", status_code=202)
    def feedback(req: FeedbackRequest) -> dict[str, str]:
        try:
            rt.pipeline.on_feedback(
                Feedback(
                    trace_id=req.trace_id,
                    rating=req.rating,
                    correction=req.correction,
                    ts=rt.clock.now(),
                )
            )
        except KeyError as exc:
            raise HTTPException(404, "unknown trace_id") from exc
        return {"status": "accepted"}

    @app.get("/v1/traces/{trace_id}")
    def trace(trace_id: str) -> dict[str, object]:
        t = rt.store.get_trace(trace_id)
        if t is None:
            raise HTTPException(404, "unknown trace_id")
        return {
            "trace": t.model_dump(),
            "spans": rt.store.spans_for(trace_id),
            "evals": [e.model_dump() for e in rt.store.evals_for(trace_id)],
            "feedback": (fb.model_dump() if (fb := rt.store.feedback_for(trace_id)) else None),
        }

    @app.get("/v1/metrics")
    def metrics(
        window_hours: float = Query(1.0, gt=0, le=168),
        lookback_hours: float = Query(24.0, gt=0, le=24 * 31),
    ) -> list[dict]:
        end = rt.clock.now()
        start = end - lookback_hours * 3600
        return aggregate(TraceFrame.load(rt.store, start, end), start, end, window_hours * 3600)

    @app.post("/v1/alerts/evaluate")
    def evaluate_alerts() -> list[dict]:
        return run_alerts(rt.store, load_rules(settings.alerts_path))

    @app.get("/v1/alerts")
    def alerts() -> list[dict]:
        return rt.store.alerts()

    @app.get("/v1/review")
    def review(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
        return queue.pending(limit)

    @app.post("/v1/review/{trace_id}/approve")
    def approve(trace_id: str, d: ReviewDecision) -> dict[str, str]:
        try:
            return {"status": queue.approve(trace_id, d.reviewer)}
        except KeyError as exc:
            raise HTTPException(404, "unknown trace_id") from exc

    @app.post("/v1/review/{trace_id}/reject")
    def reject(trace_id: str, d: ReviewDecision) -> dict[str, str]:
        queue.reject(trace_id, d.reviewer, d.note)
        return {"status": "rejected"}

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard(rt.store)

    return app
