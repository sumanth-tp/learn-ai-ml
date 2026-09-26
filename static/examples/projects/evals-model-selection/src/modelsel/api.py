"""HTTP API: trigger selection runs, poll their status, read the report.

    POST /runs                 start a run (Idempotency-Key header supported) -> 202
    GET  /runs                 list recent runs
    GET  /runs/{run_id}        status and summary
    GET  /runs/{run_id}/report the HTML report
    GET  /healthz              liveness

Runs execute in-process as background tasks, one at a time (a lock), because a
selection run is a batch job that saturates provider rate limits by itself. At
larger scale this becomes a queue and a worker; the endpoints do not change.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from modelsel.config import Settings, get_settings
from modelsel.logging_setup import configure_logging, log_event
from modelsel.pipeline import run_selection
from modelsel.store import RunStore

logger = logging.getLogger(__name__)


class RunRequest(BaseModel):
    include_private: bool = False
    models: list[str] | None = Field(default=None, max_length=20)


class RunAccepted(BaseModel):
    run_id: str
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings if hasattr(app.state, "settings") else get_settings()
    configure_logging(settings.log_level, settings.log_json)
    app.state.settings = settings
    app.state.store = RunStore(settings.db_path)
    app.state.lock = asyncio.Lock()
    yield
    app.state.store.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="modelsel", version="0.1.0", lifespan=lifespan)
    if settings is not None:
        app.state.settings = settings

    async def _execute(state: Any, run_id: str, body: RunRequest) -> None:
        async with state.lock:
            try:
                await run_selection(
                    state.settings,
                    state.store,
                    include_private=body.include_private,
                    run_id=run_id,
                    candidates=body.models,
                )
            except Exception as exc:  # the store already has status=failed; keep the server alive
                log_event(logger, "background_run_failed", logging.ERROR, run_id=run_id, error=repr(exc))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs", status_code=202, response_model=RunAccepted)
    async def start_run(
        body: RunRequest,
        request: Request,
        background: BackgroundTasks,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=128)] = None,
    ) -> RunAccepted:
        state = request.app.state
        if body.include_private and not state.settings.allow_private:
            raise HTTPException(status_code=403, detail="private split is locked (MODELSEL_ALLOW_PRIVATE=false)")
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        stored_id = state.store.create_run(run_id, state.settings.profile, ["test"], idempotency_key)
        if stored_id != run_id:
            existing = state.store.get_run(stored_id)
            return RunAccepted(run_id=stored_id, status=existing["status"] if existing else "unknown")
        background.add_task(_execute, state, run_id, body)
        return RunAccepted(run_id=run_id, status="queued")

    @app.get("/runs")
    async def list_runs(request: Request) -> list[dict[str, Any]]:
        return request.app.state.store.list_runs()

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str, request: Request) -> dict[str, Any]:
        run = request.app.state.store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {k: v for k, v in run.items() if k not in {"report_md", "report_html"}}

    @app.get("/runs/{run_id}/report", response_class=HTMLResponse)
    async def get_report(run_id: str, request: Request) -> HTMLResponse:
        run = request.app.state.store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        if not run.get("report_html"):
            raise HTTPException(status_code=409, detail=f"run is {run['status']}; no report yet")
        return HTMLResponse(run["report_html"])

    return app


app = create_app()
