"""FastAPI service: streaming answers over Server-Sent Events, approvals, time travel.

No `from __future__ import annotations` here: FastAPI resolves the Annotated
dependency aliases defined inside create_app at runtime, and string annotations
would hide them.
"""

import hmac
import json
import re
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from data_analyst.config import Settings
from data_analyst.logging_setup import configure_logging, get_logger
from data_analyst.service import AnalystService, Event

log = get_logger(__name__)
STATIC = Path(__file__).with_name("static")
THREAD_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class ApprovalRequest(BaseModel):
    approved: bool
    reviewer: str = Field(min_length=1, max_length=64)
    comment: str = Field(default="", max_length=500)


class ForkRequest(BaseModel):
    checkpoint_id: str
    sql: str | None = Field(default=None, max_length=5000)


def _sse(events: Iterator[Event]) -> Iterator[str]:
    try:
        for ev in events:
            yield f"event: {ev.type}\ndata: {json.dumps(ev.data, default=str)}\n\n"
    except Exception as e:  # a provider outage mid-stream: tell the client, then close
        log.exception("stream_failed")
        yield f"event: error\ndata: {json.dumps({'error': type(e).__name__})}\n\n"


def create_app(settings: Settings | None = None, service: AnalystService | None = None) -> FastAPI:
    load_dotenv()
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.svc = service or AnalystService.from_settings(settings)
        log.info("service_started", llm_mode=settings.llm_mode, model=settings.llm_model)
        yield
        app.state.svc.close()

    app = FastAPI(title="Data analyst agent", version="0.1.0", lifespan=lifespan)

    def svc(request: Request) -> AnalystService:
        return request.app.state.svc

    def auth(x_api_key: Annotated[str | None, Header()] = None) -> None:
        expected = settings.api_key.get_secret_value() if settings.api_key else None
        if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    def thread_id(thread: str) -> str:
        if not THREAD_RE.match(thread):
            raise HTTPException(status_code=422, detail="thread id must match [A-Za-z0-9_-]{1,64}")
        return thread

    Svc = Annotated[AnalystService, Depends(svc)]
    Thread = Annotated[str, Depends(thread_id)]
    stream_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(s: Svc) -> dict[str, Any]:
        if not s.deps.executor.ping():
            raise HTTPException(status_code=503, detail="warehouse unavailable")
        return {"status": "ready", "llm_mode": settings.llm_mode}

    @app.get("/metrics", include_in_schema=False)
    def prom() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/threads/{thread}/ask", dependencies=[Depends(auth)])
    def ask(thread: Thread, body: AskRequest, s: Svc) -> StreamingResponse:
        if s.pending_approval(thread) is not None:
            raise HTTPException(status_code=409, detail="thread is waiting for approval")
        return StreamingResponse(
            _sse(s.ask(thread, body.question)),
            media_type="text/event-stream",
            headers=stream_headers,
        )

    @app.post("/v1/threads/{thread}/approval", dependencies=[Depends(auth)])
    def approve(thread: Thread, body: ApprovalRequest, s: Svc) -> StreamingResponse:
        if s.pending_approval(thread) is None:
            raise HTTPException(status_code=409, detail="nothing to approve on this thread")
        events = s.resume(thread, body.approved, body.reviewer, body.comment)
        return StreamingResponse(
            _sse(events), media_type="text/event-stream", headers=stream_headers
        )

    @app.get("/v1/threads/{thread}", dependencies=[Depends(auth)])
    def state(thread: Thread, s: Svc) -> dict[str, Any]:
        return s.state(thread)

    @app.get("/v1/threads/{thread}/history", dependencies=[Depends(auth)])
    def history(thread: Thread, s: Svc) -> list[dict[str, Any]]:
        return s.history(thread)

    @app.post("/v1/threads/{thread}/fork", dependencies=[Depends(auth)])
    def fork(thread: Thread, body: ForkRequest, s: Svc) -> StreamingResponse:
        events = (
            s.fork_with_sql(thread, body.checkpoint_id, body.sql)
            if body.sql
            else s.replay(thread, body.checkpoint_id)
        )
        return StreamingResponse(
            _sse(events), media_type="text/event-stream", headers=stream_headers
        )

    return app
