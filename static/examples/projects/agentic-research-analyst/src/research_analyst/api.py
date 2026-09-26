"""FastAPI surface: blocking report, SSE progress stream, status and resume."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from research_analyst import __version__
from research_analyst.config import Settings, get_settings
from research_analyst.logging_setup import configure_logging
from research_analyst.models import FinalReport
from research_analyst.service import ResearchService, RunStatus

log = logging.getLogger(__name__)


class ReportRequest(BaseModel):
    question: str = Field(min_length=10, max_length=500)
    thread_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_\-]{1,64}$",
        description="Reuse to make the request idempotent.",
    )


def svc_dep(request: Request) -> ResearchService:
    return request.app.state.svc


Svc = Annotated[ResearchService, Depends(svc_dep)]


def create_app(settings: Settings | None = None, service: ResearchService | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    gate = asyncio.Semaphore(settings.max_concurrent_reports)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = service or ResearchService(settings)
        async with svc:
            app.state.svc = svc
            yield

    app = FastAPI(title="Research Analyst", version=__version__, lifespan=lifespan)

    def auth(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if settings.api_key is None:
            return
        expected = settings.api_key.get_secret_value()
        if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @asynccontextmanager
    async def slot() -> AsyncIterator[None]:
        if gate.locked():
            raise HTTPException(
                status_code=429, detail="too many concurrent reports", headers={"Retry-After": "30"}
            )
        async with gate:
            yield

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "version": __version__, "mode": settings.mode}

    @app.post("/v1/reports", response_model=FinalReport, dependencies=[Depends(auth)])
    async def create_report(req: ReportRequest, svc: Svc) -> FinalReport:
        thread_id = req.thread_id or uuid.uuid4().hex
        async with slot():
            try:
                return await asyncio.wait_for(
                    svc.run(req.question, thread_id), timeout=settings.report_timeout_s
                )
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=504,
                    detail={
                        "error": "report timed out; progress is checkpointed",
                        "thread_id": thread_id,
                        "resume": f"/v1/reports/{thread_id}/resume",
                    },
                ) from exc

    @app.post("/v1/reports/stream", dependencies=[Depends(auth)])
    async def stream_report(req: ReportRequest, svc: Svc) -> StreamingResponse:
        thread_id = req.thread_id or uuid.uuid4().hex
        if gate.locked():
            raise HTTPException(status_code=429, detail="too many concurrent reports")

        async def sse() -> AsyncIterator[str]:
            async with gate:
                yield f"event: run\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
                try:
                    async for ev in svc.stream(req.question, thread_id):
                        yield f"event: {ev.type}\ndata: {ev.model_dump_json()}\n\n"
                except Exception as exc:  # already emitted as run_failed; close cleanly
                    log.warning("stream ended with error: %s", exc)
                    return
                st = await svc.status(thread_id)
                if st.report is not None:
                    yield f"event: report\ndata: {st.report.model_dump_json()}\n\n"

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Thread-Id": thread_id},
        )

    @app.get("/v1/reports/{thread_id}", response_model=RunStatus, dependencies=[Depends(auth)])
    async def get_report(thread_id: str, svc: Svc) -> RunStatus:
        st = await svc.status(thread_id)
        if st.status == "not_found":
            raise HTTPException(status_code=404, detail="unknown thread id")
        return st

    @app.post(
        "/v1/reports/{thread_id}/resume", response_model=FinalReport, dependencies=[Depends(auth)]
    )
    async def resume_report(thread_id: str, svc: Svc) -> FinalReport:
        st = await svc.status(thread_id)
        if st.status == "not_found":
            raise HTTPException(status_code=404, detail="unknown thread id")
        if st.report is not None:
            return st.report
        async with slot():
            async for _ in svc.resume(thread_id):
                pass
        st = await svc.status(thread_id)
        if st.report is None:
            raise HTTPException(status_code=500, detail="resume did not complete")
        return st.report

    @app.delete("/v1/reports/{thread_id}", status_code=204, dependencies=[Depends(auth)])
    async def delete_report(thread_id: str, svc: Svc) -> None:
        if not await svc.delete(thread_id):
            raise HTTPException(status_code=404, detail="unknown thread id")

    return app
