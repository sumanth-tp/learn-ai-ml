"""FastAPI service: SSE chat, approval resume, history, review queue and time travel."""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import resources
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from support_agent.api.schemas import ChatRequest, ForkRequest, ReplayRequest, ResumeRequest
from support_agent.config import Settings, get_settings
from support_agent.container import Container, build_container
from support_agent.errors import PermissionDeniedError, SupportError
from support_agent.graph import ApprovalDecision, build_graph
from support_agent.logging_setup import configure_logging, request_id_var
from support_agent.persistence import open_persistence
from support_agent.runner import (
    Event,
    NoPendingApprovalError,
    SupportRunner,
    ThreadBusyError,
    UnknownThreadError,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Customer:
    user_id: str


@dataclass(frozen=True)
class Reviewer:
    name: str


def _check_token(authorization: str | None, expected: str) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    if not secrets.compare_digest(authorization.removeprefix("Bearer "), expected):
        raise HTTPException(401, "invalid token")


def sse(event: Event) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.data, default=str)}\n\n"


async def _sse_stream(events: AsyncIterator[Event]) -> AsyncIterator[str]:
    try:
        async for ev in events:
            yield sse(ev)
    except SupportError as exc:
        # Headers are already sent, so errors mid-stream become an error event.
        yield sse(Event("error", {"type": type(exc).__name__, "detail": str(exc)}))
    except Exception:
        log.exception("stream failed")
        yield sse(Event("error", {"type": "InternalError", "detail": "internal error"}))


def runner_dep(request: Request) -> SupportRunner:
    return request.app.state.runner  # type: ignore[no-any-return]


def customer(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_user_id: Annotated[str | None, Header()] = None,
) -> Customer:
    # In production an API gateway verifies the customer's session and injects
    # X-User-Id; this service only trusts it together with the service token.
    _check_token(authorization, request.app.state.settings.api_token.get_secret_value())
    if not x_user_id:
        raise HTTPException(401, "missing X-User-Id")
    return Customer(x_user_id)


def reviewer(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_reviewer: Annotated[str | None, Header()] = None,
) -> Reviewer:
    _check_token(authorization, request.app.state.settings.reviewer_token.get_secret_value())
    return Reviewer(x_reviewer or "reviewer")


RunnerDep = Annotated[SupportRunner, Depends(runner_dep)]
CustomerDep = Annotated[Customer, Depends(customer)]
ReviewerDep = Annotated[Reviewer, Depends(reviewer)]


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_json)
        c = container or build_container(settings)
        async with open_persistence(settings) as (saver, store):
            app.state.container = c
            app.state.runner = SupportRunner(c, build_graph(c.deps, saver, store))
            log.info(
                "support agent ready",
                extra={"env": settings.app_env, "fake_llm": settings.fake_llm},
            )
            yield

    app = FastAPI(title="Support agent", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    @app.middleware("http")
    async def request_id_mw(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request_id_var.set(rid)
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    def _owner_or_404(runner: SupportRunner, thread_id: str) -> str:
        try:
            return runner.thread_owner(thread_id)
        except UnknownThreadError as exc:
            raise HTTPException(404, "unknown thread") from exc

    # ---- health and metrics ------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request) -> dict[str, str]:
        with request.app.state.container.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready"}

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("support_agent.api").joinpath("static/index.html").read_text()

    # ---- customer endpoints ------------------------------------------------

    @app.post("/v1/chat")
    async def chat(body: ChatRequest, who: CustomerDep, runner: RunnerDep) -> StreamingResponse:
        if body.thread_id:
            try:
                runner.ensure_thread(body.thread_id, who.user_id)
            except PermissionDeniedError as exc:
                raise HTTPException(403, str(exc)) from exc
            if runner.is_busy(body.thread_id):
                raise HTTPException(409, "a reply is already in progress on this thread")
        return StreamingResponse(
            _sse_stream(runner.stream_turn(body.thread_id, who.user_id, body.message)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/threads/{thread_id}/history")
    async def history(thread_id: str, who: CustomerDep, runner: RunnerDep) -> dict[str, Any]:
        if _owner_or_404(runner, thread_id) != who.user_id:
            raise HTTPException(403, "not your conversation")
        return await runner.history(thread_id)

    # ---- reviewer / operator endpoints -------------------------------------

    @app.get("/v1/approvals")
    def approvals(_: ReviewerDep, runner: RunnerDep) -> list[dict[str, Any]]:
        return runner.list_pending_approvals()

    @app.post("/v1/threads/{thread_id}/resume")
    async def resume(
        thread_id: str, body: ResumeRequest, rev: ReviewerDep, runner: RunnerDep
    ) -> StreamingResponse:
        _owner_or_404(runner, thread_id)
        if runner.is_busy(thread_id):
            raise HTTPException(409, "thread is busy")
        pending = await runner.pending_interrupts(thread_id)
        ids = {p["id"] for p in pending}
        if not pending or (body.interrupt_id and body.interrupt_id not in ids):
            # Resuming twice lands here: nothing is waiting, so nothing runs twice.
            raise HTTPException(409, "no pending approval on this thread")
        decision = ApprovalDecision(approved=body.approved, reviewer=rev.name, note=body.note)
        return StreamingResponse(
            _sse_stream(runner.stream_resume(thread_id, decision, body.interrupt_id)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/threads/{thread_id}/state")
    async def thread_state(thread_id: str, _: ReviewerDep, runner: RunnerDep) -> dict[str, Any]:
        _owner_or_404(runner, thread_id)
        return await runner.history(thread_id)

    @app.get("/v1/threads/{thread_id}/checkpoints")
    async def checkpoints(
        thread_id: str, _: ReviewerDep, runner: RunnerDep
    ) -> list[dict[str, Any]]:
        _owner_or_404(runner, thread_id)
        return await runner.checkpoints(thread_id)

    @app.post("/v1/threads/{thread_id}/replay")
    async def replay(
        thread_id: str, body: ReplayRequest, _: ReviewerDep, runner: RunnerDep
    ) -> dict[str, Any]:
        _owner_or_404(runner, thread_id)
        try:
            return await runner.replay(thread_id, body.checkpoint_id)
        except ThreadBusyError as exc:
            raise HTTPException(409, "thread is busy") from exc

    @app.post("/v1/threads/{thread_id}/fork")
    async def fork(
        thread_id: str, body: ForkRequest, _: ReviewerDep, runner: RunnerDep
    ) -> dict[str, Any]:
        _owner_or_404(runner, thread_id)
        try:
            return await runner.fork(thread_id, body.checkpoint_id, body.message)
        except ThreadBusyError as exc:
            raise HTTPException(409, "thread is busy") from exc

    @app.exception_handler(NoPendingApprovalError)
    async def _no_pending(_: Request, exc: NoPendingApprovalError) -> Response:
        return Response(
            json.dumps({"detail": str(exc)}), status_code=409, media_type="application/json"
        )

    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn --factory support_agent.api.app:app_factory`."""
    return create_app()
