"""FastAPI front end: server-sent events for every turn, plus status and history."""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from mcp_host.agent import ApprovalDecision
from mcp_host.logs import configure_logging
from mcp_host.runtime import Runtime, open_runtime
from mcp_host.settings import Settings, get_settings

THREAD_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
UI = Path(__file__).with_name("ui.html")


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class PromptIn(BaseModel):
    server: str
    name: str
    arguments: dict[str, str] = Field(default_factory=dict)


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    """Build the app. Tests pass a ready ``runtime``; production opens its own."""
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if runtime is not None:
            app.state.rt = runtime
            yield
            return
        configure_logging(cfg.log_level, cfg.log_json)
        async with open_runtime(cfg) as rt:
            app.state.rt = rt
            yield

    app = FastAPI(title="MCP agent host", version="0.1.0", lifespan=lifespan)

    def auth(authorization: str | None = Header(default=None)) -> None:
        expected = cfg.api_key.get_secret_value() if cfg.api_key else None
        if expected is None:
            return
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(given, expected):
            raise HTTPException(status_code=401, detail="missing or wrong bearer token")

    def rt() -> Runtime:
        return app.state.rt

    def check_thread(thread_id: str) -> str:
        if not THREAD_RE.match(thread_id):
            raise HTTPException(status_code=422, detail="thread id must match [A-Za-z0-9_-]{1,64}")
        if rt().service.is_busy(thread_id):
            raise HTTPException(status_code=409, detail="thread is busy with another turn")
        return thread_id

    def sse(events: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
        async def body() -> AsyncIterator[str]:
            async for event in events:
                yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return UI.read_text(encoding="utf-8")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        snap = rt().host.registry.snapshot
        return {"status": "degraded" if snap.unavailable else "ok", "unavailable": snap.unavailable}

    @app.get("/servers", dependencies=[Depends(auth)])
    async def servers() -> dict[str, Any]:
        return rt().host.status()

    @app.get("/prompts", dependencies=[Depends(auth)])
    async def prompts() -> list[dict[str, Any]]:
        return rt().host.list_prompts()

    @app.get("/threads", dependencies=[Depends(auth)])
    async def threads() -> list[str]:
        return await rt().service.threads()

    @app.get("/threads/{thread_id}", dependencies=[Depends(auth)])
    async def thread(thread_id: str) -> dict[str, Any]:
        service = rt().service
        return {
            "thread_id": thread_id,
            "messages": await service.history(thread_id),
            "pending_approval": await service.pending_approval(thread_id),
        }

    @app.post("/threads/{thread_id}/messages", dependencies=[Depends(auth)])
    async def send(thread_id: str, body: MessageIn) -> StreamingResponse:
        return sse(rt().service.send(check_thread(thread_id), body.text))

    @app.post("/threads/{thread_id}/prompts", dependencies=[Depends(auth)])
    async def send_prompt(thread_id: str, body: PromptIn) -> StreamingResponse:
        return sse(
            rt().service.send_prompt(
                check_thread(thread_id), body.server, body.name, body.arguments
            )
        )

    @app.post("/threads/{thread_id}/approval", dependencies=[Depends(auth)])
    async def approve(thread_id: str, body: ApprovalDecision) -> StreamingResponse:
        return sse(rt().service.resume(check_thread(thread_id), body))

    return app
