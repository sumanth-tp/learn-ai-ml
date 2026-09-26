"""Pagination, rate limiting, timeouts, progress and log notifications, idempotency races."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from fastmcp import Client
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult

from tests.conftest import make_app, make_settings, text_of


async def test_paginate_through_everything_without_duplicates(local_client: Client) -> None:
    for i in range(23):
        await local_client.call_tool(
            "create_ticket",
            {
                "title": f"Bulk ticket {i:02d}",
                "description": "Generated for pagination.",
            },
        )
    seen: list[int] = []
    cursor = None
    pages = 0
    while True:
        args: dict[str, Any] = {"limit": 5, "query": "bulk"}
        if cursor:
            args["cursor"] = cursor
        page = (await local_client.call_tool("search_tickets", args)).structured_content
        seen += [t["id"] for t in page["items"]]
        pages += 1
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert len(seen) == 23 and len(set(seen)) == 23
    assert seen == sorted(seen, reverse=True)  # newest first, stable
    assert pages == 5


async def test_page_size_is_capped(local_client: Client) -> None:
    r = await local_client.call_tool("search_tickets", {"limit": 500}, raise_on_error=False)
    assert r.is_error and "less than or equal to 100" in text_of(r)


async def test_per_user_rate_limit(tmp_path: Path) -> None:
    app = await make_app(
        make_settings(tmp_path, auth_mode="local", rate_limit_burst=3, rate_limit_per_minute=1)
    )
    async with Client(app.mcp) as c:
        results = [await c.call_tool("search_tickets", {}, raise_on_error=False) for _ in range(5)]
    outcomes = [r.is_error for r in results]
    assert outcomes[:3] == [False, False, False]
    assert all(outcomes[3:])
    assert "[rate_limited]" in text_of(results[-1])
    metrics = app.metrics.render().decode()
    assert 'helpdesk_mcp_rate_limited_total{tenant="acme"} 2.0' in metrics
    await app.engine.dispose()


class SlowModel(BaseChatModel):
    delay: float = 5.0

    @property
    def _llm_type(self) -> str:
        return "slow"

    def _generate(self, *a, **k) -> ChatResult:  # pragma: no cover - async path used
        raise NotImplementedError

    async def _agenerate(self, *a, **k) -> ChatResult:
        await asyncio.sleep(self.delay)
        raise AssertionError("should have been cancelled")


async def test_tool_timeout_cancels_and_reports(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, auth_mode="local", tool_timeout_s=0.3, llm_timeout_s=30)
    app = await make_app(settings, chat_model=SlowModel())
    async with Client(app.mcp) as c:
        r = await c.call_tool("suggest_triage", {"ticket_id": 1}, raise_on_error=False)
    assert r.is_error and text_of(r).startswith("[timeout]")
    assert "helpdesk_mcp_errors_total" in app.metrics.render().decode()
    await app.engine.dispose()


async def test_progress_and_log_notifications(local_app) -> None:
    progress: list[tuple[float, float | None, str | None]] = []
    logs: list[str] = []

    async def on_progress(p: float, total: float | None, msg: str | None) -> None:
        progress.append((p, total, msg))

    async def on_log(message) -> None:
        logs.append(message.data["msg"])

    async with Client(local_app.mcp, log_handler=on_log, mode="legacy") as c:
        await c.call_tool("suggest_triage", {"ticket_id": 2}, progress_handler=on_progress)
        await c.call_tool("incident_report", {"window_hours": 24})
    assert [p[0] for p in progress] == [0, 1, 2]
    assert progress[-1][2] == "Done"
    assert any("Scanning tickets" in m for m in logs)


async def test_concurrent_creates_with_same_key_make_one_ticket(local_client: Client) -> None:
    key = f"race-{uuid.uuid4()}"
    args = {
        "title": "Race condition test",
        "description": "Two retries at once.",
        "idempotency_key": key,
    }
    results = await asyncio.gather(
        *[local_client.call_tool("create_ticket", args) for _ in range(5)]
    )
    ids = {r.structured_content["id"] for r in results}
    assert len(ids) == 1
    found = await local_client.call_tool("search_tickets", {"query": "race condition"})
    assert len(found.structured_content["items"]) == 1
