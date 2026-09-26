"""Integration: real transports (stdio subprocess, streamable HTTP), the API, tracing, eval."""

from __future__ import annotations

import asyncio
import json
import socket
import sys

import httpx
import uvicorn
from langchain_core.messages import AIMessage

from demo_servers import calendar_server
from mcp_host.api import create_app
from mcp_host.connection import ServerConnection
from mcp_host.evals import run_eval
from mcp_host.llm import FakeToolModel, keyword_router
from mcp_host.sampling import make_sampling_callback
from mcp_host.settings import HttpServer, PolicyConfig, StdioServer
from mcp_host.transports import http_factory, stdio_factory
from tests.conftest import ROOT, call


async def test_stdio_subprocess_end_to_end(settings, data_dirs):
    cfg = StdioServer(
        transport="stdio",
        command=sys.executable,
        args=["-m", "demo_servers.notes_server"],
        env={"NOTES_DIR": str(data_dirs["notes"])},
    )
    conn = ServerConnection("notes", stdio_factory(cfg), settings, timeout_s=10)
    conn.start()
    try:
        assert await conn.wait_ready(20), conn.last_error
        result = await conn.call_tool("list_notes", {})
        assert "sprint-goals" in result.structuredContent["result"]
    finally:
        await conn.stop()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_streamable_http_with_sampling(settings, tmp_path):
    port = free_port()
    app = calendar_server.create_server(
        tmp_path / "c.json", ROOT / "data" / "calendar_seed.json"
    ).streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    llm = FakeToolModel(responder=keyword_router)
    conn = ServerConnection(
        "calendar",
        http_factory(HttpServer(transport="http", url=f"http://127.0.0.1:{port}/mcp"), 5),
        settings,
        timeout_s=10,
        sampling_callback=make_sampling_callback(
            "calendar", PolicyConfig(allow_sampling=True), llm, 200, "fake"
        ),
    )
    conn.start()
    try:
        assert await conn.wait_ready(10), conn.last_error
        result = await conn.call_tool("summarise_day", {"day": "2026-10-05"})
        # "Summary:" comes from the host's (fake) LLM, reached via sampling over HTTP.
        assert result.structuredContent["result"].startswith("Summary:")
    finally:
        await conn.stop()
        server.should_exit = True
        await task


def test_sampling_disabled_means_no_callback():
    assert (
        make_sampling_callback(
            "x", PolicyConfig(), FakeToolModel(responder=keyword_router), 10, "m"
        )
        is None
    )


async def test_every_mcp_call_is_traced(make_runtime, spans):
    script = [call("docs__search", {"query": "leave"}, "s1"), AIMessage(content="ok")]
    async with make_runtime(script) as rt:
        async for _ in rt.service.send("tr1", "leave?"):
            pass
    names = [s.name for s in spans.get_finished_spans()]
    assert "mcp tools/call docs.search" in names
    assert "mcp initialize" in names and "mcp discover" in names
    tool_span = next(
        s for s in spans.get_finished_spans() if s.name == "mcp tools/call docs.search"
    )
    turn = next(s for s in spans.get_finished_spans() if s.name == "agent.turn")
    assert tool_span.context.trace_id == turn.context.trace_id  # nested under the turn
    assert tool_span.attributes["mcp.server.name"] == "docs"


def parse_sse(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


async def test_api_streams_and_handles_approval(make_runtime, data_dirs):
    script = [call("notes__delete_note", {"name": "reading-list"}, "d1"), AIMessage(content="gone")]
    async with make_runtime(script) as rt:
        app = create_app(rt.settings, runtime=rt)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://t"
            ) as client,
        ):
            assert (await client.get("/healthz")).json()["status"] == "ok"
            r = await client.post(
                "/threads/api-1/messages", json={"text": "delete note reading-list"}
            )
            assert r.headers["content-type"].startswith("text/event-stream")
            events = parse_sse(r.text)
            assert any(e["type"] == "approval_required" for e in events)
            state = (await client.get("/threads/api-1")).json()
            assert state["pending_approval"]["calls"][0]["id"] == "d1"
            r = await client.post("/threads/api-1/approval", json={"approve": ["d1"]})
            assert any(e.get("text") == "gone" for e in parse_sse(r.text))
            assert not (data_dirs["notes"] / "reading-list.md").exists()
            assert (
                await client.post("/threads/bad id!/messages", json={"text": "x"})
            ).status_code in {
                404,
                422,
            }


async def test_api_requires_bearer_token_when_configured(make_runtime):
    from pydantic import SecretStr

    async with make_runtime() as rt:
        cfg = rt.settings.model_copy(update={"api_key": SecretStr("s3cret")})
        app = create_app(cfg, runtime=rt)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://t"
            ) as client,
        ):
            assert (await client.get("/servers")).status_code == 401
            ok = await client.get("/servers", headers={"Authorization": "Bearer s3cret"})
            assert ok.status_code == 200 and "docs__search" in ok.json()["tools"]


async def test_tool_selection_eval_gate(settings):
    report = await run_eval(
        FakeToolModel(responder=keyword_router), settings, ROOT / "evals" / "tool_selection.jsonl"
    )
    assert report.n >= 15
    assert report.passed, report.confusions
    assert report.wrong_namespace == 0
