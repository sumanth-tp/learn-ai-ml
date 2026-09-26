"""Shared fixtures: a real gateway served over HTTP on a free local port,
with in-process upstreams (no network beyond 127.0.0.1, no API keys)."""

from __future__ import annotations

import shutil
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport

from mcp_gateway.config import Settings
from mcp_gateway.demo_upstreams import docs_server, payments_server, tickets_server
from mcp_gateway.gateway import build_gateway
from mcp_gateway.identity import mint_dev_token
from mcp_gateway.middleware import Components
from mcp_gateway.secret_broker import EnvSecretBroker
from mcp_gateway.upstreams import UpstreamSpec

ROOT = Path(__file__).resolve().parents[1]
SECRET = "test-secret-that-is-at-least-32-characters"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ServerThread:
    """Run an ASGI app with uvicorn in a daemon thread."""

    def __init__(self, app: Any) -> None:
        self.port = free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> ServerThread:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline:
                raise TimeoutError("server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    policy = tmp_path / "policy.yaml"
    shutil.copy(ROOT / "config" / "policy.yaml", policy)
    values: dict[str, Any] = {
        "jwt_secret": SECRET,
        "jwt_algorithm": "HS256",
        "state_dir": tmp_path / "var",
        "policy_file": policy,
        "upstreams_file": ROOT / "config" / "upstreams.yaml",
        "definition_refresh_seconds": 0,  # re-read definitions on every call
        "retry_base_delay_seconds": 0.01,
        "upstream_timeout_seconds": 5,
        "log_json": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def inprocess_specs() -> list[UpstreamSpec]:
    return [
        UpstreamSpec(name="docs", transport="inprocess", target="x:y",
                     cacheable_tools=["list_docs", "read_doc", "search_docs"]),
        UpstreamSpec(name="payments", transport="inprocess", target="x:y",
                     cacheable_tools=["get_balance"]),
        UpstreamSpec(name="tickets", transport="inprocess", target="x:y",
                     cacheable_tools=["search_tickets", "get_ticket"]),
    ]


@dataclass
class Harness:
    url: str
    settings: Settings
    components: Components
    upstreams: dict[str, FastMCP[Any]]

    def token(self, sub: str, groups: list[str], email: str | None = None) -> str:
        return mint_dev_token(self.settings, sub, groups, email=email)

    @asynccontextmanager
    async def client(self, sub: str = "alice", groups: list[str] | None = None,
                     token: str | None = None) -> Any:
        tok = token or self.token(sub, groups or ["employees"])
        async with Client(StreamableHttpTransport(self.url, auth=tok)) as c:
            yield c


@pytest.fixture
def upstream_servers() -> dict[str, FastMCP[Any]]:
    return {
        "docs": docs_server.create_server(),
        "payments": payments_server.create_server(),
        "tickets": tickets_server.create_server(),
    }


def start_harness(tmp_path: Path, servers: dict[str, FastMCP[Any]],
                  specs: list[UpstreamSpec] | None = None, **overrides: Any) -> tuple[Harness, ServerThread]:
    settings = make_settings(tmp_path, **overrides)
    gateway, components = build_gateway(
        settings, specs=specs or inprocess_specs(), broker=EnvSecretBroker({}),
        server_overrides=servers,
    )
    thread = ServerThread(gateway.http_app(path="/mcp")).__enter__()
    return Harness(f"http://127.0.0.1:{thread.port}/mcp", settings, components, servers), thread


@pytest.fixture
def harness(tmp_path: Path, upstream_servers: dict[str, FastMCP[Any]]) -> Iterator[Harness]:
    h, thread = start_harness(tmp_path, upstream_servers)
    try:
        yield h
    finally:
        thread.__exit__()
