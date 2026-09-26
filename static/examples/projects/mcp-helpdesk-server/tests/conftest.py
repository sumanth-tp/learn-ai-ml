"""Shared fixtures. Everything runs offline: SQLite files in tmp, fake LLM."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.utilities.tests import ASGIServer, asgi_server

from helpdesk_mcp.config import Settings
from helpdesk_mcp.db.models import Base
from helpdesk_mcp.seed import seed
from helpdesk_mcp.server import HelpdeskApp, build_server
from helpdesk_mcp.tokens import mint_token


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "environment": "test",
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        "auth_mode": "jwt",
        "rate_limit_burst": 1000,
        "rate_limit_per_minute": 60_000,
        "llm_provider": "fake",
        "log_json": False,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


async def make_app(settings: Settings, **kwargs) -> HelpdeskApp:
    app = build_server(settings, **kwargs)
    async with app.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed(app.engine)
    return app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[HelpdeskApp]:
    app = await make_app(settings)
    yield app
    await app.engine.dispose()


@pytest.fixture
async def local_app(tmp_path: Path) -> AsyncIterator[HelpdeskApp]:
    """No-token mode (as used by stdio) with an admin identity in tenant acme."""
    app = await make_app(make_settings(tmp_path, auth_mode="local", local_roles=["admin"]))
    yield app
    await app.engine.dispose()


@pytest.fixture
async def local_client(local_app: HelpdeskApp) -> AsyncIterator[Client]:
    async with Client(local_app.mcp) as client:
        yield client


@pytest.fixture
async def http(app: HelpdeskApp) -> AsyncIterator[ASGIServer]:
    """The real Starlette app (auth middleware and all), served in-process."""
    async with asgi_server(app.mcp, path=app.settings.mcp_path) as server:
        yield server


TokenFactory = Callable[..., str]


@pytest.fixture
def token(settings: Settings) -> TokenFactory:
    def _make(user: str = "alice", tenant: str = "acme", role: str = "requester", **kw) -> str:
        return mint_token(settings, user=user, tenant=tenant, roles=[role], **kw)

    return _make


@pytest.fixture
def as_user(http: ASGIServer, token: TokenFactory) -> Callable[..., Client]:
    """``async with as_user("sam", role="agent") as c:`` gives an authenticated client."""

    def _client(user: str = "alice", tenant: str = "acme", role: str = "requester", **kw):
        return http.client(auth=token(user, tenant, role), **kw)

    return _client


def text_of(result) -> str:
    return result.content[0].text if result.content else ""
