"""Shared fixtures: every test gets a fresh database, fakes and in-memory persistence."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from support_agent.config import Settings
from support_agent.container import Container, build_container
from support_agent.graph import build_graph
from support_agent.runner import SupportRunner
from support_agent.services.refunds import StubRefundGateway


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        app_env="test",
        fake_llm=True,
        database_url=f"sqlite:///{tmp_path}/orders.db",
        checkpoint_backend="memory",
        checkpoint_sqlite_path=str(tmp_path / "cp.db"),
        store_sqlite_path=str(tmp_path / "store.db"),
        tool_backoff_initial_s=0.001,
        tool_backoff_max_s=0.002,
        llm_node_retry_initial_s=0.001,
        log_json=False,
    )


@pytest.fixture
def gateway() -> StubRefundGateway:
    return StubRefundGateway()


@pytest.fixture
def container(settings: Settings, gateway: StubRefundGateway) -> Iterator[Container]:
    c = build_container(settings, gateway=gateway)
    yield c
    c.engine.dispose()


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def runner(container: Container, store: InMemoryStore) -> SupportRunner:
    graph = build_graph(container.deps, InMemorySaver(), store)
    return SupportRunner(container, graph)


@pytest.fixture
async def arunner(runner: SupportRunner) -> AsyncIterator[SupportRunner]:
    yield runner
