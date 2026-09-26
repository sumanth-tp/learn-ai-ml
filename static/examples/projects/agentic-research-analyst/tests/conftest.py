"""Shared fixtures. Everything here is offline: no API keys, no network."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ["LANGSMITH_TRACING"] = "false"
for key in ("OPENAI_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"):
    os.environ.pop(key, None)

from research_analyst.checkpoint import memory_checkpointer  # noqa: E402
from research_analyst.config import Settings  # noqa: E402
from research_analyst.deps import Deps, build_deps  # noqa: E402
from research_analyst.service import ResearchService  # noqa: E402

SODIUM_Q = "Should we pilot sodium-ion batteries instead of LFP for our 2027 grid-storage sites?"
HYDROGEN_Q = "How are green hydrogen electrolyser costs expected to change by 2030?"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        mode="offline",
        index_path=tmp_path / "index.json",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        log_json=False,
    )


@pytest.fixture
def deps(settings: Settings) -> Deps:
    return build_deps(settings)


@pytest.fixture
async def service(settings: Settings, deps: Deps):
    async with ResearchService(settings, deps=deps, checkpointer=memory_checkpointer()) as svc:
        yield svc
