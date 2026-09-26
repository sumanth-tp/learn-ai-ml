"""Shared fixtures. Every test runs offline: fake providers, heuristic judge, tmp state."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragate.config import PipelineConfig
from ragate.models import Chunk, Evidence, RetrievedChunk
from ragate.settings import Settings, get_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    for name in ("PROVIDER", "JUDGE_PROVIDER", "EMBEDDING_PROVIDER"):
        monkeypatch.setenv(f"RAGATE_{name}", "fake")
    monkeypatch.setenv("RAGATE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("RAGATE_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("RAGATE_LOG_LEVEL", "WARNING")
    monkeypatch.chdir(ROOT)
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        state_dir=tmp_path / "state",
        data_dir=ROOT / "data",
        config_dir=ROOT / "config",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(name="test")


def make_chunk(doc_id: str, text: str, rank: int) -> RetrievedChunk:
    chunk = Chunk(chunk_id=f"{doc_id}#{rank:02d}", doc_id=doc_id, title=doc_id, text=text,
                  position=rank)
    return RetrievedChunk(chunk=chunk, score=1.0 / rank, rank=rank)


@pytest.fixture
def leave_evidence() -> list[Evidence]:
    return [Evidence(doc_id="annual-leave", quote="You may carry over a maximum of 5 unused days.")]
