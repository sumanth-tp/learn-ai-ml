from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentmon.agent.backend import FakeBankBackend
from agentmon.clock import SimClock
from agentmon.config import Settings
from agentmon.runtime import Runtime, build_runtime
from agentmon.store import Store

ROOT = Path(__file__).resolve().parents[1]
T0 = 1_790_000_000.0


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="fake",
        db_path=tmp_path / "test.db",
        out_dir=tmp_path / "out",
        seed_path=ROOT / "data/seed/bank.json",
        golden_seed_path=ROOT / "data/golden/golden.jsonl",
        golden_production_path=tmp_path / "golden/production.jsonl",
        redteam_path=ROOT / "data/redteam/attacks.jsonl",
        benign_path=ROOT / "data/redteam/benign.jsonl",
        thresholds_path=ROOT / "config/thresholds.toml",
        alerts_path=ROOT / "config/alerts.toml",
        eval_backoff_base_s=0.0,
        tool_backoff_base_s=0.0,
        log_json=False,
    )


@pytest.fixture
def seed() -> dict[str, Any]:
    return json.loads((ROOT / "data/seed/bank.json").read_text())


@pytest.fixture
def backend(seed: dict[str, Any]) -> FakeBankBackend:
    return FakeBankBackend(seed)


@pytest.fixture
def clock() -> SimClock:
    return SimClock(T0)


@pytest.fixture
def rt(settings: Settings, clock: SimClock, backend: FakeBankBackend) -> Runtime:
    return build_runtime(settings, clock=clock, store=Store(":memory:"), backend=backend)
