"""Shared fixtures. Every test runs offline: fake models, temporary data and databases."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from modelsel.config import PROJECT_ROOT, Settings
from modelsel.dataset import write_benchmark
from modelsel.harness.cache import ResponseCache
from modelsel.harness.client import LLMClient
from modelsel.llm.registry import Catalogue, ModelSpec, load_catalogue

# Make sure no test can reach a real provider by accident.
for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LANGSMITH_API_KEY"):
    os.environ.pop(var, None)
os.environ["LANGSMITH_TRACING"] = "false"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    data.mkdir()
    write_benchmark(data, seed=7)
    return Settings(
        data_dir=data,
        models_file=PROJECT_ROOT / "data" / "models.toml",
        var_dir=tmp_path / "var",
        reports_dir=tmp_path / "reports",
        bootstrap_resamples=300,
        permutation_resamples=500,
        max_attempts=3,
        backoff_initial_s=0.0,
        backoff_max_s=0.0,
        request_timeout_s=2.0,
        judge_samples_for_calibration=20,
        meta_judge_spot_checks=4,
        log_json=False,
    )


@pytest.fixture
def catalogue(settings: Settings) -> Catalogue:
    cat = load_catalogue(settings.models_file)
    cat.models["fake:scripted"] = ModelSpec(
        id="fake:scripted", family="test", input_per_mtok=1.0, output_per_mtok=2.0, rpm=60000
    )
    return cat


@pytest.fixture
def client(settings: Settings, catalogue: Catalogue) -> LLMClient:
    cache = ResponseCache(settings.cache_path)
    yield LLMClient(settings, catalogue, cache)
    cache.close()
