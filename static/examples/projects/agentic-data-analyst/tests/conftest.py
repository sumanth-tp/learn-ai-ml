from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from data_analyst.config import Settings
from data_analyst.executor import WarehouseExecutor
from data_analyst.llm import OfflineAnalystLLM
from data_analyst.service import AnalystService
from data_analyst.warehouse.seed import build_warehouse


@pytest.fixture(autouse=True)
def _no_network_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never send traces and never see a developer's real keys."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(scope="session")
def warehouse_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_warehouse(tmp_path_factory.mktemp("wh") / "warehouse.duckdb")


@pytest.fixture
def settings(tmp_path: Path, warehouse_file: Path) -> Settings:
    shutil.copy(warehouse_file, tmp_path / "warehouse.duckdb")
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_mode="offline",
        data_dir=tmp_path,
        chart_timeout_s=30,
        log_level="WARNING",
    )


@pytest.fixture
def executor(settings: Settings) -> WarehouseExecutor:
    return WarehouseExecutor(settings.warehouse_path, timeout_s=5, row_cap=1000)


@pytest.fixture
def offline_llm() -> OfflineAnalystLLM:
    return OfflineAnalystLLM.from_package()


@pytest.fixture
def service(settings: Settings) -> Iterator[AnalystService]:
    svc = AnalystService.from_settings(
        settings.model_copy(update={"chart_enabled": False}), persistent=False
    )
    yield svc
    svc.close()


def scripted(script: dict[str, list[str]]) -> OfflineAnalystLLM:
    """An offline model whose package script is extended with test-specific answers."""
    base = OfflineAnalystLLM.from_package()
    return OfflineAnalystLLM({**base.script, **script})
