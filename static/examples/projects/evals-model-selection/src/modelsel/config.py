"""Runtime settings, read from the environment (prefix ``MODELSEL_``) and ``.env``.

Everything that changes between a laptop, CI and production lives here. The model
catalogue, prices and decision weights live in ``data/models.toml`` instead,
because they are reviewed like code and change with every model release.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODELSEL_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    profile: Literal["offline", "real"] = "offline"
    """Which candidate list in models.toml to run. ``offline`` uses fakes only."""

    data_dir: Path = PROJECT_ROOT / "data"
    models_file: Path = PROJECT_ROOT / "data" / "models.toml"
    var_dir: Path = PROJECT_ROOT / "var"
    reports_dir: Path = PROJECT_ROOT / "reports"

    max_concurrency: int = Field(default=8, ge=1, le=256)
    request_timeout_s: float = Field(default=60.0, gt=0)
    max_attempts: int = Field(default=4, ge=1, le=10)
    backoff_initial_s: float = Field(default=0.5, ge=0)
    backoff_max_s: float = Field(default=20.0, ge=0)

    allow_private: bool = False
    """The private test split is only scored when this is true (release decisions)."""

    bootstrap_resamples: int = Field(default=2000, ge=100)
    permutation_resamples: int = Field(default=5000, ge=100)
    seed: int = 7
    judge_samples_for_calibration: int = Field(default=40, ge=1)
    meta_judge_spot_checks: int = Field(default=8, ge=0)

    fake_sleep_scale: float = Field(default=0.0, ge=0.0)
    """Fakes report a simulated latency; multiply it by this to actually sleep."""

    log_level: str = "INFO"
    log_json: bool = True

    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_project: str = Field(default="modelsel", alias="LANGSMITH_PROJECT")

    @property
    def db_path(self) -> Path:
        return self.var_dir / "modelsel.db"

    @property
    def cache_path(self) -> Path:
        return self.var_dir / "llm_cache.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
