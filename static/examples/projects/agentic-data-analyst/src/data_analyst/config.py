"""Application settings, loaded from the environment (prefix ``ANALYST_``) and ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every tunable in one typed place. Provider keys keep their standard names."""

    model_config = SettingsConfigDict(env_prefix="ANALYST_", env_file=".env", extra="ignore")

    # --- LLM and embeddings -------------------------------------------------
    llm_mode: Literal["offline", "real"] = "offline"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"

    # --- storage -----------------------------------------------------------
    data_dir: Path = Path("data")
    warehouse_file: str = "warehouse.duckdb"
    checkpoint_file: str = "checkpoints.sqlite"
    cache_file: str = "semantic_cache.sqlite"

    # --- agent behaviour -----------------------------------------------------
    schema_top_k: int = Field(default=3, ge=1, le=20)
    max_retries: int = Field(default=3, ge=0, le=10)
    history_turns: int = Field(default=5, ge=0, le=50)

    # --- SQL safety and execution ------------------------------------------
    default_limit: int = 200
    max_limit: int = 1000
    row_cap: int = 1000
    query_timeout_s: float = 10.0
    duckdb_memory_limit: str = "512MB"
    duckdb_threads: int = 2

    # --- human approval ----------------------------------------------------
    approval_row_threshold: int = 100_000
    approval_work_threshold: int = 5_000_000

    # --- semantic cache ----------------------------------------------------
    cache_enabled: bool = True
    cache_similarity: float = Field(default=0.93, ge=0.0, le=1.0)
    cache_ttl_s: int = 7 * 24 * 3600

    # --- charts ------------------------------------------------------------
    chart_enabled: bool = True
    chart_timeout_s: float = 20.0
    chart_memory_mb: int = 1024

    # --- cost accounting (USD per million tokens, gpt-4o-mini list price) ---
    price_input_per_mtok: float = 0.15
    price_output_per_mtok: float = 0.60

    # --- service -----------------------------------------------------------
    api_key: SecretStr | None = None
    log_level: str = "INFO"
    log_json: bool = False

    @property
    def warehouse_path(self) -> Path:
        return self.data_dir / self.warehouse_file

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / self.checkpoint_file

    @property
    def cache_path(self) -> Path:
        return self.data_dir / self.cache_file

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.price_input_per_mtok + output_tokens * self.price_output_per_mtok
        ) / 1_000_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
