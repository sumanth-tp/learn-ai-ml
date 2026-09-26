"""Application settings, loaded from environment variables (prefix ``RA_``) and ``.env``."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
BUNDLED_CORPUS_DIR = PACKAGE_DIR / "corpus"


class Settings(BaseSettings):
    """Every tunable of the system. Nothing else reads ``os.environ`` directly."""

    model_config = SettingsConfigDict(
        env_prefix="RA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- providers ---------------------------------------------------------------
    mode: Literal["offline", "live"] = Field(
        default="offline",
        description="offline = deterministic fakes for LLM, embeddings and web search.",
    )
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_retries: int = 2
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    tavily_api_key: SecretStr | None = Field(default=None, validation_alias="TAVILY_API_KEY")
    web_search_url: str = "https://api.tavily.com/search"

    # --- data --------------------------------------------------------------------
    corpus_dir: Path = BUNDLED_CORPUS_DIR
    index_path: Path = Path("data/index.json")
    checkpoint_db: Path = Path("data/checkpoints.sqlite")

    # --- research behaviour --------------------------------------------------------
    max_sub_questions: int = Field(default=4, ge=1, le=8)
    retrieval_k: int = Field(default=4, ge=1, le=20)
    web_k: int = Field(default=4, ge=1, le=10)
    crag_upper: float = Field(default=0.6, ge=0, le=1)
    crag_lower: float = Field(default=0.25, ge=0, le=1)
    max_query_rewrites: int = Field(default=1, ge=0, le=3)
    min_source_quality: float = Field(default=0.3, ge=0, le=1)
    max_revisions: int = Field(default=2, ge=0, le=5)
    critic_pass_score: float = Field(default=4.0, ge=1, le=5)
    max_parallel_workers: int = Field(default=4, ge=1, le=16)

    # --- budgets, timeouts ---------------------------------------------------------
    max_cost_usd: float = Field(default=0.10, gt=0)
    max_tokens: int = Field(default=120_000, gt=0)
    soft_budget_ratio: float = Field(default=0.8, gt=0, le=1)
    search_cost_usd: float = Field(default=0.008, ge=0)
    worker_timeout_s: float = Field(default=60.0, gt=0)
    node_timeout_s: float = Field(default=90.0, gt=0)
    claim_check_timeout_s: float = Field(default=20.0, gt=0)
    http_timeout_s: float = Field(default=15.0, gt=0)

    # --- API ---------------------------------------------------------------------
    api_key: SecretStr | None = None
    max_concurrent_reports: int = Field(default=4, ge=1)
    report_timeout_s: float = Field(default=600.0, gt=0)

    # --- observability -----------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    langsmith_tracing: bool = False
    langsmith_project: str = "research-analyst"
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.crag_lower >= self.crag_upper:
            raise ValueError("RA_CRAG_LOWER must be below RA_CRAG_UPPER")
        if self.mode == "live" and self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("RA_MODE=live with the openai provider needs OPENAI_API_KEY")
        return self

    def apply_tracing_env(self) -> None:
        """LangSmith reads its own env vars; mirror our settings into them."""
        if self.langsmith_tracing and self.langsmith_api_key:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key.get_secret_value()
        else:
            os.environ["LANGSMITH_TRACING"] = "false"


@lru_cache
def get_settings() -> Settings:
    return Settings()
