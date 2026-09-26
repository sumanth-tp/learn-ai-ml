"""Application settings, loaded from environment variables and an optional .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every tunable lives here. Nothing else in the code reads os.environ."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_env: Literal["dev", "test", "prod"] = "dev"

    # --- LLM -----------------------------------------------------------------
    fake_llm: bool = Field(
        default=False, description="Use deterministic offline fakes instead of a provider."
    )
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    llm_node_retry_attempts: int = Field(default=3, ge=1)
    llm_node_retry_initial_s: float = 0.5
    embeddings_model: str = "text-embedding-3-small"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # --- Storage -------------------------------------------------------------
    database_url: str = "sqlite:///./data/support.db"
    checkpoint_backend: Literal["memory", "sqlite", "postgres"] = "sqlite"
    checkpoint_sqlite_path: str = "./data/checkpoints.db"
    store_sqlite_path: str = "./data/store.db"
    postgres_url: str | None = None

    # --- Refund provider -----------------------------------------------------
    refund_gateway: Literal["stub", "http"] = "stub"
    refund_api_url: str = "http://refunds.internal/v1"
    refund_api_key: SecretStr | None = None
    refund_timeout_s: float = 5.0
    refund_approval_threshold: float = 100.0
    stub_refund_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Budgets, memory and retries -----------------------------------------
    max_steps: int = Field(default=8, ge=1)
    max_tokens_per_request: int = Field(default=8000, ge=100)
    max_context_tokens: int = Field(default=3000, ge=200)
    summarise_after_messages: int = Field(default=16, ge=4)
    keep_last_messages: int = Field(default=6, ge=2)
    recursion_limit: int = 40
    tool_max_attempts: int = Field(default=3, ge=1)
    tool_backoff_initial_s: float = 0.2
    tool_backoff_max_s: float = 2.0
    tool_timeout_s: float = 10.0

    # --- API -----------------------------------------------------------------
    api_token: SecretStr = SecretStr("dev-customer-token")
    reviewer_token: SecretStr = SecretStr("dev-reviewer-token")

    # --- Observability -------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "support-agent"

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.checkpoint_backend == "postgres" and not self.postgres_url:
            raise ValueError("CHECKPOINT_BACKEND=postgres needs POSTGRES_URL")
        if not self.fake_llm and self.llm_provider == "openai" and self.openai_api_key is None:
            raise ValueError(
                "LLM_PROVIDER=openai needs OPENAI_API_KEY. Set FAKE_LLM=true to run offline."
            )
        if self.app_env == "prod" and self.api_token.get_secret_value().startswith("dev-"):
            raise ValueError("Refusing to start in prod with the default dev API token")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
