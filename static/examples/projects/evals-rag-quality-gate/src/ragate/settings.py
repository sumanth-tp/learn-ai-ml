"""Process-level settings, read from the environment (and an optional .env file).

Pipeline *behaviour* (chunk size, k, reranker...) lives in YAML under config/ so that
it can be versioned, diffed in a PR and hashed into every eval run. Settings here are
only about *where* and *with what credentials* the process runs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["fake", "openai", "anthropic", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGATE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    provider: Provider = "fake"
    chat_model: str = "gpt-4o-mini"
    embedding_provider: Provider = "fake"
    embedding_model: str = "text-embedding-3-small"

    judge_provider: Provider = "fake"
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    request_timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)

    data_dir: Path = Path("data")
    state_dir: Path = Path(".ragate")
    config_dir: Path = Path("config")

    log_level: str = "INFO"
    log_json: bool = False

    # Read without the RAGATE_ prefix: these are the providers' own conventions.
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL"
    )

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def golden_dir(self) -> Path:
        return self.data_dir / "golden"

    @property
    def index_dir(self) -> Path:
        return self.state_dir / "index"

    @property
    def runs_db(self) -> Path:
        return self.state_dir / "runs.db"

    @property
    def judge_cache_db(self) -> Path:
        return self.state_dir / "judge_cache.db"

    @property
    def is_offline(self) -> bool:
        return {self.provider, self.judge_provider, self.embedding_provider} == {"fake"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
