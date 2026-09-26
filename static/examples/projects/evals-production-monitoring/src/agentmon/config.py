"""Typed configuration, read from environment variables (prefix AGENTMON_) and .env."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when the configuration cannot work (for example a real provider without a key)."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTMON_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # --- models -----------------------------------------------------------------
    llm_provider: Literal["fake", "openai"] = "fake"
    model_name: str = "gpt-4o-mini"
    judge_model_name: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    price_input_per_mtok: float = 0.15
    price_output_per_mtok: float = 0.60

    # --- agent ------------------------------------------------------------------
    prompt_version: Literal["v1", "v2"] = "v1"
    guardrails_enabled: bool = True
    max_agent_steps: int = 6
    tool_timeout_s: float = 5.0
    tool_max_retries: int = 2
    tool_backoff_base_s: float = 0.2
    backend_fault_rate: float = 0.0
    seed_path: Path = Path("data/seed/bank.json")

    # --- storage and outputs ----------------------------------------------------
    db_path: Path = Path("var/agentmon.db")
    out_dir: Path = Path("out")
    golden_seed_path: Path = Path("data/golden/golden.jsonl")
    golden_production_path: Path = Path("out/golden/production.jsonl")
    redteam_path: Path = Path("data/redteam/attacks.jsonl")
    benign_path: Path = Path("data/redteam/benign.jsonl")
    thresholds_path: Path = Path("config/thresholds.toml")
    alerts_path: Path = Path("config/alerts.toml")

    # --- online evaluation ------------------------------------------------------
    sample_random_rate: float = Field(0.10, ge=0.0, le=1.0)
    sample_intent_rates: dict[str, float] = Field(
        default_factory=lambda: {"transfer": 0.5, "out_of_scope": 1.0, "other": 0.5}
    )
    sample_seed: str = "agentmon"
    eval_workers: int = 4
    eval_max_attempts: int = 3
    eval_backoff_base_s: float = 0.5
    judge_timeout_s: float = 20.0
    judge_budget_usd_per_day: float = 2.0
    inprocess_workers: bool = True

    # --- monitoring -------------------------------------------------------------
    slo_latency_p95_ms: float = 4000.0

    # --- observability ----------------------------------------------------------
    service_name: str = "banking-agent"
    otlp_endpoint: str | None = None
    log_level: str = "INFO"
    log_json: bool = True

    @model_validator(mode="after")
    def _check_provider(self) -> Settings:
        if self.llm_provider == "openai" and self.openai_api_key is None and not self.llm_base_url:
            raise ConfigError(
                "AGENTMON_LLM_PROVIDER=openai needs OPENAI_API_KEY "
                "(or AGENTMON_LLM_BASE_URL for an OpenAI-compatible local server)."
            )
        return self

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.price_input_per_mtok + output_tokens * self.price_output_per_mtok
        ) / 1_000_000
