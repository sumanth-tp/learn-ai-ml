"""Gateway settings, loaded from environment variables (prefix ``GATEWAY_``) and ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- HTTP server -------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8080
    mcp_path: str = "/mcp"
    public_hostnames: list[str] = Field(default_factory=list)

    # --- Config files --------------------------------------------------------
    upstreams_file: Path = Path("config/upstreams.yaml")
    policy_file: Path = Path("config/policy.yaml")
    state_dir: Path = Path("var")

    # --- Identity (the gateway is an OAuth 2.1 resource server) -------------
    jwt_algorithm: Literal["HS256", "RS256", "ES256"] = "HS256"
    jwt_secret: SecretStr | None = None  # HS256 only: local development and tests
    jwt_public_key: str | None = None  # PEM, for RS256/ES256 with a static key
    jwks_uri: str | None = None  # production: the IdP's JWKS endpoint
    jwt_issuer: str = "https://idp.example.internal"
    jwt_audience: str = "mcp-gateway"
    groups_claim: str = "groups"

    # --- Secret broker -------------------------------------------------------
    secrets_backend: Literal["env", "file"] = "env"
    secrets_dir: Path = Path("/run/secrets")

    # --- Security scanning ---------------------------------------------------
    pin_mode: Literal["tofu", "strict"] = "tofu"
    definition_refresh_seconds: float = 30.0
    max_output_bytes: int = 64_000
    output_injection_action: Literal["block", "annotate"] = "block"
    llm_scanner: bool = False

    # --- Resilience ----------------------------------------------------------
    upstream_timeout_seconds: float = 15.0
    read_retries: int = 2
    retry_base_delay_seconds: float = 0.2
    cache_ttl_seconds: float = 60.0
    cache_max_entries: int = 5_000
    cache_per_user: bool = True
    breaker_failure_threshold: int = 5
    breaker_reset_seconds: float = 30.0

    # --- Operations ----------------------------------------------------------
    metrics_token: SecretStr | None = None
    log_level: str = "INFO"
    log_json: bool = True

    # --- LLM second-opinion scanner (provider-agnostic via LangChain) --------
    llm_provider: str = "fake"  # "fake" (offline) or any init_chat_model provider, e.g. "openai"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 10.0

    @model_validator(mode="after")
    def _check_identity(self) -> Settings:
        if self.jwt_algorithm == "HS256":
            if self.jwt_secret is None or len(self.jwt_secret.get_secret_value()) < 32:
                raise ValueError("GATEWAY_JWT_SECRET must be set (>= 32 chars) for HS256")
        elif not (self.jwt_public_key or self.jwks_uri):
            raise ValueError("RS256/ES256 need GATEWAY_JWT_PUBLIC_KEY or GATEWAY_JWKS_URI")
        return self

    @property
    def audit_path(self) -> Path:
        return self.state_dir / "audit.jsonl"

    @property
    def state_db(self) -> Path:
        return self.state_dir / "state.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
