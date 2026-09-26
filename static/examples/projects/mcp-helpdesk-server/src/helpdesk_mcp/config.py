"""Typed configuration, loaded from environment variables (prefix ``HELPDESK_``).

Every setting has a safe default for local development. Production overrides
them through the environment (or a secrets manager that injects env vars).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AuthMode = Literal["jwt", "local"]
Transport = Literal["http", "stdio"]


class Settings(BaseSettings):
    """Server settings. Field names map to ``HELPDESK_<NAME>`` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="HELPDESK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- runtime -----------------------------------------------------------
    environment: Literal["dev", "test", "prod"] = "dev"
    transport: Transport = "http"
    host: str = "127.0.0.1"
    port: int = 8000
    mcp_path: str = "/mcp"
    # Public URL clients use (https://helpdesk-mcp.example.com). Enables the
    # OAuth protected-resource metadata endpoint so clients can discover the IdP.
    public_base_url: str | None = None
    log_level: str = "INFO"
    log_json: bool = True

    # --- database ----------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./helpdesk.db"
    db_echo: bool = False
    db_pool_size: int = 10

    # --- authentication ----------------------------------------------------
    # "jwt": every HTTP request must carry a valid bearer token.
    # "local": no token; the identity below is used (stdio / single-user only).
    auth_mode: AuthMode = "jwt"
    jwt_algorithm: Literal["HS256", "RS256"] = "HS256"
    jwt_secret: SecretStr = SecretStr("dev-only-secret-change-me-0123456789abcdef")
    jwt_public_key: str | None = None
    jwt_jwks_uri: str | None = None
    jwt_issuer: str = "https://idp.example.internal"
    jwt_audience: str = "helpdesk-mcp"
    local_user: str = "local-admin"
    local_tenant: str = "acme"
    local_roles: list[str] = Field(default_factory=lambda: ["admin"])

    # --- robustness --------------------------------------------------------
    tool_timeout_s: float = 20.0
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 30
    max_page_size: int = 100
    default_page_size: int = 20
    confirm_token_ttl_s: int = 300
    cursor_secret: SecretStr = SecretStr("dev-only-cursor-secret-change-me")

    # --- LLM (triage suggestions) -------------------------------------------
    # "fake" runs a deterministic offline model; anything else is passed to
    # LangChain's init_chat_model as model_provider (openai, anthropic, ...).
    llm_provider: str = "fake"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 15.0
    llm_max_retries: int = 2

    @model_validator(mode="after")
    def _check_prod_safety(self) -> Settings:
        """Refuse to start in prod with dev secrets or without real auth."""
        if self.environment == "prod":
            if self.auth_mode != "jwt":
                raise ValueError("auth_mode must be 'jwt' in prod")
            if self.jwt_algorithm == "HS256" and "dev-only" in self.jwt_secret.get_secret_value():
                raise ValueError("HELPDESK_JWT_SECRET must be set in prod")
            if "dev-only" in self.cursor_secret.get_secret_value():
                raise ValueError("HELPDESK_CURSOR_SECRET must be set in prod")
        loopback = self.host in {"127.0.0.1", "localhost", "::1"}
        if self.auth_mode == "local" and self.transport == "http" and not loopback:
            raise ValueError("auth_mode 'local' over HTTP is only allowed on a loopback host")
        if self.jwt_algorithm == "RS256" and not (self.jwt_public_key or self.jwt_jwks_uri):
            raise ValueError("RS256 needs HELPDESK_JWT_PUBLIC_KEY or HELPDESK_JWT_JWKS_URI")
        return self


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton (tests build their own ``Settings``)."""
    return Settings()
