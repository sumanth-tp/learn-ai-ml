"""Configuration: process settings from the environment, server catalogue from JSON.

Two sources on purpose. Environment variables hold what differs per deployment
(model, keys, paths, limits). ``servers.json`` holds what the operator reviews like
code: which servers exist, how to reach them and what each one may do.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# OpenAI-compatible function names: ^[a-zA-Z0-9_-]{1,64}$. Server names become a
# prefix of every tool name, so they are held to a stricter subset.
SERVER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,15}$")


class Settings(BaseSettings):
    """Process-level settings. Every field maps to an ``HOST_*`` environment variable."""

    model_config = SettingsConfigDict(env_prefix="HOST_", env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = Field(
        default="auto",
        description="auto | fake | any init_chat_model provider (openai, anthropic, ollama...)",
    )
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    # Servers and limits
    servers_file: Path = Path("config/servers.json")
    connect_timeout_s: float = 10.0
    call_timeout_s: float = 20.0
    ping_interval_s: float = 15.0
    ping_timeout_s: float = 5.0
    backoff_initial_s: float = 0.5
    backoff_max_s: float = 30.0
    max_tool_output_chars: int = 4000
    max_agent_steps: int = 12
    sampling_max_tokens: int = 400

    # Persistence and serving
    checkpoint_db: str = "data/state/checkpoints.sqlite"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_key: SecretStr | None = None  # when set, every API call needs "Bearer <key>"

    # Observability
    log_level: str = "INFO"
    log_json: bool = True
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    service_name: str = "mcp-agent-host"


class PolicyConfig(BaseModel):
    """What the host lets the agent do on one server. Glob patterns match raw tool names."""

    allow: list[str] = Field(default_factory=lambda: ["*"])
    deny: list[str] = Field(default_factory=list)
    destructive: list[str] = Field(
        default_factory=list,
        description="Tools that need human approval, in addition to destructiveHint=true",
    )
    trust_annotations: bool = Field(
        default=True,
        description="Honour the server's destructiveHint. Annotations can only ADD approval.",
    )
    allow_sampling: bool = False


class StdioServer(BaseModel):
    transport: Literal["stdio"]
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None


class HttpServer(BaseModel):
    transport: Literal["http"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)


class ServerConfig(BaseModel):
    """One entry of ``servers.json``."""

    connection: Annotated[StdioServer | HttpServer, Field(discriminator="transport")]
    description: str = ""
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    timeout_s: float | None = None
    max_output_chars: int | None = None
    enabled: bool = True


class ServersFile(BaseModel):
    servers: dict[str, ServerConfig]

    @field_validator("servers")
    @classmethod
    def _names_are_prefix_safe(cls, value: dict[str, ServerConfig]) -> dict[str, ServerConfig]:
        for name in value:
            if not SERVER_NAME_RE.match(name):
                raise ValueError(
                    f"server name {name!r} must match {SERVER_NAME_RE.pattern} "
                    "(it prefixes every tool name the LLM sees)"
                )
        return value

    @model_validator(mode="after")
    def _at_least_one_enabled(self) -> ServersFile:
        if not any(s.enabled for s in self.servers.values()):
            raise ValueError("servers.json enables no servers")
        return self


_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand_env(text: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` so one file serves laptop and compose."""

    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name, default)
        if value is None:
            raise ValueError(f"servers file references unset variable ${{{name}}}")
        return value

    return _ENV_REF.sub(repl, text)


def load_servers(path: Path) -> ServersFile:
    raw = _expand_env(path.read_text(encoding="utf-8"))
    return ServersFile.model_validate(json.loads(raw))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
