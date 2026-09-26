"""Upstream MCP server definitions and the client factories that reach them.

Each upstream becomes one ``ProxyProvider`` mounted under its name as a
namespace, so ``payments`` exposing ``refund`` is served as ``payments_refund``.

Security-relevant choice: the factories return a *plain* ``fastmcp.Client``,
not ``ProxyClient``. In FastMCP 4.x a ``ProxyClient`` (which ``create_proxy``
uses) forwards the caller's inbound headers, *including Authorization*, to
the upstream. That is exactly the token-passthrough anti-pattern the MCP
security guidance forbids. A plain client sends only the headers we set,
which are the per-upstream credential from the secret broker.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from pydantic import BaseModel, Field, field_validator, model_validator

from mcp_gateway.secret_broker import SecretBroker

NAME = re.compile(r"^[a-z][a-z0-9]{1,23}$")


class Credential(BaseModel):
    secret: str
    inject_as: Literal["bearer", "header", "env"] = "bearer"
    header: str = "Authorization"
    env_var: str | None = None


class UpstreamSpec(BaseModel):
    name: str
    transport: Literal["stdio", "http", "inprocess"]
    description: str = ""
    # http
    url: str | None = None
    # stdio
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    # inprocess ("package.module:attribute" of a FastMCP instance)
    target: str | None = None
    credential: Credential | None = None
    cacheable_tools: list[str] = Field(default_factory=list)
    required: bool = True
    timeout_seconds: float | None = None

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        # No underscores: the namespace separator is "_", so "pay_ments_refund"
        # would be ambiguous and could let one upstream shadow another.
        if not NAME.match(v):
            raise ValueError("upstream name must match ^[a-z][a-z0-9]{1,23}$ (no underscores)")
        return v

    @model_validator(mode="after")
    def _check_transport_fields(self) -> UpstreamSpec:
        if self.transport == "http" and not self.url:
            raise ValueError(f"{self.name}: http upstream needs url")
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"{self.name}: stdio upstream needs command")
        if self.transport == "inprocess" and not self.target:
            raise ValueError(f"{self.name}: inprocess upstream needs target")
        if self.credential and self.credential.inject_as == "env":
            if self.transport != "stdio" or not self.credential.env_var:
                raise ValueError(f"{self.name}: env credentials need stdio and env_var")
        return self


class UpstreamsFile(BaseModel):
    upstreams: list[UpstreamSpec]

    @model_validator(mode="after")
    def _unique(self) -> UpstreamsFile:
        names = [u.name for u in self.upstreams]
        if len(names) != len(set(names)):
            raise ValueError("duplicate upstream names")
        return self


_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def expand_env(text: str, environ: dict[str, str] | None = None) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` so one file serves laptop and compose."""
    env = os.environ if environ is None else environ

    def sub(m: re.Match[str]) -> str:
        value = env.get(m.group(1))
        if value is None:
            if m.group(2) is None:
                raise KeyError(f"environment variable {m.group(1)} is not set")
            return m.group(2)
        return value

    return _ENV_REF.sub(sub, text)


def load_upstreams(path: Path) -> list[UpstreamSpec]:
    data = yaml.safe_load(expand_env(path.read_text(encoding="utf-8"))) or {}
    return UpstreamsFile.model_validate(data).upstreams


ClientFactory = Callable[[], Client[Any]]


def _import_target(target: str) -> FastMCP[Any]:
    module_name, _, attr = target.partition(":")
    server = getattr(importlib.import_module(module_name), attr)
    if not isinstance(server, FastMCP):
        raise TypeError(f"{target} is not a FastMCP server")
    return server


def build_client_factory(
    spec: UpstreamSpec,
    broker: SecretBroker,
    *,
    default_timeout: float,
    server_override: FastMCP[Any] | None = None,
) -> ClientFactory:
    """Return a zero-argument factory producing a fresh, credentialed client."""
    timeout = spec.timeout_seconds or default_timeout
    cred = spec.credential

    if spec.transport == "inprocess":
        server = server_override or _import_target(spec.target or "")
        return lambda: Client(server, timeout=timeout)

    if spec.transport == "http":
        url = spec.url or ""

        def http_factory() -> Client[Any]:
            headers: dict[str, str] = {}
            if cred is not None:
                value = broker.get(cred.secret).get_secret_value()  # resolved per connection
                if cred.inject_as == "bearer":
                    headers["Authorization"] = f"Bearer {value}"
                else:
                    headers[cred.header] = value
            return Client(StreamableHttpTransport(url, headers=headers), timeout=timeout)

        return http_factory

    # stdio: one long-lived child process. The MCP SDK passes the child only a
    # small allowlist of the parent's environment (PATH, HOME, ...), plus what
    # we give it here, so the gateway's own secrets never leak to the child.
    env: dict[str, str] = {}
    if cred is not None and cred.env_var:
        env[cred.env_var] = broker.get(cred.secret).get_secret_value()
    command = sys.executable if spec.command == "python" else (spec.command or "")
    transport = StdioTransport(
        command=command, args=spec.args, env=env, cwd=spec.cwd, keep_alive=True
    )
    return lambda: Client(transport, timeout=timeout)


def upstream_of(tool_or_uri: str, names: set[str]) -> str | None:
    """Map a namespaced tool name (``payments_refund``) or resource URI
    (``docs://docs/index``) back to its upstream name."""
    if "://" in tool_or_uri:
        rest = tool_or_uri.split("://", 1)[1]
        candidate = rest.split("/", 1)[0]
    else:
        candidate = tool_or_uri.split("_", 1)[0]
    return candidate if candidate in names else None
