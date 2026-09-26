"""Secret broker: the only component that can read upstream credentials.

Clients of the gateway never see these values. The gateway resolves a secret
*reference* from ``upstreams.yaml`` at connection time and injects it into the
upstream connection (a header for HTTP, an environment variable for stdio).
Resolving on every connection means a rotated secret is picked up without a
restart when the file backend is used (Kubernetes and Docker mount secrets as
files and update them in place).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Protocol

from pydantic import SecretStr

_REF = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class SecretNotFoundError(LookupError):
    """Raised when a referenced secret does not exist. Never includes the value."""


class SecretBroker(Protocol):
    def get(self, ref: str) -> SecretStr: ...


def _check_ref(ref: str) -> None:
    if not _REF.match(ref):
        raise ValueError(f"invalid secret reference {ref!r}")


class EnvSecretBroker:
    """Reads ``GATEWAY_SECRET_<REF>``. Good for local development and CI."""

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else os.environ

    def get(self, ref: str) -> SecretStr:
        _check_ref(ref)
        value = self._environ.get(f"GATEWAY_SECRET_{ref.upper()}")
        if not value:
            raise SecretNotFoundError(f"secret {ref!r} is not set")
        return SecretStr(value)


class FileSecretBroker:
    """Reads ``<secrets_dir>/<ref>``: Docker/Kubernetes mounted secrets."""

    def __init__(self, secrets_dir: Path) -> None:
        self._dir = secrets_dir

    def get(self, ref: str) -> SecretStr:
        _check_ref(ref)  # the regex also rules out path traversal
        path = self._dir / ref
        try:
            value = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise SecretNotFoundError(f"secret {ref!r} not found") from exc
        if not value:
            raise SecretNotFoundError(f"secret {ref!r} is empty")
        return SecretStr(value)


def build_broker(backend: str, secrets_dir: Path) -> SecretBroker:
    if backend == "file":
        return FileSecretBroker(secrets_dir)
    return EnvSecretBroker()
