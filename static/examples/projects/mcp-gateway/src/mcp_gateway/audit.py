"""Append-only, tamper-evident audit log.

One JSON line per decision. Each record carries ``prev`` (the previous
record's hash) and ``hash`` (SHA-256 of its own canonical body including
``prev``), so editing or deleting any line breaks the chain from that point
on and ``verify_chain`` reports where. Ship the file to WORM storage (S3
Object Lock, a SIEM) for true immutability; the chain makes tampering
*detectable* even before that.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from mcp_gateway.redaction import redact_text, redact_value

GENESIS = "0" * 64


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def args_hash(args: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(args).encode()).hexdigest()


class AuditRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = Field(default_factory=time.time)
    kind: Literal["tool", "resource", "security"] = "tool"
    user: str
    groups: list[str] = Field(default_factory=list)
    upstream: str | None = None
    target: str  # tool name or resource URI
    args_sha256: str | None = None
    args_preview: Any = None  # redacted and truncated
    decision: Literal["allow", "deny", "error", "alert"]
    reason: str = ""
    rule_id: str | None = None
    result_bytes: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    findings: list[str] = Field(default_factory=list)


def _preview(args: dict[str, Any], limit: int = 256) -> Any:
    text = canonical(redact_value(args))
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS
        last = GENESIS
        with self.path.open("rb") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)["hash"]
        return last

    def write(self, record: AuditRecord, args: dict[str, Any] | None = None) -> AuditRecord:
        if args is not None:
            record.args_sha256 = args_hash(args)
            record.args_preview = _preview(args)
        record.reason = redact_text(record.reason)
        with self._lock:
            body = record.model_dump() | {"prev": self._last}
            digest = hashlib.sha256(canonical(body).encode()).hexdigest()
            body["hash"] = digest
            with self.path.open("a", encoding="utf-8") as f:
                f.write(canonical(body) + "\n")
                f.flush()
            self._last = digest
        return record


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def verify_chain(path: Path) -> tuple[bool, int, str]:
    """Return (ok, records_checked, message)."""
    prev = GENESIS
    n = 0
    for n, rec in enumerate(iter_records(path), start=1):
        claimed = rec.pop("hash", None)
        if rec.get("prev") != prev:
            return False, n, f"record {n}: chain broken (prev mismatch)"
        if hashlib.sha256(canonical(rec).encode()).hexdigest() != claimed:
            return False, n, f"record {n}: content hash mismatch (edited)"
        prev = claimed
    return True, n, f"{n} records verified"
