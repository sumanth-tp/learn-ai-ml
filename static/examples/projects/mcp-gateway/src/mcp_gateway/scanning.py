"""Security scanning: tool pinning (rug-pull detection), suspicious-instruction
detection in descriptions and outputs, and output size caps."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from mcp_gateway.audit import canonical


def fingerprint(tool: Any) -> str:
    """SHA-256 over everything the model reads or the client relies on.

    Includes the description, input schema, output schema, title and
    annotations: a rug pull can hide in any of them (for example a new
    ``notes`` parameter whose schema description carries the instruction).
    """
    annotations = getattr(tool, "annotations", None)
    # The name is the pin's key, not part of its value, so the namespaced
    # listing ("docs_read_doc") and the upstream's own ("read_doc") agree.
    body = {
        "title": getattr(tool, "title", None),
        "description": getattr(tool, "description", None) or "",
        "input_schema": getattr(tool, "parameters", None) or {},
        "output_schema": getattr(tool, "output_schema", None),
        "annotations": annotations.model_dump(mode="json") if annotations is not None else None,
    }
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def schema_text(tool: Any) -> str:
    """Every human-readable string the model will see for this tool."""
    parts = [getattr(tool, "description", None) or ""]

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in {"description", "title", "default"} and isinstance(v, str):
                    parts.append(v)
                elif k == "enum" and isinstance(v, list):
                    parts.extend(str(x) for x in v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(getattr(tool, "parameters", None) or {})
    return "\n".join(parts)


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("override", re.compile(
        r"(?i)\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all|earlier)\b.{0,20}"
        r"\b(instructions?|rules?|prompts?|guidelines)\b")),
    ("hidden_directive", re.compile(r"(?i)<\s*(important|system|secret|instructions?)\s*>")),
    ("concealment", re.compile(
        r"(?i)\b(do not|don't|never)\b.{0,30}\b(tell|mention|reveal|inform|show)\b.{0,30}"
        r"\b(user|human|anyone)\b")),
    ("role_hijack", re.compile(
        r"(?i)\b(you are now|act as|new instructions|system prompt|developer mode)\b")),
    ("sensitive_path", re.compile(
        r"(?i)(~/\.ssh|id_rsa|\.aws/credentials|/etc/passwd|\.env\b|mcp\.json|"
        r"claude_desktop_config)")),
    ("exfiltration", re.compile(
        r"(?i)\b(send|post|upload|forward|include|append)\b.{0,60}"
        r"(https?://|webhook|\bcurl\b|\bto the (url|endpoint|address)\b)")),
    ("tool_chaining", re.compile(
        r"(?i)\b(before|after|instead of)\b.{0,20}\b(using|calling)\b.{0,40}\b(tool|function)\b")),
    ("secret_request", re.compile(
        r"(?i)\b(pass|provide|include|read)\b.{0,30}\b(api[_ ]?key|password|token|credentials?|"
        r"private key)\b")),
]

_INVISIBLE = {"Cf", "Co"}  # format chars (zero-width, bidi) and private use


def scan_text(text: str) -> list[Finding]:
    findings: list[Finding] = []
    normal = unicodedata.normalize("NFKC", text)
    for code, pattern in _RULES:
        m = pattern.search(normal)
        if m:
            findings.append(Finding(code, m.group(0)[:80]))
    hidden = [c for c in text if unicodedata.category(c) in _INVISIBLE and c not in "‍"]
    tags = [c for c in text if 0xE0000 <= ord(c) <= 0xE007F]  # "ASCII smuggling" tag chars
    if hidden or tags:
        findings.append(Finding("invisible_chars", f"{len(hidden) + len(tags)} hidden characters"))
    return findings


def cross_server_refs(text: str, own_upstream: str, all_upstreams: set[str]) -> list[Finding]:
    """Flag descriptions that talk about *other* servers' tools (shadowing)."""
    out = []
    for other in sorted(all_upstreams - {own_upstream}):
        if re.search(rf"\b{re.escape(other)}_\w+", text):
            out.append(Finding("cross_server_reference", f"mentions {other}_* tools"))
    return out


def result_text(result: Any) -> str:
    """Flatten a ToolResult / ResourceResult to the text a model would read."""
    chunks: list[str] = []
    for item in getattr(result, "content", None) or getattr(result, "contents", None) or []:
        for attr in ("text", "content"):
            value = getattr(item, attr, None)
            if isinstance(value, str):
                chunks.append(value)
                break
            if isinstance(value, bytes):
                chunks.append(value.decode("utf-8", "replace"))
                break
    structured = getattr(result, "structured_content", None)
    if structured:
        chunks.append(canonical(structured))
    return "\n".join(chunks)


def result_size(result: Any) -> int:
    return len(result_text(result).encode("utf-8"))
