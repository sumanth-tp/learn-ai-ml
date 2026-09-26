"""The tool registry: one namespaced, policy-filtered view over every server's tools.

The LLM never sees a raw MCP tool name. It sees ``<server>__<tool>``, so two servers
can both offer ``search`` without one shadowing the other, and the host can always
route a call back to the server that owns it. Policy is applied twice: denied tools
are never shown to the model, and every call is checked again at execution time.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from mcp import types

from mcp_host.settings import PolicyConfig

if TYPE_CHECKING:
    from mcp_host.connection import ServerConnection

log = logging.getLogger(__name__)
SEP = "__"
MAX_NAME = 64
HOST = "host"
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    APPROVE = "needs_approval"


@dataclass(frozen=True)
class ToolEntry:
    qualified: str
    server: str
    tool: str
    description: str
    input_schema: dict[str, Any]
    destructive: bool = False
    read_only: bool = False

    def to_openai(self) -> dict[str, Any]:
        schema = self.input_schema or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.qualified,
                "description": f"[{self.server}] {self.description}".strip()[:1024],
                "parameters": schema,
            },
        }


HOST_TOOLS: tuple[ToolEntry, ...] = (
    ToolEntry(
        qualified="host__list_resources",
        server=HOST,
        tool="list_resources",
        description="List readable MCP resources (documents, notes, calendar views) by URI.",
        input_schema={"type": "object", "properties": {}},
        read_only=True,
    ),
    ToolEntry(
        qualified="host__read_resource",
        server=HOST,
        tool="read_resource",
        description="Read one MCP resource by URI, e.g. docs://doc/leave-policy.",
        input_schema={
            "type": "object",
            "properties": {"uri": {"type": "string", "description": "Resource URI"}},
            "required": ["uri"],
        },
        read_only=True,
    ),
)


def qualify(server: str, tool: str) -> str:
    """``server__tool``, sanitised to the OpenAI name grammar and capped at 64 chars."""
    name = f"{server}{SEP}{_UNSAFE.sub('_', tool)}"
    if len(name) > MAX_NAME:
        digest = hashlib.sha1(tool.encode()).hexdigest()[:6]
        name = f"{name[: MAX_NAME - 7]}_{digest}"
    return name


def matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


@dataclass
class RegistrySnapshot:
    tools: dict[str, ToolEntry] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)  # server -> reason
    missing_tools: dict[str, list[str]] = field(default_factory=dict)  # last known, now offline
    hidden: dict[str, list[str]] = field(default_factory=dict)  # server -> denied tools
    collisions: dict[str, list[str]] = field(default_factory=dict)  # raw name -> servers
    resource_prefixes: dict[str, str] = field(default_factory=dict)  # uri prefix -> server
    version: int = 0


class ToolRegistry:
    def __init__(self, policies: Mapping[str, PolicyConfig]) -> None:
        self.policies = dict(policies)
        self._snap = RegistrySnapshot(tools={t.qualified: t for t in HOST_TOOLS})

    @property
    def snapshot(self) -> RegistrySnapshot:
        return self._snap

    def rebuild(self, connections: Mapping[str, ServerConnection]) -> RegistrySnapshot:
        """Recompute the catalogue. Called on start-up, on list_changed and on state changes."""
        from mcp_host.connection import ServerState

        snap = RegistrySnapshot(
            tools={t.qualified: t for t in HOST_TOOLS}, version=self._snap.version + 1
        )
        owners: dict[str, list[str]] = defaultdict(list)
        for name, conn in connections.items():
            if conn.state != ServerState.READY:
                snap.unavailable[name] = conn.last_error or str(conn.state)
                snap.missing_tools[name] = [qualify(name, t.name) for t in conn.catalogue.tools]
                continue
            policy = self.policies.get(name, PolicyConfig())
            for tool in conn.catalogue.tools:
                owners[tool.name].append(name)
                if not matches(tool.name, policy.allow) or matches(tool.name, policy.deny):
                    snap.hidden.setdefault(name, []).append(tool.name)
                    continue
                entry = self._entry(name, tool, policy)
                if entry.qualified in snap.tools:  # two raw names sanitised to the same string
                    digest = hashlib.sha1(tool.name.encode()).hexdigest()[:6]
                    entry = ToolEntry(
                        **{**entry.__dict__, "qualified": f"{entry.qualified}_{digest}"}
                    )
                snap.tools[entry.qualified] = entry
            for res in conn.catalogue.resources:
                snap.resource_prefixes[str(res.uri)] = name
            for tpl in conn.catalogue.resource_templates:
                snap.resource_prefixes[tpl.uriTemplate.split("{", 1)[0]] = name
        snap.collisions = {raw: s for raw, s in owners.items() if len(s) > 1}
        if snap.collisions:
            log.info(
                "tool name collisions resolved by namespacing",
                extra={"collisions": snap.collisions},
            )
        self._snap = snap
        return snap

    @staticmethod
    def _entry(server: str, tool: types.Tool, policy: PolicyConfig) -> ToolEntry:
        ann = tool.annotations
        hinted = bool(policy.trust_annotations and ann and ann.destructiveHint is True)
        return ToolEntry(
            qualified=qualify(server, tool.name),
            server=server,
            tool=tool.name,
            description=(tool.description or tool.title or tool.name).strip(),
            input_schema=dict(tool.inputSchema),
            # Annotations can only ADD a gate. A server that omits the hint on a
            # dangerous tool is caught by the operator's explicit list.
            destructive=hinted or matches(tool.name, policy.destructive),
            read_only=bool(ann and ann.readOnlyHint),
        )

    def resolve(self, qualified: str) -> ToolEntry | None:
        return self._snap.tools.get(qualified)

    def server_for_uri(self, uri: str) -> str | None:
        best = ""
        for prefix in self._snap.resource_prefixes:
            if uri.startswith(prefix) and len(prefix) > len(best):
                best = prefix
        return self._snap.resource_prefixes.get(best) if best else None

    def decide(self, entry: ToolEntry) -> Decision:
        if entry.server == HOST:
            return Decision.ALLOW
        policy = self.policies.get(entry.server, PolicyConfig())
        if not matches(entry.tool, policy.allow) or matches(entry.tool, policy.deny):
            return Decision.DENY
        return Decision.APPROVE if entry.destructive else Decision.ALLOW

    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.to_openai() for t in self._snap.tools.values()]

    def availability_note(self) -> str:
        """The sentence that tells the model what it cannot do right now."""
        if not self._snap.unavailable:
            return "All configured MCP servers are available."
        parts = []
        for server, reason in sorted(self._snap.unavailable.items()):
            tools = ", ".join(self._snap.missing_tools.get(server, [])) or "no tools known yet"
            parts.append(f"{server} ({reason}; tools: {tools})")
        return (
            "These MCP servers are currently UNAVAILABLE, so their tools cannot be called: "
            + "; ".join(parts)
            + ". If the user needs them, say so plainly instead of guessing."
        )
