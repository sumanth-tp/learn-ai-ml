"""Policy-as-code: YAML rules deciding who may call which tool, with which arguments.

Semantics (kept deliberately small so a reviewer can hold them in their head):

1. Default deny. Nothing is callable unless an ``allow`` rule matches.
2. Deny overrides allow. A matching ``deny`` rule wins, whatever else matches.
3. An ``allow`` rule matches when the subject (user or group) and the tool
   glob match; it *grants* only if every argument constraint also holds.
4. Constraints fail closed: a constrained argument that is missing, of the
   wrong type or unparsable is a violation.
"""

from __future__ import annotations

import fnmatch
import logging
import math
import posixpath
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Principal:
    subject: str
    groups: frozenset[str]
    email: str | None = None

    @property
    def ids(self) -> set[str]:
        return {self.subject} | ({self.email} if self.email else set())


class Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefix: str | None = None
    max: float | None = None
    min: float | None = None
    enum: list[str | int | float] | None = None
    pattern: str | None = None
    max_length: int | None = None
    optional: bool = False

    @model_validator(mode="after")
    def _compile(self) -> Constraint:
        if self.pattern is not None:
            re.compile(self.pattern)
        if self.prefix is not None and not self.prefix.startswith("/"):
            raise ValueError("prefix constraints must be absolute paths")
        return self


class Subjects(BaseModel):
    model_config = ConfigDict(extra="forbid")

    users: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    effect: Literal["allow", "deny"]
    description: str = ""
    subjects: Subjects
    tools: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    constraints: dict[str, Constraint] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _deny_is_unconditional(self) -> Rule:
        if self.effect == "deny" and self.constraints:
            raise ValueError(f"rule {self.id}: deny rules cannot have constraints")
        if not (self.tools or self.resources):
            raise ValueError(f"rule {self.id}: needs tools or resources")
        return self


class Limit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_minute: int = 60
    per_day: int = 5_000


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: Limit = Field(default_factory=Limit)
    groups: dict[str, Limit] = Field(default_factory=dict)
    tools: dict[str, Limit] = Field(default_factory=dict)


class PolicyDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    rules: list[Rule]
    limits: Limits = Field(default_factory=Limits)

    @model_validator(mode="after")
    def _unique_ids(self) -> PolicyDocument:
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate rule ids")
        return self


@dataclass(frozen=True)
class Decision:
    allowed: bool
    rule_id: str | None
    reason: str

    @property
    def label(self) -> str:
        return "allow" if self.allowed else "deny"


def _subject_matches(rule: Rule, p: Principal) -> bool:
    s = rule.subjects
    if "*" in s.groups:
        return True
    return bool(p.ids & set(s.users)) or bool(p.groups & set(s.groups))


def _glob_any(name: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, g) for g in globs)


def _check_constraint(name: str, c: Constraint, args: dict[str, Any]) -> str | None:
    """Return a violation message, or None when the constraint holds."""
    if name not in args or args[name] is None:
        return None if c.optional else f"argument '{name}' is required by policy"
    value = args[name]

    if c.prefix is not None:
        if not isinstance(value, str) or "\x00" in value:
            return f"'{name}' must be a path string"
        # normpath collapses "a/../b" so "/srv/docs/../../etc/passwd" cannot pass
        normal = posixpath.normpath(value if value.startswith("/") else "/" + value)
        root = c.prefix.rstrip("/")
        if not (normal == root or normal.startswith(root + "/")):
            return f"'{name}' must be under {c.prefix}"

    if c.max is not None or c.min is not None:
        # bool is an int subclass; reject it so True cannot pass as 1
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"'{name}' must be a number"
        if not math.isfinite(value):
            return f"'{name}' must be finite"
        if c.max is not None and value > c.max:
            return f"'{name}'={value} exceeds max {c.max:g}"
        if c.min is not None and value < c.min:
            return f"'{name}'={value} is below min {c.min:g}"

    if c.enum is not None and value not in c.enum:
        return f"'{name}' must be one of {c.enum}"

    if c.pattern is not None and (
        not isinstance(value, str) or re.fullmatch(c.pattern, value) is None
    ):
        return f"'{name}' does not match the allowed pattern"

    if c.max_length is not None and len(str(value)) > c.max_length:
        return f"'{name}' is longer than {c.max_length}"
    return None


class PolicyEngine:
    def __init__(self, doc: PolicyDocument) -> None:
        self.doc = doc

    @classmethod
    def from_yaml(cls, text: str) -> PolicyEngine:
        return cls(PolicyDocument.model_validate(yaml.safe_load(text)))

    def _evaluate(
        self, p: Principal, kind: Literal["tools", "resources"], name: str,
        args: dict[str, Any] | None,
    ) -> Decision:
        matching = [
            r for r in self.doc.rules
            if _subject_matches(r, p) and _glob_any(name, getattr(r, kind))
        ]
        for r in matching:
            if r.effect == "deny":
                return Decision(False, r.id, f"denied by rule '{r.id}'")
        violations: list[str] = []
        for r in matching:  # allow rules, in file order
            if args is None:  # visibility checks and resources: constraints are about tool args
                return Decision(True, r.id, f"allowed by rule '{r.id}'")
            errs = [m for k, c in r.constraints.items() if (m := _check_constraint(k, c, args))]
            if not errs:
                return Decision(True, r.id, f"allowed by rule '{r.id}'")
            violations.append(f"{r.id}: {'; '.join(errs)}")
        if violations:
            return Decision(False, None, "constraint violated: " + " | ".join(violations))
        return Decision(False, None, "no rule allows this (default deny)")

    def check_tool(self, p: Principal, tool: str, args: dict[str, Any]) -> Decision:
        return self._evaluate(p, "tools", tool, args)

    def tool_visible(self, p: Principal, tool: str) -> bool:
        return self._evaluate(p, "tools", tool, None).allowed

    def check_resource(self, p: Principal, uri: str) -> Decision:
        return self._evaluate(p, "resources", uri, None)

    def resource_visible(self, p: Principal, uri: str) -> bool:
        return self._evaluate(p, "resources", uri, None).allowed

    def limits_for(self, p: Principal, tool: str) -> tuple[Limit, Limit | None]:
        """(per-user limit, per-user-per-tool limit or None)."""
        lim = self.doc.limits
        group_limits = [lim.groups[g] for g in sorted(p.groups) if g in lim.groups]
        user_limit = (
            Limit(
                per_minute=max(g.per_minute for g in group_limits),
                per_day=max(g.per_day for g in group_limits),
            )
            if group_limits
            else lim.default
        )
        tool_limit = next(
            (v for k, v in lim.tools.items() if fnmatch.fnmatchcase(tool, k)), None
        )
        return user_limit, tool_limit


class PolicyStore:
    """Holds the active engine and hot-reloads the file when it changes.

    A broken edit never takes effect: the previous, valid policy stays active
    and the error is logged, so a typo cannot open or close the whole gateway.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._mtime = path.stat().st_mtime_ns
        self._engine = PolicyEngine.from_yaml(path.read_text(encoding="utf-8"))
        self.last_error: str | None = None

    @property
    def engine(self) -> PolicyEngine:
        try:
            mtime = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return self._engine
        if mtime != self._mtime:
            with self._lock:
                if mtime != self._mtime:
                    self._mtime = mtime
                    try:
                        self._engine = PolicyEngine.from_yaml(
                            self.path.read_text(encoding="utf-8")
                        )
                        self.last_error = None
                        log.info("policy reloaded from %s", self.path)
                    except Exception as exc:
                        self.last_error = str(exc)
                        log.error("policy reload rejected, keeping previous: %s", exc)
        return self._engine
