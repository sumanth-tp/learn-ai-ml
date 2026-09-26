"""Unit tests for the policy engine: precedence, constraints, fail-closed, reload."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_gateway.policy import PolicyEngine, PolicyStore, Principal

POLICY = """
version: 1
rules:
  - id: eng-docs
    effect: allow
    subjects: {groups: [eng]}
    tools: ["docs_*"]
    constraints:
      path: {prefix: /srv/docs/}
  - id: fin-refund
    effect: allow
    subjects: {groups: [finance]}
    tools: [payments_refund]
    constraints:
      amount: {min: 0.01, max: 100}
      currency: {enum: [EUR]}
  - id: bob-extra
    effect: allow
    subjects: {users: [bob@example.com]}
    tools: [payments_refund]
    constraints:
      amount: {max: 1000}
  - id: no-contractors
    effect: deny
    subjects: {groups: [contractors]}
    tools: ["*"]
limits:
  default: {per_minute: 10, per_day: 100}
  groups:
    vip: {per_minute: 100, per_day: 1000}
  tools:
    payments_*: {per_minute: 2, per_day: 5}
"""


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine.from_yaml(POLICY)


def P(sub: str, *groups: str, email: str | None = None) -> Principal:
    return Principal(sub, frozenset(groups), email)


def test_default_deny(engine: PolicyEngine) -> None:
    d = engine.check_tool(P("x", "marketing"), "docs_read", {"path": "/srv/docs/a"})
    assert not d.allowed and "default deny" in d.reason


def test_allow_with_constraint(engine: PolicyEngine) -> None:
    d = engine.check_tool(P("a", "eng"), "docs_read", {"path": "/srv/docs/a.md"})
    assert d.allowed and d.rule_id == "eng-docs"


@pytest.mark.parametrize("path", [
    "/srv/docs/../secrets/key", "/srv/docsx/a", "/etc/passwd", "srv/docs/../../etc", "/srv/docs\x00/a",
])
def test_prefix_blocks_traversal_and_lookalikes(engine: PolicyEngine, path: str) -> None:
    assert not engine.check_tool(P("a", "eng"), "docs_read", {"path": path}).allowed


def test_missing_constrained_argument_fails_closed(engine: PolicyEngine) -> None:
    d = engine.check_tool(P("a", "eng"), "docs_read", {})
    assert not d.allowed and "required by policy" in d.reason


@pytest.mark.parametrize("amount,ok", [(50, True), (100, True), (100.01, False), (0, False),
                                       ("50", False), (True, False), (float("nan"), False),
                                       (float("inf"), False)])
def test_numeric_constraints(engine: PolicyEngine, amount: object, ok: bool) -> None:
    d = engine.check_tool(P("f", "finance"), "payments_refund", {"amount": amount, "currency": "EUR"})
    assert d.allowed is ok


def test_enum(engine: PolicyEngine) -> None:
    d = engine.check_tool(P("f", "finance"), "payments_refund", {"amount": 5, "currency": "USD"})
    assert not d.allowed and "one of" in d.reason


def test_user_rule_by_email_extends_group(engine: PolicyEngine) -> None:
    bob = P("u-1", "finance", email="bob@example.com")
    assert engine.check_tool(bob, "payments_refund", {"amount": 900, "currency": "EUR"}).allowed
    alice = P("u-2", "finance", email="alice@example.com")
    assert not engine.check_tool(alice, "payments_refund", {"amount": 900, "currency": "EUR"}).allowed


def test_deny_overrides_allow(engine: PolicyEngine) -> None:
    d = engine.check_tool(P("c", "eng", "contractors"), "docs_read", {"path": "/srv/docs/a"})
    assert not d.allowed and d.rule_id == "no-contractors"


def test_visibility_ignores_arguments(engine: PolicyEngine) -> None:
    assert engine.tool_visible(P("a", "eng"), "docs_read")
    assert not engine.tool_visible(P("a", "eng"), "payments_refund")
    assert not engine.tool_visible(P("c", "eng", "contractors"), "docs_read")


def test_limits_resolution(engine: PolicyEngine) -> None:
    user, tool = engine.limits_for(P("a", "eng"), "payments_refund")
    assert (user.per_minute, tool and tool.per_day) == (10, 5)
    user, tool = engine.limits_for(P("v", "vip"), "docs_read")
    assert user.per_minute == 100 and tool is None


@pytest.mark.parametrize("bad", [
    "version: 1\nrules:\n  - {id: a, effect: deny, subjects: {groups: [x]}, tools: [t], constraints: {a: {max: 1}}}",
    "version: 1\nrules:\n  - {id: a, effect: allow, subjects: {groups: [x]}}",
    "version: 1\nrules:\n  - {id: a, effect: allow, subjects: {groups: [x]}, tools: [t], typo_field: 1}",
    "version: 1\nrules:\n  - {id: a, effect: allow, subjects: {groups: [x]}, tools: [t], constraints: {p: {prefix: relative/}}}",
])
def test_invalid_policies_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        PolicyEngine.from_yaml(bad)


def test_store_hot_reload_keeps_last_good(tmp_path: Path) -> None:
    f = tmp_path / "p.yaml"
    f.write_text(POLICY)
    store = PolicyStore(f)
    assert store.engine.tool_visible(P("a", "eng"), "docs_read")
    time.sleep(0.01)
    f.write_text(POLICY.replace("[eng]", "[platform]"))
    assert not store.engine.tool_visible(P("a", "eng"), "docs_read")
    time.sleep(0.01)
    f.write_text("version: 1\nrules: [{id: broken}]")
    assert store.engine.tool_visible(P("a", "platform"), "docs_read")  # previous policy kept
    assert store.last_error is not None
