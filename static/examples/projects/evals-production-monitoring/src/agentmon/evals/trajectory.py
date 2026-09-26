"""Agent-specific metrics: tool-call accuracy, trajectory matching, step efficiency,
loop detection.

Trajectory modes (names follow the agentevals convention):
  exact     same calls, same order, nothing extra
  in_order  expected calls appear in order; extra calls allowed in between
  unordered same multiset of calls, any order, nothing extra
  superset  every expected call appears somewhere (extras allowed)
  subset    every actual call is one of the expected calls (no unexpected tools)
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field

TrajectoryMode = Literal["exact", "in_order", "unordered", "superset", "subset"]


class ExpectedCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.lower().split())
    if isinstance(v, int | float) and not isinstance(v, bool):
        return round(float(v), 2)
    return v


def args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Every argument the expectation pins down must match; unpinned args are free.
    Strings compare case- and whitespace-insensitively, numbers to 2 d.p."""
    return all(k in actual and _norm(actual[k]) == _norm(v) for k, v in expected.items())


def call_matches(e: ExpectedCall, a: dict[str, Any]) -> bool:
    return e.name == a["name"] and args_match(e.args, a.get("args", {}))


def tool_call_accuracy(expected: list[ExpectedCall], actual: list[dict[str, Any]]) -> float:
    """Fraction of expected calls matched one-to-one by an actual call (name AND args).
    No expected calls: 1.0 if the agent also made none, else 0.0."""
    if not expected:
        return 1.0 if not actual else 0.0
    remaining = list(actual)
    hits = 0
    for e in expected:
        for i, a in enumerate(remaining):
            if call_matches(e, a):
                hits += 1
                del remaining[i]
                break
    return hits / len(expected)


def _key(name: str, args: dict[str, Any]) -> str:
    return name + json.dumps({k: _norm(v) for k, v in sorted(args.items())})


def trajectory_match(
    expected: list[ExpectedCall],
    actual: list[dict[str, Any]],
    mode: TrajectoryMode,
    with_args: bool = True,
) -> bool:
    def m(e: ExpectedCall, a: dict[str, Any]) -> bool:
        return call_matches(e, a) if with_args else e.name == a["name"]

    if mode == "exact":
        if len(expected) != len(actual):
            return False
        return all(m(e, a) for e, a in zip(expected, actual, strict=True))
    if mode == "in_order":
        it = iter(actual)
        return all(any(m(e, a) for a in it) for e in expected)
    if mode == "unordered":
        if len(expected) != len(actual):
            return False
        if with_args:
            return tool_call_accuracy(expected, actual) == 1.0
        return Counter(e.name for e in expected) == Counter(a["name"] for a in actual)
    if mode == "superset":
        return all(any(m(e, a) for a in actual) for e in expected)
    if mode == "subset":
        return all(any(m(e, a) for e in expected) for a in actual)
    raise ValueError(f"unknown mode {mode}")


def step_efficiency(expected: list[ExpectedCall], actual: list[dict[str, Any]]) -> float:
    """1.0 when the agent used no more tool calls than the reference path."""
    if not actual:
        return 1.0
    return min(1.0, max(len(expected), 1) / len(actual))


def detect_loop(actual: list[dict[str, Any]], max_repeats: int = 2) -> bool:
    """A loop is the same call repeated more than `max_repeats` times, or an A-B-A-B cycle."""
    keys = [_key(a["name"], a.get("args", {})) for a in actual]
    if any(c > max_repeats for c in Counter(keys).values()):
        return True
    return any(
        keys[i] == keys[i + 2] and keys[i + 1] == keys[i + 3] and keys[i] != keys[i + 1]
        for i in range(len(keys) - 3)
    )
