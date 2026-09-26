"""Backward-compatibility gate for tool schemas.

Tool schemas are a public API: clients cache them and prompts are tuned to
them. Policy (enforced by ``tests/test_schema_compat.py`` against the
committed ``tests/snapshots/tool_schemas.json``):

* Allowed in a minor release: new tools, new *optional* inputs, new output
  fields, wider enums on outputs, better descriptions.
* Breaking (needs a new tool name such as ``search_tickets_v2`` and a
  deprecation window for the old one): removing a tool, removing or renaming
  an input, making an input required, changing an input's type, removing an
  enum value from an input, removing or retyping an output field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp import Client, FastMCP


def _types(prop: dict[str, Any]) -> set[str]:
    if "type" in prop:
        t = prop["type"]
        return set(t) if isinstance(t, list) else {t}
    out: set[str] = set()
    for sub in prop.get("anyOf", []) + prop.get("oneOf", []):
        out |= _types(sub)
    return out


def _enum(prop: dict[str, Any]) -> set[Any] | None:
    if "enum" in prop:
        return set(prop["enum"])
    values: set[Any] = set()
    for sub in prop.get("anyOf", []):
        if "enum" in sub:
            values |= set(sub["enum"])
    return values or None


async def snapshot(mcp: FastMCP) -> dict[str, Any]:
    """The contract as a client sees it (run with an admin identity to see every tool)."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
    return {
        t.name: {
            "input": t.input_schema,
            "output": t.output_schema,
            "annotations": t.annotations.model_dump(exclude_none=True) if t.annotations else {},
        }
        for t in sorted(tools, key=lambda t: t.name)
    }


def breaking_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for name, before in old.items():
        after = new.get(name)
        if after is None:
            problems.append(f"{name}: tool removed")
            continue
        b_in, a_in = before["input"], after["input"]
        b_props, a_props = b_in.get("properties", {}), a_in.get("properties", {})
        for field, b_prop in b_props.items():
            a_prop = a_props.get(field)
            if a_prop is None:
                problems.append(f"{name}: input '{field}' removed")
                continue
            if _types(b_prop) - _types(a_prop):
                problems.append(f"{name}: input '{field}' type narrowed")
            b_enum, a_enum = _enum(b_prop), _enum(a_prop)
            if b_enum and a_enum is not None and b_enum - a_enum:
                problems.append(f"{name}: input '{field}' lost enum values {b_enum - a_enum}")
        newly_required = set(a_in.get("required", [])) - set(b_in.get("required", []))
        for field in sorted(newly_required):
            problems.append(f"{name}: input '{field}' became required")
        b_out = (before.get("output") or {}).get("properties", {})
        a_out = (after.get("output") or {}).get("properties", {})
        for field, b_prop in b_out.items():
            if field not in a_out:
                problems.append(f"{name}: output '{field}' removed")
            elif _types(b_prop) and _types(b_prop) != _types(a_out[field]):
                problems.append(f"{name}: output '{field}' type changed")
        if before["annotations"].get("destructive_hint") is True and not after["annotations"].get(
            "destructive_hint"
        ):
            problems.append(f"{name}: destructive_hint removed")
    return problems


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
