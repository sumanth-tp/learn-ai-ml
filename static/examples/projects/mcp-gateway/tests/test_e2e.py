"""End-to-end: a real MCP client -> gateway over streamable HTTP with JWTs ->
in-process upstreams. Covers the happy paths and the infrastructure failure paths."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from conftest import Harness, start_harness
from fastmcp import FastMCP

from mcp_gateway.audit import iter_records, verify_chain
from mcp_gateway.upstreams import UpstreamSpec

ENG = ["employees", "engineering"]


def audit(h: Harness) -> list[dict[str, Any]]:
    return list(iter_records(h.settings.audit_path))


async def test_tools_are_namespaced_and_filtered_per_group(harness: Harness) -> None:
    async with harness.client("ana", ["employees"]) as c:
        assert sorted(t.name for t in await c.list_tools()) == [
            "docs_list_docs", "docs_read_doc", "docs_search_docs"]
    async with harness.client("sam", ["support"]) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == {"tickets_search_tickets", "tickets_get_ticket", "tickets_add_comment",
                     "payments_refund"}


async def test_allowed_call_is_audited(harness: Harness) -> None:
    async with harness.client("eli", ENG) as c:
        r = await c.call_tool("docs_read_doc", {"path": "/engineering/runbook.md"})
    assert "Gateway runbook" in r.content[0].text
    rec = audit(harness)[-1]
    assert (rec["user"], rec["target"], rec["upstream"], rec["decision"]) == (
        "eli", "docs_read_doc", "docs", "allow")
    assert rec["rule_id"] == "engineering-read" and rec["result_bytes"] > 0
    assert len(rec["args_sha256"]) == 64 and rec["latency_ms"] > 0
    assert verify_chain(harness.settings.audit_path)[0]


async def test_denied_call_names_the_reason(harness: Harness) -> None:
    async with harness.client("eli", ENG) as c:
        r = await c.call_tool("docs_read_doc", {"path": "/finance/q3-forecast.md"},
                              raise_on_error=False)
    assert r.is_error and "Denied by gateway" in r.content[0].text
    assert "must be under /engineering/" in r.content[0].text
    assert audit(harness)[-1]["decision"] == "deny"


async def test_read_only_results_are_cached_per_user(harness: Harness) -> None:
    args = {"path": "/public/handbook.md"}
    async with harness.client("ana", ["employees"]) as c:
        await c.call_tool("docs_read_doc", args)
        await c.call_tool("docs_read_doc", args)
    async with harness.client("bo", ["employees"]) as c:
        await c.call_tool("docs_read_doc", args)
    hits = [r["cache_hit"] for r in audit(harness) if r["target"] == "docs_read_doc"]
    assert hits == [False, True, False]


async def test_writes_are_not_cached_and_are_idempotent(harness: Harness) -> None:
    args = {"order_id": "O-1", "amount": 25, "currency": "EUR", "idempotency_key": "key-0000001"}
    async with harness.client("sam", ["support"]) as c:
        first = await c.call_tool("payments_refund", args)
        second = await c.call_tool("payments_refund", args)
    assert first.structured_content["replayed"] is False
    assert second.structured_content["replayed"] is True
    assert second.structured_content["refund_id"] == first.structured_content["refund_id"]
    assert not any(r["cache_hit"] for r in audit(harness))


async def test_pii_reaches_authorised_client_but_not_the_audit_log(harness: Harness) -> None:
    async with harness.client("sam", ["support"]) as c:
        r = await c.call_tool("tickets_get_ticket", {"ticket_id": "T-1"})
        await c.call_tool("tickets_add_comment", {"ticket_id": "T-1",
                                                   "comment": "called ana.silva@example.com"})
    assert r.structured_content["email"] == "ana.silva@example.com"
    raw = harness.settings.audit_path.read_text()
    assert "ana.silva@example.com" not in raw and "[REDACTED:EMAIL]" in raw


async def test_resources_are_namespaced_and_policed(harness: Harness) -> None:
    async with harness.client("ana", ["employees"]) as c:
        uris = [str(r.uri) for r in await c.list_resources()]
        assert uris == ["docs://docs/index"]
        content = await c.read_resource("docs://docs/index")
        assert "/public/handbook.md" in content[0].text
    async with harness.client("sam", ["support"]) as c:
        assert await c.list_resources() == []
        with pytest.raises(Exception, match="Denied by gateway"):
            await c.read_resource("docs://docs/index")


async def test_output_cap_truncates_text_results(tmp_path: Path,
                                                 upstream_servers: dict[str, FastMCP]) -> None:
    h, t = start_harness(tmp_path, upstream_servers, max_output_bytes=20)
    try:
        async with h.client("ana", ["employees"]) as c:
            r = await c.call_tool("docs_read_doc", {"path": "/public/handbook.md"})
        assert "[gateway: truncated" in r.content[0].text
        assert len(r.content[0].text.split("\n[gateway")[0].encode()) <= 20
    finally:
        t.__exit__()


async def test_output_cap_blocks_structured_results(tmp_path: Path,
                                                    upstream_servers: dict[str, FastMCP]) -> None:
    h, t = start_harness(tmp_path, upstream_servers, max_output_bytes=20)
    try:
        async with h.client("sam", ["support"]) as c:
            r = await c.call_tool("tickets_get_ticket", {"ticket_id": "T-1"},
                                  raise_on_error=False)
        assert r.is_error and "exceeds the 20-byte cap" in r.content[0].text
    finally:
        t.__exit__()


async def test_timeout_opens_circuit_breaker(tmp_path: Path,
                                             upstream_servers: dict[str, FastMCP]) -> None:
    slow = upstream_servers["docs"]

    @slow.tool(annotations={"readOnlyHint": True})
    async def slow_search(query: str) -> str:
        """A deliberately slow tool."""
        await asyncio.sleep(2)
        return query

    policy = tmp_path / "slow-policy.yaml"
    policy.write_text("version: 1\nrules:\n  - {id: all, effect: allow, "
                      "subjects: {groups: ['*']}, tools: ['docs_*']}\n")
    h, t = start_harness(tmp_path, upstream_servers, upstream_timeout_seconds=0.2,
                         breaker_failure_threshold=2, read_retries=0, policy_file=policy)
    h.settings.policy_file.write_text(policy.read_text())
    try:
        async with h.client("ana", ["employees"]) as c:
            msgs = []
            for _ in range(3):
                r = await c.call_tool("docs_slow_search", {"query": "x"}, raise_on_error=False)
                msgs.append(r.content[0].text)
        assert "unavailable (timeout)" in msgs[0] and "unavailable (timeout)" in msgs[1]
        assert "circuit is open" in msgs[2]
        assert h.components.breakers["docs"].state.name == "OPEN"
    finally:
        t.__exit__()


async def test_unreachable_http_upstream_is_contained(tmp_path: Path,
                                                      upstream_servers: dict[str, FastMCP]) -> None:
    from conftest import free_port, inprocess_specs

    specs = [*inprocess_specs()[:1], UpstreamSpec(
        name="payments", transport="http", url=f"http://127.0.0.1:{free_port()}/mcp",
        required=False)]
    h, t = start_harness(tmp_path, upstream_servers, specs=specs)
    try:
        async with h.client("ana", ["employees"]) as c:
            # one dead upstream must not break listing of the healthy ones
            names = {x.name for x in await c.list_tools()}
            assert "docs_read_doc" in names
        async with h.client("sam", ["support"]) as c:
            r = await c.call_tool("payments_refund", {
                "order_id": "O-1", "amount": 5, "currency": "EUR",
                "idempotency_key": "key-0000009"}, raise_on_error=False)
        assert r.is_error and "upstream 'payments' unavailable" in r.content[0].text
        assert h.components.breakers["payments"].failures >= 1
    finally:
        t.__exit__()
