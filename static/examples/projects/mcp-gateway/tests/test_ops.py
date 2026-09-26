"""Operations: health, readiness, metrics, and the admin CLI."""

from __future__ import annotations

from pathlib import Path

import httpx2
import pytest
from conftest import SECRET, Harness, free_port, inprocess_specs, start_harness
from fastmcp import FastMCP

from mcp_gateway import admin
from mcp_gateway.upstreams import UpstreamSpec


def base(h: Harness) -> str:
    return h.url.removesuffix("/mcp")


async def test_health_and_readiness(harness: Harness) -> None:
    async with httpx2.AsyncClient() as http:
        assert (await http.get(base(harness) + "/healthz")).json() == {"status": "ok"}
        r = await http.get(base(harness) + "/readyz")
    assert r.status_code == 200, r.json()
    assert r.json()["upstreams"] == {"docs": "ok", "payments": "ok", "tickets": "ok"}


async def test_not_ready_when_required_upstream_down(
    tmp_path: Path, upstream_servers: dict[str, FastMCP]
) -> None:
    specs = [
        *inprocess_specs()[:2],
        UpstreamSpec(name="tickets", transport="http", url=f"http://127.0.0.1:{free_port()}/mcp"),
    ]
    h, t = start_harness(tmp_path, upstream_servers, specs=specs)
    try:
        async with httpx2.AsyncClient() as http:
            r = await http.get(base(h) + "/readyz")
        assert r.status_code == 503 and r.json()["upstreams"]["tickets"].startswith("unreachable")
    finally:
        t.__exit__()


async def test_metrics_exposed_and_protected(
    tmp_path: Path, upstream_servers: dict[str, FastMCP]
) -> None:
    h, t = start_harness(tmp_path, upstream_servers, metrics_token="metrics-secret-123")
    try:
        async with h.client("ana", ["employees"]) as c:
            await c.call_tool("docs_read_doc", {"path": "/public/handbook.md"})
            await c.call_tool("docs_read_doc", {"path": "/hr/salaries.csv"}, raise_on_error=False)
        async with httpx2.AsyncClient() as http:
            assert (await http.get(base(h) + "/metrics")).status_code == 401
            body = (
                await http.get(
                    base(h) + "/metrics", headers={"Authorization": "Bearer metrics-secret-123"}
                )
            ).text
        assert (
            'mcp_gateway_calls_total{decision="allow",target="docs_read_doc",upstream="docs"} 1.0'
            in body
        )
        assert 'mcp_gateway_denials_total{reason="policy"} 1.0' in body
        assert "mcp_gateway_upstream_latency_seconds_bucket" in body
    finally:
        t.__exit__()


@pytest.fixture
def admin_env(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> Harness:
    monkeypatch.setenv("GATEWAY_JWT_SECRET", SECRET)
    monkeypatch.setenv("GATEWAY_STATE_DIR", str(harness.settings.state_dir))
    monkeypatch.setenv("GATEWAY_POLICY_FILE", str(harness.settings.policy_file))
    return harness


async def test_admin_cli(admin_env: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    async with admin_env.client("ana", ["employees"]) as c:
        await c.call_tool(
            "docs_read_doc", {"path": "/finance/q3-forecast.md"}, raise_on_error=False
        )
    assert admin.main(["denials"]) == 0
    out = capsys.readouterr().out
    assert "ana" in out and "docs_read_doc" in out
    assert admin.main(["verify-audit"]) == 0
    assert "OK" in capsys.readouterr().out
    assert admin.main(["pins"]) == 0
    assert "docs_read_doc" in capsys.readouterr().out
    assert admin.main(["policies"]) == 0
    assert "contractors-no-payments" in capsys.readouterr().out
    assert admin.main(["upstreams"]) == 0
    assert "payments_token (bearer)" in capsys.readouterr().out
    assert (
        admin.main(
            [
                "check",
                "--user",
                "sam",
                "--groups",
                "support",
                "--tool",
                "payments_refund",
                "--args",
                '{"amount": 500, "currency": "EUR", "idempotency_key": "k-12345678"}',
            ]
        )
        == 1
    )
    assert "exceeds max 100" in capsys.readouterr().out
    assert admin.main(["validate-policy"]) == 0
    assert admin.main(["mint-token", "--sub", "x", "--groups", "a,b"]) == 0
    assert capsys.readouterr().out.count(".") >= 2


def test_admin_detects_tampering(admin_env: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    path = admin_env.settings.audit_path
    path.parent.mkdir(parents=True, exist_ok=True)
    from mcp_gateway.audit import AuditLog, AuditRecord

    log = AuditLog(path)
    log.write(AuditRecord(user="u", target="t", decision="deny"))
    log.write(AuditRecord(user="u", target="t", decision="deny"))
    lines = path.read_text().splitlines()
    path.write_text(lines[1] + "\n")  # someone deleted the first record
    assert admin.main(["verify-audit"]) == 2
    assert "TAMPERED" in capsys.readouterr().out
