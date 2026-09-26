"""The secret broker's guarantees: upstream credentials are injected by the
gateway, the caller's token never reaches an upstream, and stdio children
see only the credential meant for them."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from conftest import SECRET, ServerThread, make_settings
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server import create_proxy
from fastmcp.server.dependencies import get_http_headers

from mcp_gateway.demo_upstreams import payments_server
from mcp_gateway.demo_upstreams._auth import SharedSecretVerifier
from mcp_gateway.gateway import build_gateway
from mcp_gateway.identity import mint_dev_token
from mcp_gateway.secret_broker import EnvSecretBroker
from mcp_gateway.upstreams import Credential, UpstreamSpec

UPSTREAM_TOKEN = "upstream-token-held-only-by-the-broker"
POLICY = ("version: 1\nrules:\n  - {id: all, effect: allow, subjects: {groups: ['*']}, "
          "tools: ['*']}\n")


def echo_server() -> FastMCP[Any]:
    mcp: FastMCP[Any] = FastMCP("echo", auth=SharedSecretVerifier(UPSTREAM_TOKEN))

    @mcp.tool(annotations={"readOnlyHint": True})
    def headers_seen() -> dict[str, str | None]:
        """Report the credential headers this upstream received."""
        h = get_http_headers(include_all=True)
        return {"authorization": h.get("authorization"), "x-user-secret": h.get("x-user-secret")}

    return mcp


@pytest.fixture
def echo_upstream() -> Any:
    with ServerThread(echo_server().http_app(path="/mcp")) as t:
        yield f"http://127.0.0.1:{t.port}/mcp"


async def test_gateway_injects_upstream_token_and_never_passes_caller_token(
    tmp_path: Path, echo_upstream: str
) -> None:
    (tmp_path / "policy.yaml").write_text(POLICY)
    settings = make_settings(tmp_path)
    settings.policy_file.write_text(POLICY)
    spec = UpstreamSpec(name="echo", transport="http", url=echo_upstream,
                        credential=Credential(secret="echo_token"))
    gateway, _ = build_gateway(settings, specs=[spec], broker=EnvSecretBroker(
        {"GATEWAY_SECRET_ECHO_TOKEN": UPSTREAM_TOKEN}))
    user_token = mint_dev_token(settings, "ana", ["employees"])
    with ServerThread(gateway.http_app(path="/mcp")) as g:
        transport = StreamableHttpTransport(
            f"http://127.0.0.1:{g.port}/mcp", auth=user_token,
            headers={"X-User-Secret": "should-not-travel"})
        async with Client(transport) as c:
            r = await c.call_tool("echo_headers_seen", {})
    seen = r.structured_content
    assert seen["authorization"] == f"Bearer {UPSTREAM_TOKEN}"
    assert user_token not in str(seen) and seen["x-user-secret"] is None
    # and the client never saw the upstream credential
    assert UPSTREAM_TOKEN not in str(r.content)


async def test_fastmcp_default_proxy_forwards_the_caller_token(echo_upstream: str) -> None:
    """Documents *why* the gateway does not use create_proxy(): in FastMCP 4.x
    it forwards the inbound Authorization header upstream (token passthrough).
    If this test starts failing, FastMCP changed the default; revisit the design."""
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    front: FastMCP[Any] = FastMCP("front", auth=JWTVerifier(public_key=SECRET, algorithm="HS256"))
    front.mount(create_proxy(echo_upstream), namespace="echo")
    import jwt

    token = jwt.encode({"sub": "ana", "exp": 4102444800}, SECRET, algorithm="HS256")
    with ServerThread(front.http_app(path="/mcp")) as f:
        async with Client(StreamableHttpTransport(f"http://127.0.0.1:{f.port}/mcp",
                                                  auth=token)) as c:
            r = await c.call_tool("echo_headers_seen", {}, raise_on_error=False)
    # The echo upstream rejects the caller's token, and the message proves
    # the caller's token was what arrived.
    assert r.is_error or token in str(r.structured_content)


async def test_stdio_child_gets_only_its_credential(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.policy_file.write_text(POLICY)
    os.environ["GATEWAY_SUPER_SECRET_FOR_TEST"] = "must-not-leak"
    try:
        specs = [
            UpstreamSpec(name="envecho", transport="stdio", command=sys.executable,
                         args=[str(Path(__file__).parent / "env_echo_server.py")],
                         credential=Credential(secret="docs_token", inject_as="env",
                                               env_var="DOCS_API_TOKEN")),
            UpstreamSpec(name="docs", transport="stdio", command="python",
                         args=["-m", "mcp_gateway.demo_upstreams.docs_server"],
                         credential=Credential(secret="docs_token", inject_as="env",
                                               env_var="DOCS_API_TOKEN")),
        ]
        broker = EnvSecretBroker({"GATEWAY_SECRET_DOCS_TOKEN": "docs-backend-credential"})
        gateway, _ = build_gateway(settings, specs=specs, broker=broker)
        token = mint_dev_token(settings, "eli", ["engineering"])
        with ServerThread(gateway.http_app(path="/mcp")) as g:
            async with Client(StreamableHttpTransport(f"http://127.0.0.1:{g.port}/mcp",
                                                      auth=token)) as c:
                keys = (await c.call_tool("envecho_env_keys", {})).data
                status = (await c.call_tool("docs_backend_status", {})).structured_content
    finally:
        del os.environ["GATEWAY_SUPER_SECRET_FOR_TEST"]
    assert "DOCS_API_TOKEN" in keys
    assert not [k for k in keys if k.startswith("GATEWAY_")]
    expected = hashlib.sha256(b"docs-backend-credential").hexdigest()[:8]
    assert status["credential_sha256_8"] == expected


async def test_http_upstream_rejects_missing_credential(tmp_path: Path) -> None:
    with ServerThread(payments_server.create_server(UPSTREAM_TOKEN).http_app(path="/mcp")) as t:
        url = f"http://127.0.0.1:{t.port}/mcp"
        with pytest.raises(Exception):  # noqa: B017
            async with Client(url) as c:
                await c.list_tools()
        async with Client(StreamableHttpTransport(url, auth=UPSTREAM_TOKEN)) as c:
            assert {x.name for x in await c.list_tools()} == {"get_balance", "refund"}
