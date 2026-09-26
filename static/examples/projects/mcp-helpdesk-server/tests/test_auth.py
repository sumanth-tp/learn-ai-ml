"""Authentication over the real HTTP stack: tokens, claims and role-based visibility."""

from __future__ import annotations

import pytest
from fastmcp.utilities.tests import ASGIServer

from helpdesk_mcp.config import Settings
from helpdesk_mcp.tokens import mint_token
from tests.conftest import text_of


async def test_no_token_is_401_with_bearer_challenge(http: ASGIServer) -> None:
    async with http.http_client() as h:
        r = await h.post(http.url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")


@pytest.mark.parametrize(
    "overrides",
    [
        {"secret": "a-completely-different-secret-value-123456"},  # forged signature
        {"audience": "some-other-api"},  # token for another service
        {"issuer": "https://evil.example"},  # wrong issuer
        {"ttl_s": -10},  # expired
    ],
    ids=["bad-signature", "wrong-audience", "wrong-issuer", "expired"],
)
async def test_invalid_tokens_rejected(http: ASGIServer, settings: Settings, overrides) -> None:
    bad = mint_token(settings, user="alice", tenant="acme", roles=["admin"], **overrides)
    async with http.http_client(headers={"Authorization": f"Bearer {bad}"}) as h:
        r = await h.post(http.url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert r.status_code == 401


async def test_token_without_tenant_claim_is_denied(http: ASGIServer, settings: Settings) -> None:
    import time

    import jwt

    now = int(time.time())
    tok = jwt.encode(
        {
            "sub": "mallory",
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "iat": now,
            "exp": now + 60,
            "roles": ["admin"],
        },
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )
    async with http.client(auth=tok) as c:
        r = await c.call_tool("search_tickets", {}, raise_on_error=False)
    assert r.is_error and "[permission_denied]" in text_of(r)


async def test_tool_visibility_follows_role(as_user) -> None:
    async with as_user("alice", role="requester") as c:
        requester = {t.name for t in await c.list_tools()}
    async with as_user("sam", role="agent") as c:
        agent = {t.name for t in await c.list_tools()}
    async with as_user("ada", role="admin") as c:
        admin = {t.name for t in await c.list_tools()}
    assert requester == {"create_ticket", "get_ticket", "search_tickets", "add_comment"}
    assert "update_ticket" in agent and "delete_ticket" not in agent
    assert "delete_ticket" in admin and len(admin) == 9


async def test_hidden_tool_is_still_enforced(as_user) -> None:
    """Hiding a tool is UX; calling it by name must still be refused."""
    async with as_user("alice", role="requester") as c:
        r = await c.call_tool(
            "update_ticket", {"ticket_id": 1, "priority": "p1"}, raise_on_error=False
        )
        assert r.is_error and "[permission_denied]" in text_of(r)
    async with as_user("sam", role="agent") as c:
        r = await c.call_tool("delete_ticket", {"ticket_id": 1}, raise_on_error=False)
        assert r.is_error and "needs role 'admin'" in text_of(r)


async def test_unknown_roles_default_to_requester(as_user) -> None:
    async with as_user("eve", role="superuser") as c:
        names = {t.name for t in await c.list_tools()}
    assert "update_ticket" not in names


def test_prod_refuses_dev_secrets() -> None:
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings(_env_file=None, environment="prod")


def test_local_mode_refused_on_public_interface() -> None:
    with pytest.raises(ValueError, match="loopback"):
        Settings(_env_file=None, auth_mode="local", host="0.0.0.0")  # noqa: S104
