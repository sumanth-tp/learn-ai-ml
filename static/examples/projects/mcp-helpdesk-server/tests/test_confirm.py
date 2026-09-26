"""Destructive-action confirmation on all three client kinds."""

from __future__ import annotations

from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult

from tests.conftest import text_of


def handler(answer: bool | None, seen: list[str]):
    """An elicitation handler that plays a user clicking yes, no, or cancel."""

    async def _h(message, response_type, params, context):
        seen.append(message)
        if answer is None:
            return ElicitResult(action="cancel")
        if answer is False:
            return ElicitResult(action="decline")
        props = params.requested_schema.get("properties", {})
        return {"confirm": True} if "confirm" in props else {"value": True}

    return _h


async def _exists(c: Client, ticket_id: int) -> bool:
    r = await c.call_tool("get_ticket", {"ticket_id": ticket_id}, raise_on_error=False)
    return not r.is_error


async def test_modern_protocol_uses_input_required_round_trip(local_app) -> None:
    seen: list[str] = []
    async with Client(local_app.mcp, elicitation_handler=handler(True, seen)) as c:
        assert c.protocol_version == "2026-07-28"
        r = await c.call_tool("delete_ticket", {"ticket_id": 4})
        assert r.structured_content["status"] == "deleted"
        assert not await _exists(c, 4)
    assert seen == ["Permanently delete ticket #4 'Monitor flickers'?"]


async def test_modern_protocol_decline_keeps_ticket(local_app) -> None:
    async with Client(local_app.mcp, elicitation_handler=handler(False, [])) as c:
        r = await c.call_tool("delete_ticket", {"ticket_id": 4})
        assert r.structured_content["status"] == "cancelled"
        assert await _exists(c, 4)


async def test_legacy_protocol_uses_ctx_elicit(local_app) -> None:
    seen: list[str] = []
    async with Client(local_app.mcp, elicitation_handler=handler(True, seen), mode="legacy") as c:
        assert c.protocol_version == "2025-11-25"
        r = await c.call_tool("delete_ticket", {"ticket_id": 3})
        assert r.structured_content["status"] == "deleted"
    assert len(seen) == 1


async def test_legacy_cancel_keeps_ticket(local_app) -> None:
    async with Client(local_app.mcp, elicitation_handler=handler(None, []), mode="legacy") as c:
        r = await c.call_tool("delete_ticket", {"ticket_id": 3})
        assert r.structured_content["status"] == "cancelled"
        assert await _exists(c, 3)


async def test_confirm_token_is_single_use(local_client: Client) -> None:
    first = (await local_client.call_tool("delete_ticket", {"ticket_id": 2})).structured_content
    tok = first["confirm_token"]
    done = await local_client.call_tool("delete_ticket", {"ticket_id": 2, "confirm_token": tok})
    assert done.structured_content["status"] == "deleted"
    again = await local_client.call_tool(
        "delete_ticket", {"ticket_id": 2, "confirm_token": tok}, raise_on_error=False
    )
    assert again.is_error  # ticket gone, and the token is burnt


async def test_confirm_token_bound_to_target(local_client: Client) -> None:
    tok = (await local_client.call_tool("delete_ticket", {"ticket_id": 1})).structured_content[
        "confirm_token"
    ]
    r = await local_client.call_tool(
        "delete_ticket", {"ticket_id": 3, "confirm_token": tok}, raise_on_error=False
    )
    assert r.is_error and text_of(r).startswith("[invalid_confirm_token]")
    assert await _exists(local_client, 3)


async def test_confirm_token_bound_to_user(as_user) -> None:
    async with as_user("ada", role="admin") as c:
        tok = (await c.call_tool("delete_ticket", {"ticket_id": 1})).structured_content[
            "confirm_token"
        ]
    async with as_user("root", role="admin") as c:
        r = await c.call_tool(
            "delete_ticket", {"ticket_id": 1, "confirm_token": tok}, raise_on_error=False
        )
    assert r.is_error and "[invalid_confirm_token]" in text_of(r)


async def test_expired_confirm_token(tmp_path) -> None:
    from tests.conftest import make_app, make_settings

    app = await make_app(make_settings(tmp_path, auth_mode="local", confirm_token_ttl_s=0))
    async with Client(app.mcp) as c:
        tok = (await c.call_tool("delete_ticket", {"ticket_id": 1})).structured_content[
            "confirm_token"
        ]
        r = await c.call_tool(
            "delete_ticket", {"ticket_id": 1, "confirm_token": tok}, raise_on_error=False
        )
    assert r.is_error and "expired" in text_of(r)
    await app.engine.dispose()
