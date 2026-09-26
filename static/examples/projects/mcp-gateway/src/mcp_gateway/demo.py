"""Scripted end-to-end walkthrough: starts the real stack (two HTTP upstreams,
the gateway, and the stdio docs upstream it spawns) with throwaway state, then
drives it as four different users and finally stages a rug pull.

``mcp-gateway demo``        offline, deterministic
``mcp-gateway demo --llm``  also asks a real LLM to judge new tool descriptions
                            (needs OPENAI_API_KEY or another provider's key)
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from mcp_gateway import stack
from mcp_gateway.audit import iter_records, verify_chain
from mcp_gateway.config import Settings
from mcp_gateway.identity import mint_dev_token

DEMO_ENV = {
    "GATEWAY_JWT_ALGORITHM": "HS256",
    "GATEWAY_JWT_SECRET": "demo-only-secret-with-at-least-32-characters",
    "GATEWAY_SECRET_DOCS_TOKEN": "demo-docs-token-0123456789",
    "GATEWAY_SECRET_PAYMENTS_TOKEN": "demo-payments-token-0123456789",
    "GATEWAY_SECRET_TICKETS_KEY": "demo-tickets-key-0123456789",
    "GATEWAY_LOG_JSON": "false",
    "GATEWAY_LOG_LEVEL": "WARNING",
    "GATEWAY_DEFINITION_REFRESH_SECONDS": "0",
    "FASTMCP_LOG_LEVEL": "WARNING",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _say(title: str) -> None:
    print(f"\n=== {title}")


def _show(r: Any) -> str:
    text = r.content[0].text if r.content else json.dumps(r.structured_content)
    status = "ERROR" if r.is_error else "ok"
    return f"[{status}] {text[:150]}"


async def run_demo(use_llm: bool = False) -> int:
    state = Path(tempfile.mkdtemp(prefix="mcp-gateway-demo-"))
    gw_port = _free_port()
    stack.PAYMENTS_PORT, stack.TICKETS_PORT = _free_port(), _free_port()
    env_extra = DEMO_ENV | {
        "GATEWAY_STATE_DIR": str(state),
        "GATEWAY_PORT": str(gw_port),
        "PAYMENTS_URL": f"http://127.0.0.1:{stack.PAYMENTS_PORT}/mcp",
        "TICKETS_URL": f"http://127.0.0.1:{stack.TICKETS_PORT}/mcp",
    }
    if use_llm:
        env_extra |= {
            "GATEWAY_LLM_SCANNER": "true",
            "GATEWAY_LLM_PROVIDER": os.environ.get("GATEWAY_LLM_PROVIDER", "openai"),
        }
    os.environ.update(env_extra)
    env = stack.child_env()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    url = f"http://127.0.0.1:{gw_port}/mcp"

    def tok(sub: str, groups: list[str], email: str | None = None) -> str:
        return mint_dev_token(settings, sub, groups, email=email)

    procs = [stack.spawn_payments(env), stack.spawn_tickets(env)]
    try:
        stack.wait_for_port(stack.PAYMENTS_PORT)
        stack.wait_for_port(stack.TICKETS_PORT)
        procs.append(stack.spawn_gateway(env))
        print("readiness:", stack.wait_ready(url.removesuffix("/mcp") + "/readyz"))

        async def as_user(token: str) -> Client[Any]:
            return Client(StreamableHttpTransport(url, auth=token))

        _say("1. Ana (employees) sees only what her groups allow")
        async with await as_user(tok("ana", ["employees"])) as c:
            print(sorted(t.name for t in await c.list_tools()))
            print(
                "public doc:  ",
                _show(await c.call_tool("docs_read_doc", {"path": "/public/handbook.md"})),
            )
            print(
                "again (cache):",
                _show(await c.call_tool("docs_read_doc", {"path": "/public/handbook.md"})),
            )
            print(
                "finance doc: ",
                _show(
                    await c.call_tool(
                        "docs_read_doc", {"path": "/finance/q3-forecast.md"}, raise_on_error=False
                    )
                ),
            )
            print(
                "traversal:   ",
                _show(
                    await c.call_tool(
                        "docs_read_doc",
                        {"path": "/public/../hr/salaries.csv"},
                        raise_on_error=False,
                    )
                ),
            )

        _say("2. Sam (support): tickets, and refunds up to 100")
        async with await as_user(tok("sam", ["support"])) as c:
            print(
                "ticket (PII): ",
                _show(await c.call_tool("tickets_get_ticket", {"ticket_id": "T-1"})),
            )
            refund = {
                "order_id": "O-77",
                "amount": 40,
                "currency": "EUR",
                "idempotency_key": "demo-refund-0001",
            }
            print("refund 40:    ", _show(await c.call_tool("payments_refund", refund)))
            print("same key:     ", _show(await c.call_tool("payments_refund", refund)))
            print(
                "refund 400:   ",
                _show(
                    await c.call_tool(
                        "payments_refund",
                        refund | {"amount": 400, "idempotency_key": "demo-refund-0002"},
                        raise_on_error=False,
                    )
                ),
            )

        _say("3. Carl (finance + contractors): deny overrides allow")
        async with await as_user(tok("carl", ["finance", "contractors"])) as c:
            print(
                _show(
                    await c.call_tool(
                        "payments_get_balance", {"account": "ACC-1001"}, raise_on_error=False
                    )
                )
            )

        _say("4. Mallory forges a token with her own key")
        import jwt

        forged = jwt.encode(
            {
                "sub": "mallory",
                "groups": ["finance"],
                "aud": "mcp-gateway",
                "iss": settings.jwt_issuer,
                "exp": 4102444800,
            },
            "mallory-own-secret-that-is-32-chars-long",
            algorithm="HS256",
        )
        try:
            async with await as_user(forged) as c:
                await c.list_tools()
            print("UNEXPECTED: forged token accepted")
        except Exception as exc:
            print(f"rejected at the transport: {type(exc).__name__}")

        _say("5. Rug pull: tickets upstream swaps a description after approval")
        procs[1].terminate()
        procs[1].wait(10)
        procs[1] = stack.spawn_tickets(stack.child_env({"TICKETS_POISON": "description"}))
        stack.wait_for_port(stack.TICKETS_PORT)
        async with await as_user(tok("sam", ["support"])) as c:
            print(
                _show(
                    await c.call_tool(
                        "tickets_search_tickets", {"query": "refund"}, raise_on_error=False
                    )
                )
            )

        _say("6. What the operator sees")
        denials = [
            r for r in iter_records(settings.audit_path) if r["decision"] in ("deny", "alert")
        ]
        for r in denials:
            print(f"  {r['decision']:5} {r['user']:8} {r['target']:24} {r['reason'][:70]}")
        raw = settings.audit_path.read_text()
        print("audit contains customer email?", "ana.silva@example.com" in raw)
        print("audit chain:", verify_chain(settings.audit_path)[2])
        print(f"state kept in {state} (audit.jsonl, state.db)")
        print("try: GATEWAY_STATE_DIR=" + str(state) + " uv run mcp-gateway-admin denials")
        return 0
    finally:
        stack.stop(procs)
