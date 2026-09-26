"""Helpers shared by the three demo servers."""

from __future__ import annotations

import os
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse


def advertise_list_changed(mcp: FastMCP) -> None:
    """Make the server declare ``listChanged: true`` for tools, resources and prompts.

    FastMCP builds its initialisation options with default ``NotificationOptions``, which
    advertise ``listChanged: false`` even though the server can send the notification.
    A well-behaved client only subscribes to what is advertised, so we fix the capability
    at the one place FastMCP builds it. This touches a private attribute; re-check it on
    every SDK upgrade (the test ``test_capabilities_advertise_list_changed`` will fail).
    """
    low = mcp._mcp_server
    original = low.create_initialization_options

    def create_initialization_options(
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
    ):
        return original(
            notification_options
            or NotificationOptions(prompts_changed=True, resources_changed=True, tools_changed=True),
            experimental_capabilities,
        )

    low.create_initialization_options = create_initialization_options  # type: ignore[method-assign]


def transport_security() -> TransportSecuritySettings:
    """DNS-rebinding protection that also accepts the compose service names."""
    hosts = os.environ.get("MCP_ALLOWED_HOSTS", "127.0.0.1:*,localhost:*").split(",")
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[h.strip() for h in hosts if h.strip()],
        allowed_origins=[f"http://{h.strip()}" for h in hosts if h.strip()],
    )


def add_health_route(mcp: FastMCP) -> None:
    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "server": mcp.name})


def serve_http(mcp: FastMCP, default_port: int) -> None:
    """Run a FastMCP server over streamable HTTP at ``/mcp``."""
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", default_port))
    uvicorn.run(mcp.streamable_http_app(), host=host, port=port, log_level="warning")
