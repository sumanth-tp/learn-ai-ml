"""``mcp-gateway`` command: serve the gateway, run the local stack, or the demo."""

from __future__ import annotations

import argparse
import sys

import uvicorn
from dotenv import load_dotenv


def serve() -> None:
    from mcp_gateway.config import get_settings
    from mcp_gateway.gateway import build_gateway
    from mcp_gateway.observability import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    gateway, _ = build_gateway(settings)
    app = gateway.http_app(
        path=settings.mcp_path,
        allowed_hosts=settings.public_hostnames or None,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning",
                proxy_headers=True, timeout_graceful_shutdown=10)


def main(argv: list[str] | None = None) -> None:
    # Load .env into the process environment so the env secret broker and the
    # upstream children see the same values as pydantic-settings does.
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(prog="mcp-gateway")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the gateway (HTTP)")
    sub.add_parser("stack", help="run demo upstreams + gateway locally, one command")
    demo = sub.add_parser("demo", help="scripted end-to-end walkthrough")
    demo.add_argument("--llm", action="store_true", help="use a real LLM for description scans")
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        serve()
    elif args.cmd == "stack":
        from mcp_gateway.stack import run_stack

        sys.exit(run_stack())
    else:
        import asyncio

        from mcp_gateway.demo import run_demo

        sys.exit(asyncio.run(run_demo(use_llm=args.llm)))


if __name__ == "__main__":
    main()
