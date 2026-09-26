"""Command line: ``helpdesk-mcp <command>``.

serve            run the server (HTTP by default, ``--transport stdio`` for local clients)
migrate          apply Alembic migrations to HELPDESK_DATABASE_URL
seed             insert demo tenants, KB articles and tickets (idempotent)
mint-token       print a dev JWT for a user (HS256 only)
demo             full end-to-end scenario over HTTP on a random port
eval             run the triage evaluation and apply the regression gate
schema-snapshot  write the tool-schema snapshot used by the compatibility test
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn

from helpdesk_mcp.config import Settings
from helpdesk_mcp.logging_setup import configure_logging

ROOT = Path.cwd()


def _serve(settings: Settings, transport: str | None) -> None:
    from helpdesk_mcp.server import build_server

    transport = transport or settings.transport
    if transport == "stdio":
        # stdio: one local user, no network, identity from HELPDESK_LOCAL_*.
        settings = settings.model_copy(update={"auth_mode": "local", "transport": "stdio"})
        app = build_server(settings)
        app.mcp.run(transport="stdio", show_banner=False)
        return
    app = build_server(settings)
    asgi = app.mcp.http_app(path=settings.mcp_path)
    uvicorn.run(
        asgi,
        host=settings.host,
        port=settings.port,
        log_config=None,  # keep our JSON logging
        proxy_headers=True,  # trust X-Forwarded-* from the TLS-terminating proxy
        forwarded_allow_ips="*",
        timeout_graceful_shutdown=20,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="helpdesk-mcp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--transport", choices=["http", "stdio"])
    sub.add_parser("migrate")
    sub.add_parser("seed")
    p_tok = sub.add_parser("mint-token")
    p_tok.add_argument("--user", required=True)
    p_tok.add_argument("--tenant", required=True)
    p_tok.add_argument("--role", default="requester", choices=["requester", "agent", "admin"])
    p_tok.add_argument("--ttl", type=int, default=3600)
    sub.add_parser("demo")
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--cases", default="evals/triage_cases.jsonl")
    p_eval.add_argument("--report", default="evals/report.json")
    p_snap = sub.add_parser("schema-snapshot")
    p_snap.add_argument("--out", default="tests/snapshots/tool_schemas.json")
    args = parser.parse_args(argv)

    settings = Settings()
    configure_logging(settings.log_level, settings.log_json)

    if args.cmd == "serve":
        _serve(settings, args.transport)
    elif args.cmd == "migrate":
        from helpdesk_mcp.db.migrate import upgrade

        upgrade(settings.database_url)
        print("migrations applied", file=sys.stderr)
    elif args.cmd == "seed":
        from helpdesk_mcp.db.session import make_engine
        from helpdesk_mcp.seed import seed

        async def _seed() -> dict:
            engine = make_engine(settings)
            try:
                return await seed(engine)
            finally:
                await engine.dispose()

        print(asyncio.run(_seed()))
    elif args.cmd == "mint-token":
        from helpdesk_mcp.tokens import mint_token

        print(
            mint_token(
                settings, user=args.user, tenant=args.tenant, roles=[args.role], ttl_s=args.ttl
            )
        )
    elif args.cmd == "demo":
        from helpdesk_mcp.demo import run_demo

        result = asyncio.run(run_demo(settings))
        return 0 if all(v for k, v in result.items() if isinstance(v, bool)) else 1
    elif args.cmd == "eval":
        from helpdesk_mcp.evals import load_cases, run_eval, write_report
        from helpdesk_mcp.triage import TriageService, build_chat_model

        service = TriageService(
            build_chat_model(settings),
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
        report = asyncio.run(run_eval(service, load_cases(ROOT / args.cases)))
        write_report(report, ROOT / args.report)
        print(
            f"provider={settings.llm_provider} model={settings.llm_model} "
            f"category={report.category_accuracy:.2f} priority={report.priority_accuracy:.2f} "
            f"p1_recall={report.p1_recall:.2f} fallback={report.fallback_rate:.2f}"
        )
        problems = report.gate()
        for p in problems:
            print(f"GATE FAIL: {p}", file=sys.stderr)
        return 1 if problems else 0
    elif args.cmd == "schema-snapshot":
        from helpdesk_mcp.schema_compat import save, snapshot
        from helpdesk_mcp.server import build_server

        local = settings.model_copy(
            update={
                "auth_mode": "local",
                "local_roles": ["admin"],
                "database_url": "sqlite+aiosqlite:///:memory:",
            }
        )
        app = build_server(local)
        save(ROOT / args.out, asyncio.run(snapshot(app.mcp)))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
