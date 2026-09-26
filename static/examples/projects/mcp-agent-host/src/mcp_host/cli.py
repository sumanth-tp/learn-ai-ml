"""Command-line front end: chat, serve, servers, eval, demo."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx

from mcp_host.agent import ApprovalDecision
from mcp_host.llm import build_chat_model
from mcp_host.logs import configure_logging
from mcp_host.runtime import open_runtime
from mcp_host.service import ChatService
from mcp_host.settings import Settings, get_settings, load_servers


async def render(
    events: AsyncIterator[dict[str, Any]], out: Any = sys.stdout
) -> dict[str, Any] | None:
    """Print a turn's events; return the approval request if the turn paused."""
    pending = None
    streamed = False
    async for event in events:
        kind = event["type"]
        if kind == "token":
            out.write(event["text"])
            out.flush()
            streamed = True
        elif kind == "message" and not streamed:
            out.write(str(event["text"]))
        elif kind == "tool_call":
            out.write(f"\n  -> {event['tool']} {json.dumps(event['args'])}\n")
        elif kind == "tool_result":
            flag = " FLAGGED" if event["flagged"] else ""
            out.write(f"  <- {event['tool']} [{event['status']}{flag}]\n")
        elif kind == "approval_required":
            pending = event
        elif kind == "error":
            out.write(f"\n  ! {event['message']}\n")
    out.write("\n")
    return pending


async def ask_approval(service: ChatService, thread: str, pending: dict[str, Any]) -> None:
    while pending:
        approve = []
        for c in pending["calls"]:
            answer = await asyncio.to_thread(
                input, f"  approve {c['tool']} {json.dumps(c['args'])}? [y/N] "
            )
            if answer.strip().lower() in {"y", "yes"}:
                approve.append(c["id"])
        pending = await render(service.resume(thread, ApprovalDecision(approve=approve)))


async def chat(settings: Settings, thread: str) -> None:
    async with open_runtime(settings) as rt:
        service = rt.service
        print(f"thread {thread}. Commands: /servers  /prompts  /prompt <server> <name> k=v  /quit")
        print(rt.host.registry.availability_note())
        while True:
            try:
                line = (await asyncio.to_thread(input, "you> ")).strip()
            except EOFError:
                break
            if not line:
                continue
            if line in {"/quit", "/exit"}:
                break
            if line == "/servers":
                print(json.dumps(rt.host.status(), indent=2))
                continue
            if line == "/prompts":
                print(json.dumps(rt.host.list_prompts(), indent=2))
                continue
            if line.startswith("/prompt "):
                parts = line.split()
                args = dict(p.split("=", 1) for p in parts[3:] if "=" in p)
                events = service.send_prompt(thread, parts[1], parts[2], args)
            else:
                events = service.send(thread, line)
            pending = await render(events)
            if pending:
                await ask_approval(service, thread, pending)


@contextlib.contextmanager
def http_servers(settings: Settings) -> Iterator[None]:
    """Start the two HTTP demo servers unless something already answers on their ports."""
    procs = []
    for module, port in (
        ("demo_servers.calendar_server", 8101),
        ("demo_servers.docs_server", 8102),
    ):
        try:
            httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=0.5)
            continue
        except httpx.HTTPError:
            pass
        env = {**os.environ, "MCP_PORT": str(port)}
        procs.append(subprocess.Popen([sys.executable, "-m", module], env=env))
    try:
        deadline = time.monotonic() + 20
        for port in (8101, 8102):
            while True:
                try:
                    httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=0.5).raise_for_status()
                    break
                except httpx.HTTPError:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"demo server on port {port} did not start") from None
                    time.sleep(0.2)
        yield
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)


DEMO_SCRIPT = [
    (
        "A policy question (docs server, BM25 search)",
        "What is the meal allowance in the expense policy?",
    ),
    (
        "A name collision: notes also has 'search'",
        "What did I write about the rollout plan in my notes?",
    ),
    (
        "Sampling: the calendar server asks our LLM",
        "Give me a summary of 2026-10-05, summarise my day",
    ),
    ("A poisoned document (prompt injection)", "Tell me about the vendor onboarding portal notes"),
    ("A destructive tool behind approval", "Delete note reading-list"),
]


async def demo(settings: Settings) -> None:
    async with open_runtime(settings) as rt:
        print(json.dumps(rt.host.status()["servers"], indent=2))
        thread = f"demo-{uuid.uuid4().hex[:6]}"
        for title, text in DEMO_SCRIPT:
            print(f"\n=== {title}\nyou> {text}")
            pending = await render(rt.service.send(thread, text))
            if pending:
                print(
                    f"  approval requested for {[c['tool'] for c in pending['calls']]}; approving"
                )
                ids = [c["id"] for c in pending["calls"]]
                await render(rt.service.resume(thread, ApprovalDecision(approve=ids)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mcp-host")
    sub = parser.add_subparsers(dest="command", required=True)
    p_chat = sub.add_parser("chat", help="interactive chat in the terminal")
    p_chat.add_argument("--thread", default=None)
    sub.add_parser("serve", help="run the FastAPI server")
    sub.add_parser("servers", help="connect, print server status, exit")
    p_eval = sub.add_parser("eval", help="run the tool-selection eval")
    p_eval.add_argument("--dataset", type=Path, default=Path("evals/tool_selection.jsonl"))
    p_eval.add_argument("--report", type=Path, default=Path("evals/report.json"))
    p_eval.add_argument("--min-tool-accuracy", type=float, default=0.9)
    p_eval.add_argument("--min-args-accuracy", type=float, default=0.8)
    p_demo = sub.add_parser("demo", help="start the HTTP servers and run a scripted session")
    p_demo.add_argument("--no-spawn", action="store_true", help="servers are already running")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(
        settings.log_level if args.command == "serve" else "WARNING", settings.log_json
    )

    if args.command == "chat":
        asyncio.run(chat(settings, args.thread or f"cli-{uuid.uuid4().hex[:6]}"))
    elif args.command == "serve":
        import uvicorn

        from mcp_host.api import create_app

        uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port)
    elif args.command == "servers":

        async def show() -> None:
            async with open_runtime(settings) as rt:
                print(json.dumps(rt.host.status(), indent=2))

        asyncio.run(show())
    elif args.command == "eval":
        from mcp_host.evals import run_eval, write_report

        load_servers(settings.servers_file)  # fail fast on a broken config
        report = asyncio.run(
            run_eval(
                build_chat_model(settings),
                settings,
                args.dataset,
                args.min_tool_accuracy,
                args.min_args_accuracy,
            )
        )
        write_report(report, args.report)
        print(
            f"tool accuracy {report.tool_accuracy:.1%}  args accuracy {report.args_accuracy:.1%}  "
            f"wrong namespace {report.wrong_namespace}  n={report.n}"
        )
        for miss, count in report.confusions.items():
            print(f"  miss x{count}: {miss}")
        if not report.passed:
            print(f"FAILED gate {report.thresholds}")
            raise SystemExit(1)
        print("PASSED gate")
    elif args.command == "demo":
        ctx = contextlib.nullcontext() if args.no_spawn else http_servers(settings)
        with ctx:
            asyncio.run(demo(settings))


if __name__ == "__main__":
    main()
