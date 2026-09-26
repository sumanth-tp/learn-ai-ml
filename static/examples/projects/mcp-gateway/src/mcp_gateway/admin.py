"""``mcp-gateway-admin``: operator CLI.

Reads the same config files and state the gateway uses, so it works against a
running gateway (the SQLite state is shared, the audit log is append-only).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from mcp_gateway.audit import iter_records, verify_chain
from mcp_gateway.config import Settings
from mcp_gateway.identity import mint_dev_token
from mcp_gateway.policy import PolicyEngine, Principal
from mcp_gateway.secret_broker import build_broker
from mcp_gateway.state import PinStore, StateDB
from mcp_gateway.upstreams import build_client_factory, load_upstreams


def _table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    cells = [[str(h) for h in headers], *[[str(c) for c in r] for r in rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


async def _probe(settings: Settings) -> dict[str, str]:
    broker = build_broker(settings.secrets_backend, settings.secrets_dir)
    out: dict[str, str] = {}
    for spec in load_upstreams(settings.upstreams_file):
        try:
            factory = build_client_factory(spec, broker, default_timeout=5.0)
            async with asyncio.timeout(10), factory() as client:
                tools = await client.list_tools()
            out[spec.name] = f"ok ({len(tools)} tools)"
        except Exception as exc:
            out[spec.name] = f"error: {type(exc).__name__}: {exc}"[:120]
    return out


def cmd_upstreams(s: Settings, a: argparse.Namespace) -> int:
    specs = load_upstreams(s.upstreams_file)
    probe = asyncio.run(_probe(s)) if a.probe else {}
    rows = [
        (u.name, u.transport, u.url or " ".join([u.command or "", *u.args]) or u.target,
         f"{u.credential.secret} ({u.credential.inject_as})" if u.credential else "-",
         ",".join(u.cacheable_tools) or "-", probe.get(u.name, "-"))
        for u in specs
    ]
    print(_table(rows, ["name", "transport", "endpoint", "credential ref", "cacheable", "probe"]))
    return 0


def cmd_policies(s: Settings, a: argparse.Namespace) -> int:
    engine = PolicyEngine.from_yaml(Path(a.file or s.policy_file).read_text(encoding="utf-8"))
    rows = [
        (r.id, r.effect, ",".join(r.subjects.groups + r.subjects.users),
         ",".join(r.tools + r.resources),
         "; ".join(f"{k}:{c.model_dump(exclude_none=True, exclude_defaults=True)}"
                   for k, c in r.constraints.items()) or "-")
        for r in engine.doc.rules
    ]
    print(_table(rows, ["rule", "effect", "subjects", "targets", "constraints"]))
    return 0


def cmd_check(s: Settings, a: argparse.Namespace) -> int:
    engine = PolicyEngine.from_yaml(Path(s.policy_file).read_text(encoding="utf-8"))
    p = Principal(a.user, frozenset(a.groups.split(",")) if a.groups else frozenset(), a.email)
    d = engine.check_tool(p, a.tool, json.loads(a.args))
    print(json.dumps({"allowed": d.allowed, "rule": d.rule_id, "reason": d.reason}))
    return 0 if d.allowed else 1


def cmd_validate(s: Settings, a: argparse.Namespace) -> int:
    path = Path(a.file or s.policy_file)
    try:
        engine = PolicyEngine.from_yaml(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"INVALID {path}: {exc}")
        return 1
    print(f"OK {path}: {len(engine.doc.rules)} rules")
    return 0


def cmd_denials(s: Settings, a: argparse.Namespace) -> int:
    recs = [r for r in iter_records(s.audit_path) if r["decision"] in ("deny", "alert")]
    if a.user:
        recs = [r for r in recs if r["user"] == a.user]
    rows = [
        (time.strftime("%H:%M:%S", time.localtime(r["ts"])), r["user"], r["target"],
         r["decision"], (r.get("rule_id") or "-"), r["reason"][:70])
        for r in recs[-a.limit:]
    ]
    print(_table(rows, ["time", "user", "target", "decision", "rule", "reason"]))
    return 0


def cmd_audit_verify(s: Settings, _: argparse.Namespace) -> int:
    ok, _n, msg = verify_chain(s.audit_path)
    print(("OK " if ok else "TAMPERED ") + msg)
    return 0 if ok else 2


def cmd_pins(s: Settings, a: argparse.Namespace) -> int:
    pins = PinStore(StateDB(s.state_db))
    if a.approve:
        ok = pins.approve(a.approve)
        print(f"approved {a.approve}" if ok else f"no pin for {a.approve}")
        return 0 if ok else 1
    rows = [(p.tool, p.status, p.sha256[:12], (p.seen_sha256 or "")[:12], p.findings[:60])
            for p in pins.all()]
    print(_table(rows, ["tool", "status", "pinned", "now serving", "findings"]))
    return 0


def cmd_token(s: Settings, a: argparse.Namespace) -> int:
    print(mint_dev_token(s, a.sub, [g for g in a.groups.split(",") if g], email=a.email,
                         ttl_seconds=a.ttl))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(prog="mcp-gateway-admin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("upstreams", help="list upstreams")
    p.add_argument("--probe", action="store_true", help="connect and count tools")
    p = sub.add_parser("policies", help="list policy rules")
    p.add_argument("--file")
    p = sub.add_parser("validate-policy", help="validate a policy file before deploying it")
    p.add_argument("--file")
    p = sub.add_parser("check", help="dry-run a policy decision")
    p.add_argument("--user", required=True)
    p.add_argument("--groups", default="")
    p.add_argument("--email")
    p.add_argument("--tool", required=True)
    p.add_argument("--args", default="{}")
    p = sub.add_parser("denials", help="recent denials and security alerts")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--user")
    sub.add_parser("verify-audit", help="verify the audit hash chain")
    p = sub.add_parser("pins", help="list tool pins, or approve a changed tool")
    p.add_argument("--approve", metavar="TOOL")
    p = sub.add_parser("mint-token", help="dev-only HS256 token")
    p.add_argument("--sub", required=True)
    p.add_argument("--groups", default="employees")
    p.add_argument("--email")
    p.add_argument("--ttl", type=int, default=3600)
    a = parser.parse_args(argv)
    settings = Settings()
    handlers = {
        "upstreams": cmd_upstreams, "policies": cmd_policies, "validate-policy": cmd_validate,
        "check": cmd_check, "denials": cmd_denials, "verify-audit": cmd_audit_verify,
        "pins": cmd_pins, "mint-token": cmd_token,
    }
    return handlers[a.cmd](settings, a)


if __name__ == "__main__":
    sys.exit(main())
