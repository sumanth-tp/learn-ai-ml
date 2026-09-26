"""Command line entry point: `agentmon <command>`."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys

from agentmon.config import Settings


def _settings(args: argparse.Namespace) -> Settings:
    from agentmon.logging_setup import configure_logging

    overrides = {}
    if getattr(args, "prompt_version", None):
        overrides["prompt_version"] = args.prompt_version
    settings = Settings(**overrides)
    configure_logging(settings.log_level, settings.log_json)
    return settings


def cmd_demo(args: argparse.Namespace) -> int:
    from agentmon.demo import run_demo

    report = run_demo(_settings(args), per_day=args.per_day)
    fired = {a["rule"] for a in report["alerts"]}
    ok = (
        "required_tool_rate_low" in fired
        and report["regression"]["v1"]["gate"]["passed"]
        and not report["regression"]["v2"]["gate"]["passed"]
        and not report["redteam"]["gate_failures_on"]
    )
    print("demo story verified" if ok else "demo story NOT as expected, see out/report.json")
    return 0 if ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from agentmon.api import create_app

    uvicorn.run(create_app(_settings(args)), host=args.host, port=args.port)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from agentmon.runtime import build_runtime

    rt = build_runtime(_settings(args))
    resp = rt.service.handle(args.user, args.message)
    print(json.dumps(resp.model_dump(), indent=2))
    asyncio.run(rt.workers.run_once())
    print(json.dumps([e.model_dump() for e in rt.store.evals_for(resp.trace_id)], indent=2))
    return 0


def cmd_workers(args: argparse.Namespace) -> int:
    from agentmon.runtime import build_runtime

    settings = _settings(args)
    rt = build_runtime(settings)

    async def main() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await rt.workers.run_forever(stop)

    asyncio.run(main())
    return 0


def cmd_redteam(args: argparse.Namespace) -> int:
    from agentmon.safety.redteam import load_attacks, load_benign, redteam_gate, run_redteam

    s = _settings(args)
    report = run_redteam(
        s,
        load_attacks(s.redteam_path),
        load_benign(s.benign_path),
        guardrails=not args.no_guardrails,
    )
    print(
        json.dumps(
            {
                "asr": report.asr,
                "by_category": report.asr_by_category,
                "false_refusal_rate": report.false_refusal_rate,
                "succeeded": [o.id for o in report.outcomes if o.succeeded],
            },
            indent=2,
        )
    )
    fails = redteam_gate(report)
    print("GATE", "FAIL " + "; ".join(fails) if fails else "PASS")
    return 1 if fails and not args.no_guardrails else 0


def cmd_regress(args: argparse.Namespace) -> int:
    from agentmon.evals.judges import build_judges
    from agentmon.evals.regression import gate, load_cases, load_thresholds, run_suite
    from agentmon.llm.factory import build_judge_model

    s = _settings(args)
    cases = load_cases(s.golden_seed_path, s.golden_production_path)
    judges = build_judges(build_judge_model(s), s.cost_usd)
    report = asyncio.run(run_suite(cases, s, judges))
    result = gate(report, load_thresholds(s.thresholds_path))
    print(json.dumps(report.metrics, indent=2))
    for c in report.cases:
        if c.failures:
            print(f"  FAIL {c.id}: {'; '.join(c.failures)}")
    print("GATE", "PASS" if result.passed else "FAIL " + "; ".join(result.failures))
    return 0 if result.passed else 1


def cmd_alerts(args: argparse.Namespace) -> int:
    from agentmon.monitoring.alerts import load_rules, run_alerts
    from agentmon.store import Store

    s = _settings(args)
    for a in run_alerts(Store(s.db_path), load_rules(s.alerts_path)):
        print(f"{a['rule']}: {a['message']}")
        if "root_cause" in a["evidence"]:
            print("   ", a["evidence"]["root_cause"]["summary"])
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from agentmon.monitoring.dashboard import render_dashboard
    from agentmon.store import Store

    s = _settings(args)
    s.out_dir.mkdir(parents=True, exist_ok=True)
    path = s.out_dir / "dashboard.html"
    path.write_text(render_dashboard(Store(s.db_path)))
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentmon")
    p.add_argument("--prompt-version", choices=["v1", "v2"])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="simulate a week and run the whole pipeline")
    d.add_argument("--per-day", type=int, default=160)
    d.set_defaults(fn=cmd_demo)
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    c = sub.add_parser("chat", help="send one message through the agent")
    c.add_argument("message")
    c.add_argument("--user", default="CUST-1")
    c.set_defaults(fn=cmd_chat)
    sub.add_parser("workers", help="run eval workers until stopped").set_defaults(fn=cmd_workers)
    r = sub.add_parser("redteam", help="run the red-team suite")
    r.add_argument("--no-guardrails", action="store_true")
    r.set_defaults(fn=cmd_redteam)
    sub.add_parser("regress", help="offline regression gate").set_defaults(fn=cmd_regress)
    sub.add_parser("alerts", help="replay alert rules over the store").set_defaults(fn=cmd_alerts)
    sub.add_parser("dashboard", help="write out/dashboard.html").set_defaults(fn=cmd_dashboard)
    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
