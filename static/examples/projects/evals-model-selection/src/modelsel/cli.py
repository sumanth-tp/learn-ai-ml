"""Command line: ``modelsel <command>``. Every command reads the same Settings."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from modelsel.config import Settings
from modelsel.dataset import PrivateSplitLocked, load_human_labels, write_benchmark
from modelsel.harness.cache import ResponseCache
from modelsel.harness.client import LLMClient
from modelsel.judge.calibration import calibrate
from modelsel.llm.registry import load_catalogue
from modelsel.logging_setup import configure_logging
from modelsel.pipeline import ensure_data, run_selection
from modelsel.report import render_html, render_markdown
from modelsel.stats import minimum_detectable_effect, n_paired_means, n_paired_proportions
from modelsel.store import RunStore


def _cmd_build_data(settings: Settings, _: argparse.Namespace) -> int:
    counts = write_benchmark(settings.data_dir, seed=settings.seed)
    print(json.dumps(counts))
    return 0


def _cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    if args.include_private and not settings.allow_private:
        print("refusing: --include-private needs MODELSEL_ALLOW_PRIVATE=true", file=sys.stderr)
        return 2
    store = RunStore(settings.db_path)
    try:
        models = args.models.split(",") if args.models else None
        summary = asyncio.run(
            run_selection(settings, store, include_private=args.include_private, run_id=args.run_id, candidates=models)
        )
    except PrivateSplitLocked as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()
    rec = summary["recommendation"]
    print(f"recommended: {rec['model_id']}  ({rec['reason']})")
    for c in summary["caveats"]:
        print(f"  caveat: {c}")
    print(f"report: {settings.reports_dir / (summary['run_id'] + '.html')}")
    return 0


def _cmd_calibrate(settings: Settings, _: argparse.Namespace) -> int:
    ensure_data(settings)
    catalogue = load_catalogue(settings.models_file)
    profile = catalogue.profiles[settings.profile]
    cache = ResponseCache(settings.cache_path)
    client = LLMClient(settings, catalogue, cache)
    rows = load_human_labels(settings.data_dir)[: settings.judge_samples_for_calibration]
    report = asyncio.run(
        calibrate(
            client, catalogue, profile.judge, profile.meta_judge, rows, spot_checks=settings.meta_judge_spot_checks
        )
    )
    cache.close()
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0 if report.trusted else 1


def _cmd_sample_size(_: Settings, args: argparse.Namespace) -> int:
    out: dict[str, float | int] = {"n_for_mean_diff": n_paired_means(args.delta, args.sd)}
    if args.discordant:
        out["n_for_accuracy_mcnemar"] = n_paired_proportions(args.delta, args.discordant)
    if args.n:
        out["mde_at_n"] = minimum_detectable_effect(args.n, args.sd)
    print(json.dumps(out))
    return 0


def _cmd_report(settings: Settings, args: argparse.Namespace) -> int:
    store = RunStore(settings.db_path)
    run = store.get_run(args.run_id)
    store.close()
    if run is None or run["summary"] is None:
        print(f"no finished run {args.run_id}", file=sys.stderr)
        return 1
    print(render_markdown(run["summary"]) if args.format == "md" else render_html(run["summary"]))
    return 0


def _cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("modelsel.api:app", host=args.host, port=args.port, log_config=None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="modelsel", description="Model-selection lab for support-ticket triage")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build-data", help="regenerate the benchmark and human-label files")
    r = sub.add_parser("run", help="run the full selection pipeline and write the report")
    r.add_argument("--include-private", action="store_true", help="also score the held-out private split")
    r.add_argument("--models", help="comma-separated override of the profile's candidates")
    r.add_argument("--run-id")
    sub.add_parser("calibrate", help="calibrate the judge against human labels only")
    s = sub.add_parser("sample-size", help="items needed to detect a difference")
    s.add_argument("--delta", type=float, required=True)
    s.add_argument("--sd", type=float, required=True, help="SD of per-item paired differences")
    s.add_argument("--discordant", type=float, help="share of items where exactly one model is right")
    s.add_argument("--n", type=int, help="also report the minimum detectable effect at this n")
    rep = sub.add_parser("report", help="re-render a stored run")
    rep.add_argument("run_id")
    rep.add_argument("--format", choices=["md", "html"], default="md")
    sv = sub.add_parser("serve", help="start the HTTP API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    return p


COMMANDS = {
    "build-data": _cmd_build_data,
    "run": _cmd_run,
    "calibrate": _cmd_calibrate,
    "sample-size": _cmd_sample_size,
    "report": _cmd_report,
    "serve": _cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level, settings.log_json)
    return COMMANDS[args.command](settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
