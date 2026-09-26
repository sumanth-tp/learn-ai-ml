"""Command line: ``research-analyst {seed,run,resume,status,eval,serve}``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from research_analyst.config import Settings
from research_analyst.deps import index_path_for
from research_analyst.events import ProgressEvent
from research_analyst.logging_setup import configure_logging
from research_analyst.models import FinalReport


def _print_event(ev: ProgressEvent) -> None:
    d = ev.data
    msg = {
        "plan_ready": lambda: f"plan: {len(d.get('sub_questions', []))} sub-questions",
        "worker_started": lambda: f"  researching {d.get('sub_question_id')}: {d.get('question')}",
        "crag_verdict": lambda: (
            f"  {d.get('sub_question_id')} CRAG verdict={d.get('verdict')} "
            f"best={d.get('best_score')}"
        ),
        "web_fallback": lambda: f"  {d.get('sub_question_id')} web search: {d.get('query')}",
        "worker_finished": lambda: (
            f"  done {d.get('sub_question_id')}: {d.get('evidence')} evidence strips"
        ),
        "worker_failed": lambda: f"  FAILED {d.get('sub_question_id')}: {d.get('reason')}",
        "critique": lambda: f"critic score {d.get('overall')} revise={d.get('revision_requests')}",
        "verification": lambda: f"verified: {d}",
    }.get(ev.type, lambda: f"{ev.type} {json.dumps(d, default=str)[:160]}")
    print(msg(), file=sys.stderr, flush=True)


async def _run(
    settings: Settings, question: str, thread_id: str | None, resume: bool
) -> FinalReport | None:
    from research_analyst.service import ResearchService

    async with ResearchService(settings) as svc:
        if resume:
            assert thread_id
            async for ev in svc.resume(thread_id):
                _print_event(ev)
            st = await svc.status(thread_id)
            report = st.report
        else:
            import uuid

            thread_id = thread_id or uuid.uuid4().hex
            print(f"thread id: {thread_id}", file=sys.stderr)
            async for ev in svc.stream(question, thread_id):
                _print_event(ev)
            report = (await svc.status(thread_id)).report
    return report


def _write(report: FinalReport | None, out: Path | None) -> int:
    if report is None:
        print("no report produced", file=sys.stderr)
        return 1
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.markdown)
        out.with_suffix(".json").write_text(report.model_dump_json(indent=2))
        print(f"wrote {out} and {out.with_suffix('.json')}", file=sys.stderr)
    else:
        print(report.markdown)
    return 0


async def _status(settings: Settings, thread_id: str) -> int:
    from research_analyst.service import ResearchService

    async with ResearchService(settings) as svc:
        st = await svc.status(thread_id)
    print(json.dumps({"thread_id": st.thread_id, "status": st.status, "next": st.next_nodes}))
    return 0 if st.status != "not_found" else 1


async def _retention(settings: Settings, thread_id: str | None, days: float | None) -> int:
    from research_analyst.service import ResearchService

    async with ResearchService(settings) as svc:
        if thread_id:
            ok = await svc.delete(thread_id)
            print(f"deleted {thread_id}" if ok else f"{thread_id} not found")
            return 0 if ok else 1
        assert days is not None
        deleted = await svc.purge(days)
    print(f"purged {len(deleted)} run(s) older than {days} days")
    return 0


async def _eval(settings: Settings, out: Path) -> int:
    from research_analyst.evals.run import format_summary, run_eval, write_results

    summary = await run_eval(settings)
    print(format_summary(summary))
    write_results(summary, out)
    return 0 if summary.passed else 1


def _seed(settings: Settings, force: bool) -> int:
    from research_analyst.index import InternalIndex
    from research_analyst.providers.embeddings import build_embeddings

    path = Path(index_path_for(settings))
    if path.exists() and not force:
        print(f"index exists at {path} (use --force to rebuild)")
        return 0
    index = InternalIndex.build(settings.corpus_dir / "internal", build_embeddings(settings))
    index.save(path)
    print(f"built index at {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-analyst")
    parser.add_argument(
        "--live",
        action="store_true",
        help="use real providers (needs OPENAI_API_KEY; TAVILY_API_KEY optional)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_seed = sub.add_parser("seed", help="build the internal document index")
    p_seed.add_argument("--force", action="store_true")
    p_run = sub.add_parser("run", help="research a question and print the cited report")
    p_run.add_argument("question")
    p_run.add_argument("--thread-id")
    p_run.add_argument("--out", type=Path)
    p_res = sub.add_parser("resume", help="resume an interrupted run from its checkpoint")
    p_res.add_argument("thread_id")
    p_res.add_argument("--out", type=Path)
    p_st = sub.add_parser("status", help="show the state of a run")
    p_st.add_argument("thread_id")
    p_del = sub.add_parser("delete", help="erase all checkpoints of one run")
    p_del.add_argument("thread_id")
    p_pur = sub.add_parser("purge", help="delete runs older than the retention window")
    p_pur.add_argument("--days", type=float, default=30.0)
    p_ev = sub.add_parser("eval", help="run the offline eval set and the regression gate")
    p_ev.add_argument("--out", type=Path, default=Path("data/eval-results.json"))
    p_srv = sub.add_parser("serve", help="start the FastAPI server")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    overrides = {"mode": "live"} if args.live else {}
    settings = Settings(**overrides)
    configure_logging(settings.log_level, settings.log_json)

    match args.cmd:
        case "seed":
            return _seed(settings, args.force)
        case "run":
            return _write(
                asyncio.run(_run(settings, args.question, args.thread_id, False)), args.out
            )
        case "resume":
            return _write(asyncio.run(_run(settings, "", args.thread_id, True)), args.out)
        case "status":
            return asyncio.run(_status(settings, args.thread_id))
        case "delete":
            return asyncio.run(_retention(settings, args.thread_id, None))
        case "purge":
            return asyncio.run(_retention(settings, None, args.days))
        case "eval":
            return asyncio.run(_eval(settings, args.out))
        case "serve":
            import uvicorn

            uvicorn.run(
                "research_analyst.api:create_app",
                factory=True,
                host=args.host,
                port=args.port,
                log_config=None,
            )
            return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
