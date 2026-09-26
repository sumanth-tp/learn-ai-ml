"""Command line interface: `uv run ragate --help`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from ragate.config import load_pipeline_config
from ragate.dataset import checks, review, store
from ragate.dataset.store import DatasetError
from ragate.dataset.synth import SYNTH_PROMPT_VERSION, synthesise
from ragate.evaluation.build import JudgeBackend, run_eval
from ragate.evaluation.experiments import run_experiments
from ragate.evaluation.gate import ExitCode, compare, load_gate_config, load_noise
from ragate.evaluation.noise import measure_noise, write_noise
from ragate.evaluation.report import fmt, gate_report
from ragate.evaluation.store import RunStore, load_run_file, write_run_file
from ragate.judge.prompts import lock_violations, write_lock
from ragate.log import configure_logging
from ragate.models import QuestionType
from ragate.providers import ProviderConfigError, chat_model
from ragate.rag.chunking import chunk_corpus
from ragate.rag.corpus import corpus_hash, load_corpus
from ragate.rag.pipeline import build_pipeline
from ragate.settings import get_settings

app = typer.Typer(help="RAG evaluation system and CI release gate.", no_args_is_help=True)
dataset_app = typer.Typer(help="Golden dataset: synth, review, freeze, check.")
app.add_typer(dataset_app, name="dataset")

CONFIG = typer.Option(Path("config/pipeline.yaml"), help="Pipeline config YAML")
HEADLINE = ["recall_at_k", "mrr", "faithfulness", "answer_relevancy", "correctness",
            "citation_precision", "refusal_rate", "pii_leak_rate", "latency_p95_ms",
            "cost_per_query_usd"]


@app.callback()
def _setup() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)


def _runs() -> RunStore:
    return RunStore(get_settings().runs_db)


def _fail(message: str, code: int = ExitCode.ERROR) -> None:
    typer.secho(message, fg="red", err=True)
    raise typer.Exit(code)


@app.command()
def ingest(config: Path = CONFIG) -> None:
    """Build (or reuse) the index for a config."""
    pipeline = build_pipeline(load_pipeline_config(config), get_settings())
    index = pipeline.retriever.index
    typer.echo(f"index {index.key}: {len(index.chunks)} chunks")


@app.command()
def ask(question: str, config: Path = CONFIG) -> None:
    """Ask the assistant one question."""
    ans = build_pipeline(load_pipeline_config(config), get_settings()).ask(question)
    typer.echo(ans.answer)
    typer.echo(f"\nsources: {[c.chunk.chunk_id for c in ans.contexts]}  refused={ans.refused}"
               f"  tokens={ans.usage.total_tokens}  cost=${ans.cost_usd:.6f}")


@dataset_app.command("check")
def dataset_check(version: str | None = None) -> None:
    """Run quality, contamination and leakage checks on a frozen version."""
    s = get_settings()
    try:
        manifest, items = store.load(s.golden_dir, version or store.latest_version(s.golden_dir))
    except DatasetError as exc:
        _fail(str(exc))
    docs = load_corpus(s.corpus_dir)
    findings = checks.run_checks(items, docs)
    if manifest.corpus_sha != corpus_hash(docs):
        typer.secho("warning: corpus changed since this dataset was frozen", fg="yellow")
    for f in findings:
        typer.echo(f"{f.level:7} {f.check:20} {f.item_id:10} {f.message}")
    typer.echo(f"{manifest.version}: {len(items)} approved items, {manifest.counts_by_type}, "
               f"{sum(f.level == 'error' for f in findings)} errors")
    if checks.has_errors(findings):
        raise typer.Exit(ExitCode.ERROR)


@dataset_app.command("synth")
def dataset_synth(
    per_type: int = typer.Option(3, help="Proposals per question type"),
    out: Path = typer.Option(Path("data/golden/review/pending.csv")),
    seed: int = 13,
) -> None:
    """Generate synthetic candidates into a review CSV (status: pending)."""
    s = get_settings()
    cfg = load_pipeline_config(Path("config/pipeline.yaml"))
    chunks = chunk_corpus(load_corpus(s.corpus_dir), cfg.chunk_size, cfg.chunk_overlap)
    try:
        model = chat_model(s, role="synth")
    except ProviderConfigError as exc:
        _fail(str(exc))
    items = synthesise(model, chunks, {qt: per_type for qt in QuestionType}, seed=seed)
    review.export_csv(items, out)
    typer.echo(f"wrote {len(items)} pending items to {out}; review them, then run "
               "`ragate dataset freeze`")


@dataset_app.command("freeze")
def dataset_freeze(
    review_csv: Path = typer.Option(..., "--review"),
    version: str = typer.Option(..., help="New version, e.g. v2"),
    base: str | None = typer.Option(None, help="Parent version whose items are kept"),
) -> None:
    """Merge approved review rows (plus an optional parent version) into a new frozen version."""
    s = get_settings()
    items = []
    if base:
        _, items = store.load(s.golden_dir, base)
    try:
        new = review.approved(review.import_csv(review_csv))
    except ValueError as exc:
        _fail(str(exc))
    known = {i.item_id for i in items}
    items += [i for i in new if i.item_id not in known]
    docs = load_corpus(s.corpus_dir)
    findings = checks.run_checks(items, docs)
    if checks.has_errors(findings):
        for f in findings:
            typer.echo(f"{f.level:7} {f.check:20} {f.item_id:10} {f.message}")
        _fail("refusing to freeze a dataset with check errors")
    manifest = store.freeze(s.golden_dir, version, items, corpus_sha=corpus_hash(docs),
                            parent=base, generator={"synth_prompt": SYNTH_PROMPT_VERSION,
                                                    "model": s.chat_model})
    typer.echo(f"froze {version}: {manifest.counts_by_type} sha={manifest.sha256[:12]}")


@app.command("eval")
def eval_cmd(
    config: Path = CONFIG,
    dataset: str | None = None,
    out: Path | None = typer.Option(None, help="Also write the run JSON here"),
    judge_backend: str = typer.Option("native", help="native | deepeval"),
    force: bool = typer.Option(False, help="Recompute even if an identical run is stored"),
) -> None:
    """Evaluate a config on the golden set and store the run."""
    s = get_settings()
    backend: JudgeBackend = "deepeval" if judge_backend == "deepeval" else "native"
    try:
        run = run_eval(s, load_pipeline_config(config), runs=_runs(), dataset_version=dataset,
                       judge_backend=backend, force=force)
    except (DatasetError, ProviderConfigError) as exc:
        _fail(str(exc))
    if out:
        write_run_file(run, out)
    typer.echo(f"run {run.run_id} ({len(run.items)} items, {run.duration_s}s)")
    for m in HEADLINE:
        typer.echo(f"  {m:22} {fmt(run.aggregates.get(m), m)}")


def _gate(baseline: Path, candidate_run, report: Path | None, noise: Path | None) -> int:  # type: ignore[no-untyped-def]
    s = get_settings()
    base = load_run_file(baseline)
    result = compare(base, candidate_run, load_gate_config(s.config_dir / "gate.yaml"),
                     load_noise(noise, candidate_run.judge.model_id))
    text = gate_report(result, base, candidate_run)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text)
    _runs().record_decision(base.run_id, candidate_run.run_id, result.decision, text)
    typer.echo(text)
    return result.exit_code


@app.command()
def gate(
    baseline: Path = typer.Option(Path("baselines/baseline.json")),
    candidate: Path = typer.Option(..., help="Candidate run JSON"),
    report: Path | None = typer.Option(Path("reports/gate.md")),
    noise: Path | None = typer.Option(Path("baselines/noise.json")),
) -> None:
    """Compare two run files. Exit 0 promote, 1 block, 2 error."""
    for p in (baseline, candidate):
        if not p.exists():
            _fail(f"{p} does not exist")
    raise typer.Exit(_gate(baseline, load_run_file(candidate), report, noise))


@app.command()
def baseline(config: Path = CONFIG, out: Path = typer.Option(Path("baselines/baseline.json")),
             force: bool = False) -> None:
    """Evaluate a config and write it as the new baseline (run on main after a promote)."""
    cfg = load_pipeline_config(config).with_overrides("baseline", {})
    run = run_eval(get_settings(), cfg, runs=_runs(), force=force)
    write_run_file(run, out)
    typer.echo(f"baseline {run.run_id} written to {out}")


@app.command()
def experiments(path: Path = typer.Option(Path("config/experiments.yaml")),
                force: bool = False) -> None:
    """Run the experiment grid and write reports/experiments.md."""
    s = get_settings()
    _, best = run_experiments(s, path, _runs(), s.reports_dir, force=force)
    typer.echo((s.reports_dir / "experiments.md").read_text())
    typer.echo(f"recommended: {best}")


@app.command()
def noise(config: Path = CONFIG, repeats: int = 5,
          jitter: float = typer.Option(0.0, help="Simulated judge noise for the stub judge"),
          out: Path = typer.Option(Path("baselines/noise.json"))) -> None:
    """Measure judge noise by re-judging identical answers, uncached."""
    report = measure_noise(get_settings(), load_pipeline_config(config), repeats, jitter=jitter)
    write_noise(report, out)
    typer.echo(json.dumps(report["metrics"], indent=1))


@app.command("judge-lock")
def judge_lock(check: bool = typer.Option(False, help="Only verify; exit 2 on violations")) -> None:
    """Record judge prompt versions and fingerprints."""
    if check:
        problems = lock_violations()
        for p in problems:
            typer.secho(p, fg="red")
        raise typer.Exit(ExitCode.ERROR if problems else 0)
    write_lock()
    typer.echo("judge_prompts.lock.json updated")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the API and dashboard."""
    import uvicorn

    from ragate.api import create_app

    uvicorn.run(create_app(), host=host, port=port)


@app.command()
def e2e(config: Path = CONFIG,
        baseline_path: Path = typer.Option(Path("baselines/baseline.json"), "--baseline"),
        report: Path = typer.Option(Path("reports/gate.md"))) -> None:
    """The whole gate in one command: index, dataset checks, eval, compare, report."""
    s = get_settings()
    if lock_violations():
        _fail("judge prompt changed without a version bump; run `ragate judge-lock --check`")
    cfg = load_pipeline_config(config)
    build_pipeline(cfg, s)
    try:
        run = run_eval(s, cfg, runs=_runs())
    except (DatasetError, ProviderConfigError) as exc:
        _fail(str(exc))
    write_run_file(run, s.reports_dir / "candidate.json")
    if not baseline_path.exists():
        _fail(f"no baseline at {baseline_path}; create one with `ragate baseline`")
    code = _gate(baseline_path, run, report, s.baselines_dir / "noise.json")
    typer.secho(f"decision exit code {code}", fg="green" if code == 0 else "red", err=True)
    raise typer.Exit(code)


def main() -> None:  # pragma: no cover
    sys.exit(app())
