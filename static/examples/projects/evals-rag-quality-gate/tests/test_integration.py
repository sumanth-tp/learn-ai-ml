"""The whole system with fakes: index, dataset checks, eval, store, gate, CLI, experiments."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ragate.cli import app
from ragate.config import PipelineConfig
from ragate.dataset.store import DatasetError
from ragate.evaluation.build import build_runner, load_checked_dataset, run_eval
from ragate.evaluation.experiments import run_experiments
from ragate.evaluation.noise import measure_noise
from ragate.evaluation.store import RunStore, load_run_file
from ragate.providers import ProviderConfigError
from ragate.settings import Settings

cli = CliRunner()


def test_full_eval_on_v1_meets_the_shipped_floors(
    settings: Settings, config: PipelineConfig
) -> None:
    run = run_eval(settings, config, runs=RunStore(settings.runs_db))
    a = run.aggregates
    assert len(run.items) == 40 and a["judge_error_rate"] == 0.0
    assert a["recall_at_k"] >= 0.9 and a["faithfulness"] >= 0.85
    assert a["refusal_rate"] >= 0.8 and a["pii_leak_rate"] == 0.0
    assert a["cost_per_query_usd"] > 0 and a["tokens_per_query"] > 0
    unanswerable = [i for i in run.items if i.question_type == "unanswerable"]
    assert all(not i.scores for i in unanswerable)  # only safety metrics apply


def test_identical_eval_is_served_from_the_store(
    settings: Settings, config: PipelineConfig
) -> None:
    runs = RunStore(settings.runs_db)
    first = run_eval(settings, config, runs=runs)
    second = run_eval(settings, config, runs=runs)
    assert first.run_id == second.run_id and first.created_at == second.created_at
    changed = run_eval(settings, config.with_overrides("test", {"k": 3}), runs=runs)
    assert changed.run_id != first.run_id


def test_pipeline_failure_on_one_item_is_recorded_not_fatal(
    settings: Settings, config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, items = load_checked_dataset(settings, "v1")
    runner = build_runner(settings, config)
    real_ask = runner.pipeline.ask

    def flaky_ask(question: str):  # type: ignore[no-untyped-def]
        if question == items[0].question:
            raise TimeoutError("LLM timed out")
        return real_ask(question)

    monkeypatch.setattr(runner.pipeline, "ask", flaky_ask)
    run = runner.run(manifest, items)
    broken = next(i for i in run.items if i.item_id == items[0].item_id)
    assert broken.errors and "timed out" in broken.errors[0]
    assert run.aggregates["judge_error_rate"] == pytest.approx(1 / 40)


def test_deepeval_backend_needs_a_real_judge(settings: Settings, config: PipelineConfig) -> None:
    with pytest.raises(ProviderConfigError):
        build_runner(settings, config, judge_backend="deepeval")


def test_real_provider_without_key_fails_fast(settings: Settings, config: PipelineConfig) -> None:
    real = settings.model_copy(update={"provider": "openai"})
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        build_runner(real, config)


def test_eval_refuses_a_dataset_with_errors(settings: Settings, tmp_path: Path) -> None:
    golden = tmp_path / "data" / "golden" / "v1"
    golden.mkdir(parents=True)
    src = settings.golden_dir / "v1"
    lines = (src / "golden.jsonl").read_text().splitlines()
    (golden / "golden.jsonl").write_text("\n".join(lines[:10]) + "\n")  # too few per stratum
    (golden / "manifest.json").write_text((src / "manifest.json").read_text())
    (tmp_path / "data" / "corpus").symlink_to(settings.corpus_dir)
    broken = settings.model_copy(update={"data_dir": tmp_path / "data"})
    with pytest.raises(DatasetError):
        load_checked_dataset(broken, "v1")


def test_noise_is_zero_for_the_stub_and_positive_with_jitter(
    settings: Settings, config: PipelineConfig
) -> None:
    flat = measure_noise(settings, config, repeats=3)
    assert all(m["std"] < 1e-9 for m in flat["metrics"].values())
    noisy = measure_noise(settings, config, repeats=3, jitter=0.1)
    assert noisy["metrics"]["faithfulness"]["std"] > 0
    assert noisy["judge"] != flat["judge"]


def test_experiments_table_and_recommendation(settings: Settings, tmp_path: Path) -> None:
    exp = {
        "base": "config/pipeline.yaml",
        "metrics": ["recall_at_k", "correctness", "pii_leak_rate"],
        "select_by": "correctness",
        "constraints": {"pii_leak_rate": {"max": 0.0}},
        "variants": [
            {"name": "baseline", "overrides": {}},
            {"name": "k-1", "overrides": {"k": 1}},
            {
                "name": "unsafe",
                "overrides": {
                    "include_restricted": True,
                    "pii_redaction": False,
                    "input_guard": False,
                },
            },
        ],
    }
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(exp))
    results, best = run_experiments(settings, path, RunStore(settings.runs_db), tmp_path / "out")
    by_name = {r.name: r for r in results}
    assert by_name["k-1"].aggregates["recall_at_k"] < by_name["baseline"].aggregates["recall_at_k"]
    assert by_name["unsafe"].aggregates["pii_leak_rate"] > 0
    assert best == "baseline"
    assert "| k-1 |" in (tmp_path / "out" / "experiments.md").read_text()


# ------------------------------------------------------------------ CLI, end to end


def test_cli_e2e_promotes_the_committed_config(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    assert cli.invoke(app, ["baseline", "--out", str(baseline)]).exit_code == 0
    result = cli.invoke(
        app, ["e2e", "--baseline", str(baseline), "--report", str(tmp_path / "gate.md")]
    )
    assert result.exit_code == 0, result.output
    assert "PROMOTE" in (tmp_path / "gate.md").read_text()


def test_cli_gate_blocks_a_regressing_candidate(tmp_path: Path) -> None:
    baseline, cand = tmp_path / "baseline.json", tmp_path / "cand.json"
    cli.invoke(app, ["baseline", "--out", str(baseline)])
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(
        yaml.safe_dump(
            {
                **PipelineConfig().model_dump(),
                "name": "bad",
                "k": 1,
                "hybrid": False,
                "reranker": "none",
            }
        )
    )
    assert cli.invoke(app, ["eval", "--config", str(bad_cfg), "--out", str(cand)]).exit_code == 0
    result = cli.invoke(
        app,
        [
            "gate",
            "--baseline",
            str(baseline),
            "--candidate",
            str(cand),
            "--report",
            str(tmp_path / "gate.md"),
            "--noise",
            str(tmp_path / "none.json"),
        ],
    )
    assert result.exit_code == 1
    report = (tmp_path / "gate.md").read_text()
    assert "BLOCK" in report and "recall_at_k" in report
    assert load_run_file(cand).config["k"] == 1


def test_cli_gate_errors_on_missing_files(tmp_path: Path) -> None:
    result = cli.invoke(app, ["gate", "--candidate", str(tmp_path / "nope.json")])
    assert result.exit_code == 2


def test_cli_dataset_check_and_judge_lock() -> None:
    assert cli.invoke(app, ["dataset", "check"]).exit_code == 0
    assert cli.invoke(app, ["judge-lock", "--check"]).exit_code == 0


def test_cli_synth_then_freeze_a_new_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    data = tmp_path / "data"
    shutil.copytree(Path("data"), data)
    monkeypatch.setenv("RAGATE_DATA_DIR", str(data))
    from ragate.settings import get_settings

    get_settings.cache_clear()
    pending = tmp_path / "pending.csv"
    assert (
        cli.invoke(app, ["dataset", "synth", "--per-type", "1", "--out", str(pending)]).exit_code
        == 0
    )
    pending.write_text(pending.read_text().replace(",pending,,", ",approved,reviewer-1,"))
    freeze = ["dataset", "freeze", "--review", str(pending), "--version", "v2", "--base", "v1"]
    first = cli.invoke(app, freeze)
    if first.exit_code != 0:
        # The checks caught bad synthetic rows; the reviewer rejects exactly those.
        flagged = {
            line.split()[2] for line in first.output.splitlines() if line.startswith("error")
        }
        rows = []
        for line in pending.read_text().splitlines():
            if line.split(",", 1)[0] in flagged:
                line = line.replace(",approved,", ",rejected,")
            rows.append(line)
        pending.write_text("\n".join(rows) + "\n")
    result = cli.invoke(app, freeze)
    assert result.exit_code == 0, result.output
    assert (data / "golden" / "v2" / "manifest.json").exists()


def test_retention_prunes_old_runs_only(settings: Settings, config: PipelineConfig) -> None:
    store = RunStore(settings.runs_db)
    run = run_eval(settings, config, runs=store)
    old = run.model_copy(update={"run_id": "old-run", "created_at": "2020-01-01T00:00:00+00:00"})
    store.save(old)
    assert store.prune(days=180) == 1
    assert store.get("old-run") is None and store.get(run.run_id) is not None
