"""Experiment tracking: run a grid of config variants and tabulate the results.

Each variant is a normal eval run stored in the run store (so the dashboard shows it),
plus one row in reports/experiments.md and reports/experiments.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ragate.config import load_pipeline_config, load_yaml
from ragate.evaluation.build import run_eval
from ragate.evaluation.report import experiments_table
from ragate.evaluation.results import RunResult
from ragate.evaluation.store import RunStore
from ragate.settings import Settings


class Bound(BaseModel):
    min: float | None = None
    max: float | None = None


class Variant(BaseModel):
    name: str
    overrides: dict = Field(default_factory=dict)


class ExperimentsConfig(BaseModel):
    base: Path
    metrics: list[str]
    select_by: str
    constraints: dict[str, Bound] = Field(default_factory=dict)
    variants: list[Variant]


def satisfies(run: RunResult, constraints: dict[str, Bound]) -> bool:
    for metric, bound in constraints.items():
        value = run.aggregates.get(metric)
        if value is None:
            return False
        if bound.min is not None and value < bound.min:
            return False
        if bound.max is not None and value > bound.max:
            return False
    return True


def run_experiments(
    settings: Settings, path: Path, runs: RunStore, out_dir: Path, *, force: bool = False
) -> tuple[list[RunResult], str | None]:
    cfg = ExperimentsConfig.model_validate(load_yaml(path))
    base = load_pipeline_config(cfg.base)
    results = [
        run_eval(settings, base.with_overrides(v.name, v.overrides), runs=runs, force=force)
        for v in cfg.variants
    ]
    eligible = [r for r in results if satisfies(r, cfg.constraints)]
    best = max(eligible, key=lambda r: r.aggregates.get(cfg.select_by) or 0.0, default=None)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = experiments_table(results, cfg.metrics)
    rec = (f"Recommended: **{best.name}** (highest {cfg.select_by} among variants meeting "
           f"the constraints)." if best else "No variant met the constraints.")
    (out_dir / "experiments.md").write_text(f"# Experiments\n\n{table}\n{rec}\n")
    (out_dir / "experiments.json").write_text(json.dumps(
        [{"name": r.name, "run_id": r.run_id, "config": r.config, "aggregates": r.aggregates,
          "eligible": r in eligible} for r in results], indent=1))
    return results, best.name if best else None
