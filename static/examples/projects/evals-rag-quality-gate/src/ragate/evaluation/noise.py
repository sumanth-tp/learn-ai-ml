"""Measure judge noise: answer once, then re-judge the same answers N times, uncached.

The std of each aggregate across repeats is the noise band the gate adds to its
tolerance (noise_k x std). With a temperature-0 LLM judge it is small but not zero;
with the heuristic stub it is zero unless jitter is simulated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ragate.config import PipelineConfig
from ragate.evaluation.build import build_runner, load_checked_dataset
from ragate.metrics.aggregate import aggregate_all
from ragate.settings import Settings


def measure_noise(
    settings: Settings, config: PipelineConfig, repeats: int, *, jitter: float = 0.0
) -> dict:
    manifest, items = load_checked_dataset(settings, None)
    answers = build_runner(settings, config, use_cache=False).answer_all(items)
    per_metric: dict[str, list[float]] = {}
    judge_id = ""
    for rep in range(repeats):
        runner = build_runner(settings, config, use_cache=False, jitter=jitter, seed=rep)
        judge_id = runner.judge_info.model_id
        aggs = aggregate_all(runner.score_all(items, answers))
        for name, value in aggs.items():
            if value is not None:
                per_metric.setdefault(name, []).append(value)
    judge_metrics = {"faithfulness", "answer_relevancy", "context_relevance",
                     "contextual_recall", "correctness"}
    metrics = {
        name: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        for name, vals in per_metric.items() if name in judge_metrics
    }
    return {"dataset": manifest.version, "repeats": repeats, "jitter": jitter,
            "judge": judge_id,
            "metrics": metrics}


def write_noise(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
