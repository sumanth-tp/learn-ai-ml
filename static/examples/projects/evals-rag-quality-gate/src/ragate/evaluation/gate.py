"""The release gate: baseline vs candidate, per-metric rules, a promote/block decision.

Exit codes: 0 promote, 1 block (a regression or a hard limit), 2 error (the runs are
not comparable, or an input is missing). CI treats anything non-zero as a failed check.
"""

from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ragate.config import load_yaml
from ragate.evaluation.results import RunResult
from ragate.evaluation.stats import paired_bootstrap
from ragate.metrics.aggregate import AGGREGATES


class ExitCode(IntEnum):
    PROMOTE = 0
    BLOCK = 1
    ERROR = 2


class MetricRule(BaseModel):
    direction: Literal["higher", "lower"]
    min_delta: float = Field(default=0.0, ge=0, description="absolute tolerance")
    rel_delta: float = Field(default=0.0, ge=0, description="tolerance relative to baseline")
    noise_k: float = Field(default=2.0, ge=0, description="multiples of measured noise std")
    floor: float | None = None
    ceiling: float | None = None
    enforce: Literal["block", "warn"] = "block"


class BootstrapConfig(BaseModel):
    resamples: int = 2000
    confidence: float = 0.95
    seed: int = 7


class GateConfig(BaseModel):
    metrics: dict[str, MetricRule]
    bootstrap: BootstrapConfig = BootstrapConfig()
    require_same_dataset: bool = True
    require_same_judge: bool = True


Status = Literal["pass", "improved", "regressed", "limit", "missing", "n/a"]


class MetricVerdict(BaseModel):
    metric: str
    direction: str
    baseline: float | None
    candidate: float | None
    delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    tolerance: float = 0.0
    status: Status
    enforce: str
    note: str = ""

    @property
    def blocking(self) -> bool:
        return self.enforce == "block" and self.status in ("regressed", "limit", "missing")


class GateResult(BaseModel):
    decision: Literal["promote", "block", "error"]
    exit_code: int
    reasons: list[str]
    verdicts: list[MetricVerdict]
    baseline_run: str
    candidate_run: str


def load_gate_config(path: Path) -> GateConfig:
    return GateConfig.model_validate(load_yaml(path))


def load_noise(path: Path | None, judge_model_id: str | None = None) -> dict[str, float]:
    """Per-metric noise std. Ignored when it was measured with a different judge."""
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    if judge_model_id is not None and data.get("judge") != judge_model_id:
        return {}
    return {k: float(v["std"]) for k, v in data.get("metrics", {}).items()}


def compatibility_errors(base: RunResult, cand: RunResult, cfg: GateConfig) -> list[str]:
    errors = []
    if cfg.require_same_dataset and (base.dataset_version, base.dataset_sha) != (
        cand.dataset_version, cand.dataset_sha
    ):
        errors.append(
            f"dataset differs: baseline {base.dataset_version} vs candidate "
            f"{cand.dataset_version}; re-run the baseline on the new dataset first"
        )
    if cfg.require_same_judge and base.judge.fingerprint() != cand.judge.fingerprint():
        errors.append(
            "judge differs (model, backend or prompt versions); scores are not comparable. "
            "Re-baseline with the new judge in a separate PR"
        )
    return errors


def _verdict(
    name: str, rule: MetricRule, base: RunResult, cand: RunResult, cfg: GateConfig,
    noise: dict[str, float],
) -> MetricVerdict:
    agg = AGGREGATES[name]
    b, c = agg(base.items), agg(cand.items)
    v = MetricVerdict(metric=name, direction=rule.direction, baseline=b, candidate=c,
                      status="pass", enforce=rule.enforce)
    if c is None:
        v.status = "n/a" if b is None else "missing"
        v.note = "no items produced this metric"
        return v

    # Hard limits apply whatever the baseline did: a safety floor is not relative.
    limit_notes = []
    if rule.floor is not None and c < rule.floor:
        limit_notes.append(f"below floor {rule.floor:g}")
    if rule.ceiling is not None and c > rule.ceiling:
        limit_notes.append(f"above ceiling {rule.ceiling:g}")
    if b is None:
        v.status = "limit" if limit_notes else "pass"
        v.note = "; ".join(limit_notes) or "no baseline value"
        return v

    v.delta = c - b
    v.tolerance = max(rule.min_delta, rule.rel_delta * abs(b), rule.noise_k * noise.get(name, 0.0))
    ci = paired_bootstrap(base.items, cand.items, agg, resamples=cfg.bootstrap.resamples,
                          confidence=cfg.bootstrap.confidence, seed=cfg.bootstrap.seed)
    if ci is not None:
        v.ci_low, v.ci_high = ci.low, ci.high
    sign = 1.0 if rule.direction == "higher" else -1.0
    gain = sign * v.delta
    # Gain CI: flip the interval when lower is better.
    if ci is not None:
        g_low, g_high = sorted((sign * ci.low, sign * ci.high))
    else:
        g_low = g_high = gain

    if limit_notes:
        v.status, v.note = "limit", "; ".join(limit_notes)
    elif gain < -v.tolerance and g_high < 0:
        v.status, v.note = "regressed", "worse beyond tolerance, and the CI excludes zero"
    elif gain < -v.tolerance:
        v.status, v.note = "pass", "worse beyond tolerance but not significant (CI spans 0)"
    elif gain > v.tolerance and g_low > 0:
        v.status = "improved"
    return v


def compare(
    base: RunResult, cand: RunResult, cfg: GateConfig, noise: dict[str, float] | None = None
) -> GateResult:
    noise = noise or {}
    compat = compatibility_errors(base, cand, cfg)
    if compat:
        return GateResult(decision="error", exit_code=ExitCode.ERROR, reasons=compat,
                          verdicts=[], baseline_run=base.run_id, candidate_run=cand.run_id)
    unknown = sorted(set(cfg.metrics) - set(AGGREGATES))
    if unknown:
        return GateResult(decision="error", exit_code=ExitCode.ERROR,
                          reasons=[f"gate config names unknown metrics: {unknown}"],
                          verdicts=[], baseline_run=base.run_id, candidate_run=cand.run_id)
    verdicts = [_verdict(n, r, base, cand, cfg, noise) for n, r in cfg.metrics.items()]
    blocking = [v for v in verdicts if v.blocking]
    reasons = [f"{v.metric}: {v.status} ({v.note})" for v in blocking]
    decision = "block" if blocking else "promote"
    return GateResult(
        decision=decision,
        exit_code=ExitCode.BLOCK if blocking else ExitCode.PROMOTE,
        reasons=reasons or ["no blocking regressions"],
        verdicts=verdicts,
        baseline_run=base.run_id,
        candidate_run=cand.run_id,
    )
