"""Run the RAG app over the golden set and score every item.

Answering and scoring are separate phases: the noise tool re-scores the *same*
answers many times to isolate judge noise from generator noise.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from ragate.config import PipelineConfig
from ragate.dataset.store import Manifest
from ragate.evaluation.results import ItemResult, JudgeInfo, RunResult
from ragate.judge.base import JudgeError
from ragate.judge.prompts import fingerprint_all
from ragate.log import get_logger
from ragate.metrics import retrieval
from ragate.metrics.aggregate import aggregate_all
from ragate.metrics.generation import citation_scores
from ragate.metrics.geval import CorrectnessGEval
from ragate.metrics.rag_judges import RagJudgeMetrics
from ragate.models import ExpectedBehaviour, GoldenItem, RagAnswer
from ragate.pii import leaked_pii
from ragate.rag.pipeline import RagPipeline

log = get_logger(__name__)


@dataclass
class Answered:
    answer: RagAnswer | None
    latency_ms: float
    error: str | None = None


def git_sha() -> str:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha[:12]
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True,
                             text=True, timeout=5, check=False)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def code_fingerprint() -> str:
    """Hash of the installed ragate source. Part of the run id, so a code change is never
    served a stale stored run (the git sha alone misses uncommitted edits)."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


class EvalRunner:
    def __init__(
        self,
        pipeline: RagPipeline,
        rag_metrics: RagJudgeMetrics,
        correctness: CorrectnessGEval,
        *,
        judge_model_id: str,
        provider: dict[str, str],
    ) -> None:
        self.pipeline = pipeline
        self.rag_metrics = rag_metrics
        self.correctness = correctness
        self.judge_info = JudgeInfo(
            model_id=judge_model_id,
            backend=rag_metrics.backend,
            geval_model=correctness.model_id,
            prompts=fingerprint_all(),
            deepeval_version=version("deepeval"),
        )
        self.provider = provider

    @property
    def config(self) -> PipelineConfig:
        return self.pipeline.config

    def run_id(self, manifest: Manifest) -> str:
        raw = json.dumps([self.config.config_hash(), manifest.sha256, self.judge_info.fingerprint(),
                          self.provider, code_fingerprint()], sort_keys=True)
        return f"{self.config.name}-{hashlib.sha256(raw.encode()).hexdigest()[:10]}"

    # ---------------------------------------------------------------- phase 1
    def answer_all(self, items: list[GoldenItem]) -> dict[str, Answered]:
        out: dict[str, Answered] = {}
        for item in items:
            start = time.perf_counter()
            try:
                ans = self.pipeline.ask(item.question)
                out[item.item_id] = Answered(ans, (time.perf_counter() - start) * 1000)
            except Exception as exc:  # one broken item must not abort a 500-item run
                log.error("pipeline_failed", item=item.item_id, error=repr(exc))
                out[item.item_id] = Answered(None, (time.perf_counter() - start) * 1000,
                                             f"pipeline: {exc!r}")
        return out

    # ---------------------------------------------------------------- phase 2
    def _judge(self, item: ItemResult, name: str, fn) -> None:  # type: ignore[no-untyped-def]
        try:
            item.scores[name] = fn()
        except JudgeError as exc:
            item.scores[name] = None
            item.errors.append(f"{name}: {exc}")

    def score_item(self, item: GoldenItem, answered: Answered) -> ItemResult:
        res = ItemResult(
            item_id=item.item_id,
            question_type=item.question_type.value,
            expected_behaviour=item.expected_behaviour.value,
            question=item.question,
            reference=item.reference_answer,
            latency_ms=answered.latency_ms,
        )
        ans = answered.answer
        if ans is None:
            res.errors.append(answered.error or "pipeline: no answer")
            return res
        res.answer = ans.answer
        res.citations = ans.citations
        res.retrieved = [c.chunk.chunk_id for c in ans.contexts]
        res.refused = ans.refused
        res.pii_leaks = [kind for kind, _ in leaked_pii(ans.answer, item.question)]
        res.input_tokens = ans.usage.input_tokens
        res.output_tokens = ans.usage.output_tokens
        res.total_tokens = ans.usage.total_tokens
        res.cost_usd = ans.cost_usd

        if item.expected_behaviour == ExpectedBehaviour.REFUSE or not item.evidence:
            return res  # safety metrics only; they are computed from `refused` and `pii_leaks`

        k, ev, ctx = self.config.k, item.evidence, ans.contexts
        res.scores.update({
            "recall_at_k": retrieval.recall_at_k(ctx, ev, k),
            "precision_at_k": retrieval.precision_at_k(ctx, ev, k),
            "hit_at_k": retrieval.hit_at_k(ctx, ev, k),
            "mrr": retrieval.reciprocal_rank(ctx, ev),
            "ndcg_at_k": retrieval.ndcg_at_k(ctx, ev, k),
            "contextual_precision": retrieval.contextual_precision(ctx, ev, k),
        })
        if not ans.refused:
            res.scores.update(citation_scores(ans, ev))
        m = self.rag_metrics
        ref = item.reference_answer
        self._judge(res, "contextual_recall", lambda: m.contextual_recall(ans, ref))
        self._judge(res, "context_relevance", lambda: m.context_relevance(ans))
        if ans.refused:
            # A refusal makes no claims, so faithfulness is not applicable; it is,
            # however, useless as an answer, and correctness will say so.
            res.scores["faithfulness"] = None
            res.scores["answer_relevancy"] = 0.0
        else:
            self._judge(res, "faithfulness", lambda: m.faithfulness(ans))
            self._judge(res, "answer_relevancy", lambda: m.answer_relevancy(ans))
        self._judge(res, "correctness",
                    lambda: self.correctness.score(item.question, ans.answer, ref))
        return res

    def score_all(self, items: list[GoldenItem], answers: dict[str, Answered]) -> list[ItemResult]:
        return [self.score_item(i, answers[i.item_id]) for i in items]

    # ---------------------------------------------------------------- both
    def run(self, manifest: Manifest, items: list[GoldenItem]) -> RunResult:
        start = time.perf_counter()
        log.info("eval_started", config=self.config.name, items=len(items),
                 dataset=manifest.version)
        results = self.score_all(items, self.answer_all(items))
        run = RunResult(
            run_id=self.run_id(manifest),
            name=self.config.name,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            git_sha=git_sha(),
            code_version=code_fingerprint(),
            config=self.config.model_dump(),
            config_hash=self.config.config_hash(),
            dataset_version=manifest.version,
            dataset_sha=manifest.sha256,
            judge=self.judge_info,
            provider=self.provider,
            duration_s=round(time.perf_counter() - start, 3),
            items=results,
            aggregates=aggregate_all(results),
        )
        log.info("eval_finished", run_id=run.run_id, duration_s=run.duration_s,
                 errors=sum(bool(i.errors) for i in results))
        return run
