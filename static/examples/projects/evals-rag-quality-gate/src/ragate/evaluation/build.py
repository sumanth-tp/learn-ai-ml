"""Assemble an EvalRunner from settings. The single place that wires real vs fake parts."""

from __future__ import annotations

from typing import Literal

from ragate.config import PipelineConfig
from ragate.dataset import checks, store
from ragate.dataset.store import DatasetError
from ragate.evaluation.results import RunResult
from ragate.evaluation.runner import EvalRunner
from ragate.evaluation.store import RunStore
from ragate.judge.cache import JudgeCache
from ragate.judge.factory import build_judge
from ragate.log import get_logger
from ragate.metrics.deepeval_models import LangChainDeepEvalModel, StubGEvalModel
from ragate.metrics.geval import CorrectnessGEval
from ragate.metrics.rag_judges import DeepEvalRagMetrics, NativeRagMetrics, RagJudgeMetrics
from ragate.providers import ProviderConfigError, chat_model
from ragate.rag.corpus import load_corpus
from ragate.rag.pipeline import build_pipeline
from ragate.settings import Settings

log = get_logger(__name__)
JudgeBackend = Literal["native", "deepeval"]


def build_runner(
    settings: Settings,
    config: PipelineConfig,
    *,
    judge_backend: JudgeBackend = "native",
    use_cache: bool = True,
    jitter: float = 0.0,
    seed: int = 0,
) -> EvalRunner:
    pipeline = build_pipeline(config, settings)
    judge = build_judge(settings, use_cache=use_cache, jitter=jitter, seed=seed)
    rag_metrics: RagJudgeMetrics
    if settings.judge_provider == "fake":
        if judge_backend == "deepeval":
            raise ProviderConfigError("the deepeval backend needs a real judge provider")
        rag_metrics = NativeRagMetrics(judge)
        geval = CorrectnessGEval(StubGEvalModel())
    else:
        cache = JudgeCache(settings.judge_cache_db) if use_cache else None
        lc = chat_model(settings, role="judge", temperature=settings.judge_temperature)
        de_model = LangChainDeepEvalModel(lc, settings.judge_model, cache)
        rag_metrics = (
            DeepEvalRagMetrics(de_model) if judge_backend == "deepeval" else NativeRagMetrics(judge)
        )
        geval = CorrectnessGEval(de_model)
    model = config.generator_model or settings.chat_model
    provider = {
        "generator": f"{settings.provider}:{model}",
        "embeddings": f"{settings.embedding_provider}:{settings.embedding_model}",
        "judge": judge.model_id,
    }
    return EvalRunner(pipeline, rag_metrics, geval, judge_model_id=judge.model_id,
                      provider=provider)


def load_checked_dataset(settings: Settings, version: str | None):  # type: ignore[no-untyped-def]
    """Load a frozen version and refuse to evaluate on a dataset with check errors."""
    root = settings.golden_dir
    manifest, items = store.load(root, version or store.latest_version(root))
    findings = checks.run_checks(items, load_corpus(settings.corpus_dir))
    for f in findings:
        (log.error if f.level == "error" else log.warning)(
            "dataset_check", check=f.check, item=f.item_id, message=f.message
        )
    if checks.has_errors(findings):
        raise DatasetError(f"dataset {manifest.version} failed checks; see log")
    return manifest, items


def run_eval(
    settings: Settings,
    config: PipelineConfig,
    *,
    runs: RunStore,
    dataset_version: str | None = None,
    judge_backend: JudgeBackend = "native",
    force: bool = False,
) -> RunResult:
    """Idempotent: an identical (config, dataset, judge, provider) run is served from the store."""
    manifest, items = load_checked_dataset(settings, dataset_version)
    runner = build_runner(settings, config, judge_backend=judge_backend)
    run_id = runner.run_id(manifest)
    if not force and (existing := runs.get(run_id)) is not None:
        log.info("eval_reused", run_id=run_id)
        return existing
    run = runner.run(manifest, items)
    runs.save(run)
    return run
