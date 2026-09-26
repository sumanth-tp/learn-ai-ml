"""Build the configured judge (LLM or heuristic), wrapped in the SQLite cache."""

from __future__ import annotations

from ragate.judge.base import Judge
from ragate.judge.cache import CachedJudge, JudgeCache
from ragate.judge.heuristic import HeuristicJudge
from ragate.judge.llm_judge import LLMJudge
from ragate.providers import chat_model
from ragate.settings import Settings


def build_judge(settings: Settings, *, use_cache: bool = True, jitter: float = 0.0,
                seed: int = 0) -> Judge:
    judge: Judge
    if settings.judge_provider == "fake":
        judge = HeuristicJudge(jitter=jitter, seed=seed)
    else:
        model = chat_model(settings, role="judge", temperature=settings.judge_temperature)
        judge = LLMJudge(model, settings.judge_model, settings.judge_temperature,
                         attempts=settings.max_retries)
    if use_cache:
        return CachedJudge(judge, JudgeCache(settings.judge_cache_db))
    return judge
