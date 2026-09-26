"""Composition root: builds every component from Settings. The API, the CLI, the
simulator and the tests all get their wiring from here, so there is one way to
assemble the system."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from agentmon.agent.backend import FakeBankBackend
from agentmon.agent.service import AgentService
from agentmon.clock import Clock, SystemClock
from agentmon.config import Settings
from agentmon.evals.judges import build_judges
from agentmon.evals.pipeline import CascadingEvaluator, OnlineEvalPipeline
from agentmon.evals.sampling import (
    AlwaysSampleRules,
    CompositeSampler,
    RandomSampler,
    StratifiedSampler,
)
from agentmon.evals.workers import EvalWorkerPool
from agentmon.llm.factory import build_agent_model, build_judge_model
from agentmon.store import Store
from agentmon.tracing import Tracing, build_tracer_provider


@dataclass
class Runtime:
    settings: Settings
    clock: Clock
    store: Store
    backend: FakeBankBackend
    tracing: Tracing
    service: AgentService
    pipeline: OnlineEvalPipeline
    cascade: CascadingEvaluator
    workers: EvalWorkerPool
    judge_model: BaseChatModel


def build_sampler(settings: Settings) -> CompositeSampler:
    return CompositeSampler(
        [
            AlwaysSampleRules(),
            StratifiedSampler(settings.sample_intent_rates, settings.sample_seed),
            RandomSampler(settings.sample_random_rate, settings.sample_seed),
        ]
    )


def build_runtime(
    settings: Settings,
    clock: Clock | None = None,
    store: Store | None = None,
    backend: FakeBankBackend | None = None,
    agent_model: BaseChatModel | None = None,
    judge_model: BaseChatModel | None = None,
) -> Runtime:
    clock = clock or SystemClock()
    store = store or Store(settings.db_path)
    backend = backend or FakeBankBackend.from_file(settings.seed_path, settings.backend_fault_rate)
    tracing = Tracing(
        build_tracer_provider(store, settings.service_name, settings.otlp_endpoint), clock
    )
    pipeline = OnlineEvalPipeline(
        store, build_sampler(settings), clock, settings.slo_latency_p95_ms
    )
    service = AgentService(
        settings,
        store,
        backend,
        agent_model or build_agent_model(settings, clock),
        tracing,
        clock,
        sink=pipeline,
    )
    judge_model = judge_model or build_judge_model(settings, clock)
    cascade = CascadingEvaluator(
        store, build_judges(judge_model, settings.cost_usd), settings.judge_budget_usd_per_day
    )
    workers = EvalWorkerPool(
        store,
        cascade,
        clock,
        concurrency=settings.eval_workers,
        timeout_s=settings.judge_timeout_s,
        max_attempts=settings.eval_max_attempts,
        backoff_base_s=settings.eval_backoff_base_s,
    )
    return Runtime(
        settings, clock, store, backend, tracing, service, pipeline, cascade, workers, judge_model
    )
