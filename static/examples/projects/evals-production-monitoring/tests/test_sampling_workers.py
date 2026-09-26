import asyncio

from test_judges_cascade import make_trace

from agentmon.evals.heuristics import run_heuristics
from agentmon.evals.judges import JudgeError, build_judges
from agentmon.evals.pipeline import CascadeOutcome, CascadingEvaluator, OnlineEvalPipeline
from agentmon.evals.sampling import (
    AlwaysSampleRules,
    CompositeSampler,
    RandomSampler,
    StratifiedSampler,
    unit_hash,
)
from agentmon.evals.workers import EvalWorkerPool
from agentmon.llm.fake_judge import RuleJudgeModel
from agentmon.models import Feedback
from agentmon.store import Store
from agentmon.clock import SimClock


def test_unit_hash_is_deterministic_and_uniform() -> None:
    assert unit_hash("a", "s") == unit_hash("a", "s") != unit_hash("a", "t")
    values = [unit_hash(str(i), "s") for i in range(5000)]
    assert 0.45 < sum(v < 0.5 for v in values) / 5000 < 0.55


def test_random_sampler_rate() -> None:
    s = RandomSampler(0.1, "seed")
    hits = sum(s.decide(make_trace(trace_id=f"t{i}")).sampled for i in range(4000))
    assert 320 < hits < 480


def test_always_rules_and_composite_order() -> None:
    sampler = CompositeSampler([AlwaysSampleRules(), StratifiedSampler({"transfer": 1.0}, "s"),
                                RandomSampler(0.0, "s")])
    assert sampler.decide(make_trace(status="error")).reason == "always:error"
    assert sampler.decide(make_trace(flags=["blocked:toxic_request"])).reason == "always:guardrail"
    assert sampler.decide(make_trace(intent="transfer")).reason == "stratified:transfer"
    assert not sampler.decide(make_trace()).sampled


def _pipeline(store: Store) -> OnlineEvalPipeline:
    return OnlineEvalPipeline(store, CompositeSampler([RandomSampler(1.0, "s")]), SimClock(0), 4000)


def test_negative_feedback_enqueues_and_opens_review() -> None:
    store = Store(":memory:")
    t = make_trace()
    store.insert_trace(t)
    pipe = OnlineEvalPipeline(store, CompositeSampler([RandomSampler(0.0, "s")]), SimClock(0), 4000)
    pipe.ingest(t)
    assert store.get_job(t.trace_id) is None
    pipe.on_feedback(Feedback(trace_id=t.trace_id, rating=-1, correction="wrong"))
    assert store.get_job(t.trace_id)["reason"] == "always:negative_feedback"
    assert store.review_items("pending")[0]["reason"] == "feedback:thumbs_down"


async def test_workers_drain_queue_and_are_idempotent() -> None:
    store = Store(":memory:")
    pipe = _pipeline(store)
    for i in range(10):
        t = make_trace(trace_id=f"t{i}", request_id=f"r{i}")
        store.insert_trace(t)
        pipe.ingest(t)
    cascade = CascadingEvaluator(store, build_judges(RuleJudgeModel()), 10.0)
    pool = EvalWorkerPool(store, cascade, SimClock(0), concurrency=3, backoff_base_s=0)
    assert await pool.run_once() == 10
    assert store.job_counts() == {"done": 10}
    assert len([e for e in store.evals_for("t3") if e.tier == "judge"]) == 3
    assert await pool.run_once() == 0  # nothing left; results are not duplicated
    assert len([e for e in store.evals_for("t3") if e.tier == "judge"]) == 3


class Flaky(CascadingEvaluator):
    def __init__(self, store: Store, failures: int, exc: Exception) -> None:
        super().__init__(store, [], 1.0)
        self.failures, self.exc, self.calls = failures, exc, 0

    async def evaluate(self, trace):  # type: ignore[override]
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return CascadeOutcome([], "done")


async def test_worker_retries_transient_judge_failures() -> None:
    store = Store(":memory:")
    t = make_trace()
    store.insert_trace(t)
    store.enqueue_eval(t.trace_id, "random", t.ts)
    flaky = Flaky(store, failures=2, exc=JudgeError("bad json"))
    pool = EvalWorkerPool(store, flaky, SimClock(0), max_attempts=3, backoff_base_s=0)
    await pool.run_once()
    assert store.get_job(t.trace_id)["status"] == "done"
    assert pool.stats["retries"] == 2


async def test_worker_dead_letters_after_max_attempts_on_timeout() -> None:
    store = Store(":memory:")
    t = make_trace()
    store.insert_trace(t)
    store.enqueue_eval(t.trace_id, "random", t.ts)

    class Slow(CascadingEvaluator):
        async def evaluate(self, trace):  # type: ignore[override]
            await asyncio.sleep(1)
            return CascadeOutcome([], "done")

    pool = EvalWorkerPool(store, Slow(store, [], 1.0), SimClock(0), timeout_s=0.01,
                          max_attempts=2, backoff_base_s=0)
    await pool.run_once()
    job = store.get_job(t.trace_id)
    assert job["status"] == "failed" and "TimeoutError" in job["last_error"]


def test_claims_are_exclusive() -> None:
    store = Store(":memory:")
    for i in range(5):
        store.enqueue_eval(f"t{i}", "random", float(i))
    a = store.claim_jobs(3, 10.0)
    b = store.claim_jobs(3, 10.0)
    assert {j["trace_id"] for j in a}.isdisjoint({j["trace_id"] for j in b})
    assert len(a) + len(b) == 5


def test_stale_running_jobs_are_requeued() -> None:
    store = Store(":memory:")
    store.enqueue_eval("t1", "random", 0.0)
    store.claim_jobs(1, 0.0)
    assert store.requeue_stale_jobs(older_than_ts=100.0) == 1
    assert store.get_job("t1")["status"] == "pending"


def test_heuristic_failure_opens_review_item() -> None:
    store = Store(":memory:")
    t = make_trace(tool_calls=[], output="£1.00")
    store.insert_trace(t)
    _pipeline(store).ingest(t)
    assert store.review_items()[0]["reason"].startswith("heuristic:required_tool_called")
    assert run_heuristics(t, 4000)
