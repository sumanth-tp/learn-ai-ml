import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agentmon.agent.prompts import CANARY
from agentmon.evals.heuristics import run_heuristics
from agentmon.evals.judges import JudgeError, LLMJudge, build_judges
from agentmon.evals.pipeline import CascadingEvaluator
from agentmon.llm.fake_judge import RuleJudgeModel, judge_rules
from agentmon.models import EvalResult, ToolCallRecord, TraceRecord
from agentmon.store import Store


def make_trace(**kw) -> TraceRecord:
    base = dict(trace_id="t1", request_id="r1", ts=1_790_000_000.0, user_id="CUST-1",
                session_id="s", input="What's the balance on ACC-1001?",
                output="The available balance on ACC-1001 (current) is £4,210.55.",
                intent="balance", prompt_version="v1", model="m", status="ok", latency_ms=900,
                tool_calls=[ToolCallRecord(name="get_balance", args={"account_id": "ACC-1001"},
                                           output='{"balance": 4210.55}')])
    base.update(kw)
    return TraceRecord(**base)


def test_fake_judge_rules() -> None:
    assert judge_rules("groundedness", "", '{"balance": 4210.55}', "It is £4,210.55.")[0] == 5
    assert judge_rules("groundedness", "", "(no tools were called)", "It is £901.13.")[0] == 1
    assert judge_rules("policy", "", "", "You should buy Bitcoin")[0] == 1
    assert judge_rules("policy", "", "", f"ref {CANARY}")[0] == 1
    assert judge_rules("helpfulness", "", "", "Your recent transactions all look normal")[0] == 2


async def test_llm_judge_through_the_real_prompt_and_parser() -> None:
    judge = LLMJudge("groundedness", RuleJudgeModel(), cost_fn=lambda i, o: (i + o) * 1e-6)
    good = await judge.aevaluate(make_trace())
    assert good.passed and good.score == 1.0 and good.tier == "judge" and good.cost_usd > 0
    bad = await judge.aevaluate(make_trace(tool_calls=[], output="It is £901.13."))
    assert not bad.passed and bad.score == 0.0


async def test_llm_judge_parses_a_scripted_provider_reply() -> None:
    model = FakeListChatModel(responses=['{"reasoning": "fine", "score": 4}'])
    result = await LLMJudge("helpfulness", model).aevaluate(make_trace())
    assert result.passed and result.score == 0.75


async def test_unparsable_judge_output_raises_judge_error() -> None:
    model = FakeListChatModel(responses=["I think it is great!"])
    with pytest.raises(JudgeError):
        await LLMJudge("policy", model).aevaluate(make_trace())


def test_unknown_criterion_rejected() -> None:
    with pytest.raises(ValueError):
        LLMJudge("vibes", RuleJudgeModel())


def test_heuristics_on_a_hallucinated_balance() -> None:
    t = make_trace(tool_calls=[], output="The balance on ACC-1001 is £901.13.")
    failed = {r.evaluator for r in run_heuristics(t, 4000) if not r.passed}
    assert failed == {"required_tool_called"}


def test_heuristics_skip_required_tool_for_clarifying_questions() -> None:
    t = make_trace(tool_calls=[], output="Which account? Please give me the account id.")
    assert all(r.passed for r in run_heuristics(t, 4000))


async def test_cascade_skips_judges_when_heuristics_decided() -> None:
    store = Store(":memory:")
    t = make_trace(tool_calls=[], output="£1.00")
    store.upsert_evals(run_heuristics(t, 4000))
    outcome = await CascadingEvaluator(store, build_judges(RuleJudgeModel()), 1.0).evaluate(t)
    assert outcome.status == "decided_by_heuristics"
    assert [r.evaluator for r in outcome.results] == ["cascade"]


async def test_cascade_runs_all_judges_and_respects_budget() -> None:
    store = Store(":memory:")
    t = make_trace()
    store.upsert_evals(run_heuristics(t, 4000))
    cascade = CascadingEvaluator(store, build_judges(RuleJudgeModel()), budget_usd_per_day=0.01)
    outcome = await cascade.evaluate(t)
    assert outcome.status == "done"
    assert {r.evaluator for r in outcome.results} == {
        "judge_groundedness", "judge_helpfulness", "judge_policy"}
    store.upsert_evals([EvalResult(trace_id="other", evaluator="judge_policy", score=1,
                                   passed=True, tier="judge", cost_usd=0.02, ts=t.ts)])
    assert (await cascade.evaluate(t)).status == "skipped_budget"


async def test_cascade_does_not_judge_guardrail_refusals() -> None:
    store = Store(":memory:")
    t = make_trace(status="blocked", tool_calls=[], output="I'm sorry, I can't help with that.")
    store.upsert_evals(run_heuristics(t, 4000))
    outcome = await CascadingEvaluator(store, build_judges(RuleJudgeModel()), 1.0).evaluate(t)
    assert outcome.status == "skipped_blocked"
