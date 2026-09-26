from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from ragate.judge import prompts
from ragate.judge.base import JudgeError, JudgeOutput, Verdict
from ragate.judge.cache import CachedJudge, JudgeCache
from ragate.judge.heuristic import HeuristicJudge
from ragate.judge.llm_judge import LLMJudge
from ragate.judge.prompts import REGISTRY, JudgePrompt

CTX = "[travel] Hotel rates are capped at 180 GBP per night in London."


class ScriptedModel:
    """Stands in for a chat model: with_structured_output replays scripted results."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls = 0

    def with_structured_output(self, schema):  # type: ignore[no-untyped-def]
        def run(_prompt: str):  # type: ignore[no-untyped-def]
            self.calls += 1
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        return RunnableLambda(run)


def test_heuristic_faithfulness_separates_supported_from_invented() -> None:
    judge = HeuristicJudge()
    good = judge.evaluate(REGISTRY["faithfulness"], {
        "context": CTX, "answer": "Hotel rates are capped at 180 GBP per night in London."})
    bad = judge.evaluate(REGISTRY["faithfulness"], {
        "context": CTX, "answer": "Hotels are free for directors on weekends."})
    assert good.score == 1.0 and bad.score == 0.0


def test_heuristic_jitter_is_seeded_and_bounded() -> None:
    v = {"context": CTX, "answer": "Hotel rates are capped at 180 GBP."}
    a = HeuristicJudge(jitter=0.2, seed=1).evaluate(REGISTRY["faithfulness"], v)
    b = HeuristicJudge(jitter=0.2, seed=1).evaluate(REGISTRY["faithfulness"], v)
    assert a == b and 0.0 <= a.score <= 1.0


def test_unknown_prompt_is_a_judge_error() -> None:
    with pytest.raises(JudgeError):
        HeuristicJudge().evaluate(JudgePrompt(name="x", version="1", template=""), {})


def test_cache_serves_repeat_calls_without_calling_the_judge(tmp_path: Path) -> None:
    model = ScriptedModel([JudgeOutput(reason="ok", score=8)])
    judge = CachedJudge(LLMJudge(model, "m", 0.0), JudgeCache(tmp_path / "c.db"))  # type: ignore[arg-type]
    v = {"context": CTX, "answer": "180 GBP"}
    first = judge.evaluate(REGISTRY["faithfulness"], v)
    second = judge.evaluate(REGISTRY["faithfulness"], v)
    assert first.score == 0.8 and not first.cached
    assert second.cached and second.score == 0.8 and model.calls == 1


def test_cache_key_changes_with_prompt_version_and_model() -> None:
    p = REGISTRY["faithfulness"]
    bumped = p.model_copy(update={"version": "1.0.1"})
    v = {"a": "b"}
    assert JudgeCache.key("m", p, v) != JudgeCache.key("m", bumped, v)
    assert JudgeCache.key("m", p, v) != JudgeCache.key("other", p, v)


def test_llm_judge_retries_invalid_output_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragate.retry.time.sleep", lambda _: None)
    model = ScriptedModel([ValueError("not json"), {"reason": "fine", "score": 10}])
    judge = LLMJudge(model, "m", 0.0, attempts=2)  # type: ignore[arg-type]
    assert judge.evaluate(REGISTRY["answer_relevancy"], {"question": "q", "answer": "a"}) == \
        Verdict(score=1.0, reason="fine")


def test_llm_judge_raises_judge_error_when_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragate.retry.time.sleep", lambda _: None)
    model = ScriptedModel([TimeoutError("slow")] * 3)
    judge = LLMJudge(model, "m", 0.0, attempts=3)  # type: ignore[arg-type]
    with pytest.raises(JudgeError, match="slow"):
        judge.evaluate(REGISTRY["answer_relevancy"], {"question": "q", "answer": "a"})


def test_judge_model_id_includes_temperature() -> None:
    assert LLMJudge(ScriptedModel([]), "gpt-4o-mini", 0.0).model_id == "gpt-4o-mini@t0"  # type: ignore[arg-type]


def test_prompt_lock_is_up_to_date() -> None:
    assert prompts.lock_violations() == []


def test_editing_a_prompt_without_a_version_bump_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    edited = REGISTRY["faithfulness"].model_copy(update={"template": "Be lenient. {answer}"})
    monkeypatch.setitem(REGISTRY, "faithfulness", edited)
    assert any("faithfulness" in p for p in prompts.lock_violations())
    bumped = edited.model_copy(update={"version": "2.0.0"})
    monkeypatch.setitem(REGISTRY, "faithfulness", bumped)
    assert prompts.lock_violations() == []
