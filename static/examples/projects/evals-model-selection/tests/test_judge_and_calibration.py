from __future__ import annotations

import math

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from modelsel.config import Settings
from modelsel.dataset import load_human_labels
from modelsel.harness.client import LLMClient
from modelsel.judge.calibration import calibrate, cohen_kappa, fleiss_kappa, pad_reply, spearman
from modelsel.judge.judge import Judge, probability_weighted_score
from modelsel.llm.fakes import FakeJudgeModel
from modelsel.llm.registry import Catalogue, JudgeProfile


def _lp(dist: dict[str, float]) -> dict:
    top = [{"token": t, "logprob": math.log(p)} for t, p in dist.items()]
    return {"content": [{"token": "Score"}, {"token": ":"}, {"token": " 4", "top_logprobs": top}]}


def test_probability_weighted_score() -> None:
    assert probability_weighted_score(_lp({"4": 0.5, "5": 0.5})) == pytest.approx(4.5)
    # non-score alternatives are ignored and the rest renormalised
    assert probability_weighted_score(_lp({"3": 0.3, "4": 0.3, "x": 0.4})) == pytest.approx(3.5)
    assert probability_weighted_score(None) is None
    assert probability_weighted_score({"content": [{"token": "hello"}]}) is None


async def test_pointwise_uses_logprobs_from_fake_judge(client: LLMClient, settings: Settings) -> None:
    row = load_human_labels(settings.data_dir)[0]
    res = await Judge(client, "fake:judge-large").pointwise(row.ticket, row.reply_a, row.reference_reply)
    assert res.weighted and res.score is not None and 1 <= res.score <= 5


async def test_pointwise_falls_back_to_parsed_score(client: LLMClient) -> None:
    client.register_model("fake:scripted", GenericFakeChatModel(messages=iter([AIMessage(content="checks...\nScore: 2")])))
    res = await Judge(client, "fake:scripted").pointwise("t", "r", None)
    assert (res.score, res.weighted) == (2.0, False)


async def test_pairwise_swap_maps_second_verdict_back(client: LLMClient) -> None:
    # The judge always says "A": first order A wins, swapped order B (= original A's rival) wins.
    msgs = iter([AIMessage(content="Verdict: A"), AIMessage(content="Verdict: A")])
    client.register_model("fake:scripted", GenericFakeChatModel(messages=msgs))
    res = await Judge(client, "fake:scripted").pairwise("t", "reply one", "reply two", None)
    assert res.verdicts == ("A", "B")
    assert not res.consistent and res.score_a == 0.5, "pure position bias must cancel out"


async def test_unparseable_verdict_counts_as_tie(client: LLMClient) -> None:
    client.register_model("fake:scripted", GenericFakeChatModel(messages=iter([AIMessage(content="I cannot decide")])))
    res = await Judge(client, "fake:scripted").pairwise("t", "a", "b", None, swap=False)
    assert res.score_a == 0.5


def test_agreement_statistics() -> None:
    assert cohen_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert cohen_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"]) == pytest.approx(0.0)
    # weighted kappa gives partial credit to near misses
    near = cohen_kappa([1, 2, 3, 4, 5], [2, 3, 4, 5, 5], labels=[1, 2, 3, 4, 5], weights="quadratic")
    unweighted = cohen_kappa([1, 2, 3, 4, 5], [2, 3, 4, 5, 5], labels=[1, 2, 3, 4, 5])
    assert near > unweighted
    assert fleiss_kappa([[1, 1, 1], [2, 2, 2], [3, 3, 3]], [1, 2, 3]) == pytest.approx(1.0)
    assert fleiss_kappa([[1, 2, 3], [2, 3, 1], [3, 1, 2]], [1, 2, 3]) < 0
    with pytest.raises(ValueError):
        fleiss_kappa([[1, 2], [1]], [1, 2])
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_pad_reply_keeps_signoff_last() -> None:
    padded = pad_reply("Hi Tom,\n\nBody.\n\nKind regards,\nSupport Team")
    assert padded.endswith("Kind regards,\nSupport Team") and len(padded.split()) > 60


def test_fake_judge_biases_are_what_calibration_measures() -> None:
    reply = "Hi Tom,\n\nThanks for getting in touch about your Cobalt Blender.\n\nKind regards,\nSupport Team"
    ref = "Hi Tom,\n\nThanks for getting in touch about your Cobalt Blender. I have issued a full refund."
    plain = FakeJudgeModel(model_id="j", profile=JudgeProfile(noise=0))
    wordy = FakeJudgeModel(model_id="j", profile=JudgeProfile(noise=0, verbosity_bias=1.0))
    assert wordy.quality(pad_reply(reply), "t", ref, True) > plain.quality(pad_reply(reply), "t", ref, True)
    # without anchors, scores are compressed towards the middle
    assert abs(plain.quality(reply, "t", ref, False) - 3) < abs(plain.quality(reply, "t", ref, True) - 3)


async def test_calibration_detects_position_and_verbosity_bias(client: LLMClient, catalogue: Catalogue, settings: Settings) -> None:
    rows = load_human_labels(settings.data_dir)[:20]
    report = await calibrate(client, catalogue, "fake:judge-large", "fake:meta-judge", rows, spot_checks=3)
    assert report.spearman > 0.6, "the default judge should track humans"
    assert report.position_consistency < 0.9
    assert report.verbosity_delta > 0.1 and report.verbosity_p < 0.05
    assert report.ablation["anchored+reference (default)"] > report.ablation["bare prompt"]
    assert report.pairwise_kappa_swapped >= report.pairwise_kappa_single
    assert len(report.meta_checks) == 3 and report.meta_agreement is not None
    assert any("Verbosity bias" in c for c in report.caveats)
