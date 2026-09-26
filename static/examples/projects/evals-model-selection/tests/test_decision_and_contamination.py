from __future__ import annotations

from modelsel.config import Settings
from modelsel.contamination import completion_probe, overlap
from modelsel.dataset import load_split
from modelsel.decision import ModelSummary, apply_gates, dominates, pareto_frontier, recommend, weighted_matrix
from modelsel.harness.client import LLMClient
from modelsel.llm.fakes import FakeTicketModel
from modelsel.llm.registry import DecisionConfig, FakeProfile


def _m(mid: str, q: float, cost: float, p95: float, json_ok: float = 1.0) -> ModelSummary:
    return ModelSummary(
        model_id=mid, quality=q, quality_low=q - 0.02, quality_high=q + 0.02, accuracy=q, macro_f1=q,
        json_validity=json_ok, field_accuracy=q, reply_score=4.0, p50_latency_ms=p95 / 2, p95_latency_ms=p95,
        cost_per_1k_usd=cost,
    )


def test_dominance_and_frontier() -> None:
    a, b, c = _m("a", 0.9, 2.0, 2000), _m("b", 0.8, 0.2, 1000), _m("c", 0.79, 0.3, 1200)
    assert dominates(b, c) and not dominates(a, b) and not dominates(b, a)
    assert pareto_frontier([a, b, c]) == ["a", "b"]


def test_gates_block_and_blocked_reasons() -> None:
    cfg = DecisionConfig(min_json_validity=0.95, max_p95_latency_ms=3000, max_cost_per_1k_tickets_usd=1.0)
    models = [_m("ok", 0.8, 0.5, 1000), _m("bad_json", 0.9, 0.5, 1000, json_ok=0.9), _m("slow", 0.9, 0.5, 5000)]
    apply_gates(models, cfg, blocked={"ok": "contamination suspected"})
    assert [m.passes for m in models] == [False, False, False]
    assert "JSON validity" in models[1].gate_failures[0]
    assert "p95 latency" in models[2].gate_failures[0]


def test_weighted_matrix_normalises_and_inverts_cost() -> None:
    cfg = DecisionConfig(weights={"quality": 1.0, "cost": 1.0, "latency": 0.0})
    m = weighted_matrix([_m("a", 0.9, 2.0, 1000), _m("b", 0.8, 0.2, 1000)], cfg)
    assert m["a"]["quality"] == 1.0 and m["a"]["cost"] == 0.0
    assert m["b"]["cost"] == 1.0 and m["a"]["total"] == m["b"]["total"] == 0.5


def test_recommend_prefers_cheaper_model_when_quality_tie_is_not_significant() -> None:
    cfg = DecisionConfig(weights={"quality": 0.9, "cost": 0.05, "latency": 0.05}, max_cost_per_1k_tickets_usd=10)
    big, small = _m("big", 0.90, 3.0, 2000), _m("small", 0.89, 0.2, 1000)
    rec = recommend([big, small], cfg, p_values_vs={("big", "small"): 0.4})
    assert rec.model_id == "small" and rec.significant is False and rec.caveats
    rec2 = recommend([_m("big", 0.95, 3.0, 2000), _m("small", 0.85, 0.2, 1000)], cfg, p_values_vs={("big", "small"): 0.001})
    assert rec2.model_id == "big" and rec2.significant is True


def test_recommend_with_no_eligible_model() -> None:
    cfg = DecisionConfig(min_json_validity=0.99)
    rec = recommend([_m("a", 0.9, 0.1, 100, json_ok=0.5)], cfg, p_values_vs={})
    assert rec.model_id is None


def test_overlap() -> None:
    truth = "one two three four five six seven eight nine ten"
    assert overlap(truth, truth) == 1.0
    assert overlap(truth, "completely different words here") == 0.0


async def test_probe_flags_memorised_items_only(client: LLMClient, settings: Settings) -> None:
    test = load_split(settings.data_dir, "test")[:10]
    private = load_split(settings.data_dir, "private", allow_private=True)[:10]
    leaky = FakeTicketModel(model_id="fake:leaky-tuned", profile=FakeProfile(), memorised={it.ticket: it for it in test})
    client.register_model("fake:leaky-tuned", leaky)
    on_test = await completion_probe(client, "fake:leaky-tuned", test)
    on_private = await completion_probe(client, "fake:leaky-tuned", private)
    assert on_test.flagged and on_test.mean_overlap > 0.9
    assert not on_private.flagged and on_private.mean_overlap < 0.2
