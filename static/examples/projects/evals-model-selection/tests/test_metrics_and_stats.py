from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from modelsel.metrics.classification import exact_match, macro_f1, per_class_f1
from modelsel.metrics.extraction import field_accuracy, field_matches
from modelsel.metrics.operational import cost_per_1k, percentile
from modelsel.schemas import Priority, Sentiment, TicketFields
from modelsel.stats import (
    bootstrap_ci,
    holm,
    mcnemar,
    minimum_detectable_effect,
    n_paired_means,
    n_paired_proportions,
    paired_bootstrap,
    permutation_test,
)


def test_exact_match_and_macro_f1_known_values() -> None:
    gold = ["a", "a", "a", "b"]
    pred = ["a", "a", "a", "a"]
    assert exact_match(gold, pred) == 0.75
    f1 = per_class_f1(gold, pred)
    assert f1["a"] == pytest.approx(2 * 0.75 * 1 / 1.75)
    assert f1["b"] == 0.0
    # accuracy 75% hides that class b is never found; macro-F1 does not
    assert macro_f1(gold, pred) == pytest.approx((f1["a"] + 0) / 2)


def test_macro_f1_invented_label_costs_precision_only() -> None:
    assert macro_f1(["a", "b"], ["a", "invalid"]) == pytest.approx((1.0 + 0.0) / 2)


def test_field_accuracy_normalises_and_zeroes_invalid() -> None:
    gold = TicketFields(order_id="ORD-00001", product="Cobalt Blender", amount=10.0, priority=Priority.LOW, sentiment=Sentiment.NEUTRAL)
    pred = TicketFields(order_id="ORD-00001", product="cobalt  blender", amount=10.001, priority=Priority.HIGH, sentiment=Sentiment.NEUTRAL)
    matches = field_matches(gold, pred)
    assert matches == {"order_id": True, "product": True, "amount": True, "priority": False, "sentiment": True}
    assert field_accuracy(gold, pred) == 0.8
    assert field_accuracy(gold, None) == 0.0


def test_operational_metrics() -> None:
    assert percentile([100, 200, 300, 400, 1000], 50) == 300
    assert cost_per_1k([0.001, 0.003]) == pytest.approx(2.0)


def test_bootstrap_ci_brackets_the_mean() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.7, 0.1, 200)
    ci = bootstrap_ci(x, resamples=500)
    assert ci.low < x.mean() < ci.high
    assert ci.high - ci.low < 0.05


def test_paired_tests_detect_real_difference_and_not_noise() -> None:
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 1, 100)
    better = base + 0.1 + rng.normal(0, 0.05, 100)
    same = base + rng.normal(0, 0.05, 100)
    assert paired_bootstrap(better, base, resamples=500).p_value < 0.01
    assert permutation_test(better, base, resamples=500).p_value < 0.01
    assert permutation_test(same, base, resamples=500).p_value > 0.05


def test_mcnemar_exact_matches_binomial() -> None:
    a = [True] * 10 + [False] * 2 + [True] * 50
    b = [False] * 10 + [True] * 2 + [True] * 50
    r = mcnemar(a, b)
    assert (r.only_a_correct, r.only_b_correct, r.exact) == (10, 2, True)
    assert r.p_value == pytest.approx(sps.binomtest(2, 12, 0.5).pvalue)
    assert mcnemar([True, False], [True, False]).p_value == 1.0


def test_mcnemar_uses_chi_square_for_many_discordant_pairs() -> None:
    a = [True] * 30 + [False] * 10
    b = [False] * 30 + [True] * 10
    r = mcnemar(a, b)
    assert not r.exact
    assert r.statistic == pytest.approx((abs(30 - 10) - 1) ** 2 / 40)


def test_holm_is_stricter_than_raw_alpha() -> None:
    res = holm({"x": 0.01, "y": 0.03, "z": 0.04})
    assert res == {"x": True, "y": False, "z": False}


def test_sample_size_formulas() -> None:
    expected = math.ceil(((sps.norm.ppf(0.975) + sps.norm.ppf(0.8)) * 0.5 / 0.1) ** 2)
    assert n_paired_means(0.1, 0.5) == expected == 197
    # MDE at that n is (just under) the target effect: the two functions are inverses
    assert minimum_detectable_effect(197, 0.5) == pytest.approx(0.1, rel=0.01)
    assert n_paired_proportions(0.05, 0.2) > n_paired_proportions(0.05, 0.1)
    with pytest.raises(ValueError):
        n_paired_proportions(0.3, 0.1)
