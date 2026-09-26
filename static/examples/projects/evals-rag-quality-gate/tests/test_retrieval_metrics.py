import math

import pytest

from ragate.metrics import retrieval as r
from ragate.models import Evidence
from tests.conftest import make_chunk

EV = [Evidence(doc_id="leave", quote="carry over a maximum of 5 unused days"),
      Evidence(doc_id="leave", quote="allowance increases to 28 days after 5 years")]
IRRELEVANT = make_chunk("travel", "Hotels are capped at 180 GBP.", 1)
HIT_1 = make_chunk("leave", "You may carry over a maximum of 5 unused days.", 2)
HIT_2 = make_chunk("leave", "Your allowance increases to 28 days after 5 years.", 3)


def test_relevance_needs_same_doc_and_quote_coverage() -> None:
    wrong_doc = make_chunk("other", "You may carry over a maximum of 5 unused days.", 1)
    assert r.relevance_vector([IRRELEVANT, HIT_1, wrong_doc], EV) == [False, True, False]


def test_recall_precision_hit() -> None:
    ranked = [IRRELEVANT, HIT_1, HIT_2]
    assert r.recall_at_k(ranked, EV, 2) == 0.5
    assert r.recall_at_k(ranked, EV, 3) == 1.0
    assert r.precision_at_k(ranked, EV, 3) == pytest.approx(2 / 3)
    assert r.hit_at_k(ranked, EV, 1) == 0.0


def test_mrr_and_ndcg_by_hand() -> None:
    ranked = [IRRELEVANT, HIT_1, HIT_2]
    assert r.reciprocal_rank(ranked, EV) == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert r.ndcg_at_k(ranked, EV, 3) == pytest.approx(dcg / idcg)
    assert r.ndcg_at_k([HIT_1, HIT_2], EV, 2) == pytest.approx(1.0)


def test_contextual_precision_rewards_relevant_first() -> None:
    assert r.contextual_precision([HIT_1, HIT_2, IRRELEVANT], EV, 3) == 1.0
    assert r.contextual_precision([IRRELEVANT, HIT_1, HIT_2], EV, 3) == pytest.approx(
        (1 / 2 + 2 / 3) / 2
    )
    assert r.contextual_precision([IRRELEVANT], EV, 1) == 0.0


def test_recall_without_evidence_is_an_error() -> None:
    with pytest.raises(ValueError):
        r.recall_at_k([HIT_1], [], 1)
