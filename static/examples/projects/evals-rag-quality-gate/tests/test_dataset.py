from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragate.dataset import checks, review, store
from ragate.dataset.store import DatasetError
from ragate.dataset.synth import synthesise
from ragate.fakes import FakeSynthChatModel
from ragate.models import Evidence, GoldenItem, QuestionType, ReviewStatus
from ragate.rag.chunking import chunk_corpus
from ragate.rag.corpus import load_corpus
from ragate.rag.generator import FEW_SHOT_QUESTIONS
from tests.conftest import ROOT

DOCS = load_corpus(ROOT / "data" / "corpus")
GOLDEN = ROOT / "data" / "golden"


def _item(item_id: str, question: str, **kw) -> GoldenItem:  # type: ignore[no-untyped-def]
    base = dict(question_type=QuestionType.FACTOID, reference_answer="Laptops: every 3 years.",
                evidence=[Evidence(doc_id="equipment-it", quote="Laptops are refreshed every 3 years.")])
    return GoldenItem(item_id=item_id, question=question, **{**base, **kw})


def test_v1_is_frozen_stratified_and_clean() -> None:
    manifest, items = store.load(GOLDEN, "v1")
    assert len(items) == 40 and manifest.counts_by_type == {
        "factoid": 16, "multi_hop": 8, "unanswerable": 8, "adversarial": 8}
    assert checks.run_checks(items, DOCS) == []


def test_modifying_a_frozen_version_is_detected(tmp_path: Path) -> None:
    _, items = store.load(GOLDEN, "v1")
    store.freeze(tmp_path, "v1", items[:30], corpus_sha="x")
    with (tmp_path / "v1" / "golden.jsonl").open("a") as fh:
        fh.write(items[35].model_dump_json() + "\n")
    with pytest.raises(DatasetError, match="modified after freezing"):
        store.load(tmp_path, "v1")
    with pytest.raises(DatasetError, match="already frozen"):
        store.freeze(tmp_path, "v1", items, corpus_sha="x")


def test_latest_version_orders_numerically(tmp_path: Path) -> None:
    _, items = store.load(GOLDEN, "v1")
    for v in ("v2", "v10"):
        store.freeze(tmp_path, v, items, corpus_sha="x")
    assert store.latest_version(tmp_path) == "v10"


def _checks_for(items: list[GoldenItem]) -> set[str]:
    return {f.check for f in checks.run_checks(items, DOCS, min_per_type=0)}


def test_stale_evidence_is_an_error() -> None:
    stale = _item("a", "How often do laptops get replaced?",
                  evidence=[Evidence(doc_id="equipment-it", quote="Laptops are refreshed yearly.")])
    assert "evidence" in _checks_for([stale])


def test_few_shot_contamination_is_an_error() -> None:
    assert "prompt_contamination" in _checks_for([_item("a", FEW_SHOT_QUESTIONS[0])])


def test_answer_leakage_and_near_duplicates() -> None:
    leak = _item("a", "Laptops are refreshed every 3 years, right?")
    dup1 = _item("b", "How often are the company laptops refreshed?")
    dup2 = _item("c", "How often are the company laptops refreshed ?")
    found = _checks_for([leak, dup1, dup2])
    assert {"answer_leakage", "near_duplicate"} <= found


def test_verbatim_copy_of_corpus_is_a_warning() -> None:
    copied = _item("a", "If a laptop is lost or stolen, report it to the IT service desk within 24 hours?")
    findings = checks.run_checks([copied], DOCS, min_per_type=0)
    assert any(f.check == "verbatim_leakage" and f.level == "warning" for f in findings)


def test_stratification_minimum() -> None:
    assert "stratification" in {f.check for f in checks.run_checks([_item("a", "q laptops?")], DOCS)}


def test_restricted_evidence_is_rejected() -> None:
    item = _item("a", "Who is head of people?", reference_answer="Priya Raman.",
                 evidence=[Evidence(doc_id="hr-contacts",
                                    quote="The Head of People Operations is Priya Raman.")])
    assert "evidence" in _checks_for([item])


def test_review_csv_roundtrip_and_reviewer_required(tmp_path: Path) -> None:
    items = [_item("a", "How often are laptops refreshed?"),
             _item("b", "Laptop swap cadence?", review_status=ReviewStatus.PENDING)]
    path = tmp_path / "r.csv"
    review.export_csv(items, path)
    back = review.import_csv(path)
    assert back[0].evidence == items[0].evidence and back[1].review_status == "pending"
    with pytest.raises(ValueError, match="without a reviewer"):
        review.approved(back)
    back[0].reviewer = "sme"
    assert [i.item_id for i in review.approved(back)] == ["a"]


def test_synthesis_is_stratified_pending_and_grounded() -> None:
    chunks = chunk_corpus(DOCS, 400, 80)
    items = synthesise(FakeSynthChatModel(), chunks, {qt: 2 for qt in QuestionType}, seed=1)
    assert {i.question_type for i in items} == set(QuestionType)
    assert all(i.review_status == ReviewStatus.PENDING and i.source == "synthetic" for i in items)
    by_id = {c.doc_id: c for c in chunks}
    for i in items:
        for e in i.evidence:
            assert e.doc_id in by_id


def test_synthesis_drops_unparseable_proposals() -> None:
    chunks = chunk_corpus(DOCS, 400, 80)
    model = FakeListChatModel(responses=["not json at all"] * 10)
    assert synthesise(model, chunks, {QuestionType.FACTOID: 2}) == []


def test_synthesis_drops_quotes_not_in_the_passage() -> None:
    chunks = chunk_corpus(DOCS, 400, 80)
    bogus = '{"question": "Invented question?", "answer": "x", ' \
            '"evidence": [{"doc_id": "travel", "quote": "made up text"}]}'
    model = FakeListChatModel(responses=[bogus] * 10)
    assert synthesise(model, chunks, {QuestionType.FACTOID: 1}) == []
