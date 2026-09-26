from __future__ import annotations

import json

import pytest

from modelsel.config import Settings
from modelsel.dataset import CANARY, PrivateSplitLocked, dataset_hash, load_human_labels, load_split, split_path
from modelsel.schemas import Label
from modelsel.tasks import parse_fields, parse_label


def test_splits_have_expected_sizes_and_are_disjoint(settings: Settings) -> None:
    dev = load_split(settings.data_dir, "dev")
    test = load_split(settings.data_dir, "test")
    private = load_split(settings.data_dir, "private", allow_private=True)
    assert (len(dev), len(test), len(private)) == (40, 80, 30)
    ids = [it.id for it in dev + test + private]
    assert len(ids) == len(set(ids)) == 150
    assert {it.label for it in test} == set(Label), "stratification must cover every label in test"


def test_every_split_file_carries_the_canary(settings: Settings) -> None:
    for split in ("dev", "test", "private"):
        first = json.loads(split_path(settings.data_dir, split).read_text().splitlines()[0])
        assert first["_meta"]["canary"] == CANARY


def test_private_split_is_locked_by_default(settings: Settings) -> None:
    with pytest.raises(PrivateSplitLocked):
        load_split(settings.data_dir, "private")


def test_dataset_hash_is_stable_and_content_sensitive(settings: Settings) -> None:
    h1 = dataset_hash(settings.data_dir, ["test"])
    assert h1 == dataset_hash(settings.data_dir, ["test"])
    path = split_path(settings.data_dir, "test")
    path.write_text(path.read_text() + "\n")
    assert dataset_hash(settings.data_dir, ["test"]) != h1


def test_human_labels_never_use_private_items(settings: Settings) -> None:
    rows = load_human_labels(settings.data_dir)
    assert len(rows) == 40
    assert not any(r.item_id.startswith("private") for r in rows)
    assert all(len(r.ratings_a) == 3 and all(1 <= x <= 5 for x in r.ratings_a) for r in rows)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("refund", "refund"), ("Refund.", "refund"), ("  BILLING\n", "billing"), ("label: shipping", "shipping"),
     ("money back", "invalid"), ("", "invalid")],
)
def test_parse_label(text: str, expected: str) -> None:
    assert parse_label(text) == expected


def test_parse_fields_accepts_valid_and_fenced_json() -> None:
    raw = '{"order_id": "ORD-12345", "product": "Cobalt Blender", "amount": 12.5, "priority": "high", "sentiment": "negative"}'
    fields, err = parse_fields(raw)
    assert err is None and fields is not None and fields.order_id == "ORD-12345"
    fenced, err2 = parse_fields(f"```json\n{raw}\n```")
    assert err2 is None and fenced == fields


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ('Sure! {"order_id": null', "json_decode"),
        ("[1, 2]", "json_not_object"),
        ('{"order_id": null, "product": null, "amount": null, "priority": "normal", "sentiment": "neutral"}', "schema"),
        ('{"order_id": "12345", "product": null, "amount": null, "priority": "low", "sentiment": "neutral"}', "schema"),
        (
            '{"order_id": null, "product": null, "amount": null, "priority": "low", "sentiment": "neutral", "x": 1}',
            "schema",
        ),
    ],
)
def test_parse_fields_rejects_invalid_output(raw: str, reason: str) -> None:
    fields, err = parse_fields(raw)
    assert fields is None and err is not None and err.startswith(reason)
