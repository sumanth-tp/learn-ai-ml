"""Versioned, immutable golden datasets.

Layout::

    data/golden/v1/golden.jsonl    one GoldenItem per line
    data/golden/v1/manifest.json   version, parent, sha256, counts, corpus hash

A version is frozen once written: `verify` recomputes the hash, and any edit to a
frozen file is an error. Changing the dataset means creating v2, because a run is only
comparable with another run on the *same* dataset version.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from ragate.models import GoldenItem, ReviewStatus


class DatasetError(ValueError):
    pass


class Manifest(BaseModel):
    version: str
    parent: str | None = None
    created_at: str
    sha256: str
    corpus_sha: str
    counts_by_type: dict[str, int]
    counts_by_status: dict[str, int]
    generator: dict[str, str] = Field(default_factory=dict)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[GoldenItem]:
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                items.append(GoldenItem.model_validate_json(line))
            except ValueError as exc:
                raise DatasetError(f"{path}:{n}: {exc}") from exc
    return items


def write_jsonl(path: Path, items: list[GoldenItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")


def freeze(
    root: Path,
    version: str,
    items: list[GoldenItem],
    *,
    corpus_sha: str,
    parent: str | None = None,
    generator: dict[str, str] | None = None,
) -> Manifest:
    directory = root / version
    if (directory / "manifest.json").exists():
        raise DatasetError(f"dataset {version} is already frozen; create a new version")
    ids = [i.item_id for i in items]
    if len(ids) != len(set(ids)):
        raise DatasetError("duplicate item_id values")
    write_jsonl(directory / "golden.jsonl", items)
    manifest = Manifest(
        version=version,
        parent=parent,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        sha256=_sha(directory / "golden.jsonl"),
        corpus_sha=corpus_sha,
        counts_by_type=dict(Counter(i.question_type.value for i in items)),
        counts_by_status=dict(Counter(i.review_status.value for i in items)),
        generator=generator or {},
    )
    (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def load(
    root: Path, version: str, *, approved_only: bool = True
) -> tuple[Manifest, list[GoldenItem]]:
    directory = root / version
    try:
        manifest = Manifest.model_validate_json((directory / "manifest.json").read_text())
    except FileNotFoundError as exc:
        raise DatasetError(f"dataset {version} not found under {root}") from exc
    actual = _sha(directory / "golden.jsonl")
    if actual != manifest.sha256:
        raise DatasetError(
            f"dataset {version} was modified after freezing (sha {actual[:12]} != "
            f"{manifest.sha256[:12]}); frozen versions are immutable"
        )
    items = read_jsonl(directory / "golden.jsonl")
    if approved_only:
        items = [i for i in items if i.review_status == ReviewStatus.APPROVED]
    return manifest, items


def latest_version(root: Path) -> str:
    versions = sorted(
        (p.name for p in root.iterdir() if (p / "manifest.json").exists()),
        key=lambda v: int(v.lstrip("v") or 0),
    )
    if not versions:
        raise DatasetError(f"no frozen dataset versions under {root}")
    return versions[-1]
