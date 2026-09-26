"""Load the handbook: Markdown files with a YAML front-matter block."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from ragate.models import Document


class CorpusError(ValueError):
    pass


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise CorpusError(f"{path.name}: missing front matter")
    try:
        _, front, body = raw.split("---\n", 2)
    except ValueError as exc:
        raise CorpusError(f"{path.name}: unterminated front matter") from exc
    meta = yaml.safe_load(front) or {}
    if "id" not in meta or "title" not in meta:
        raise CorpusError(f"{path.name}: front matter needs id and title")
    return Document(
        doc_id=str(meta["id"]),
        title=str(meta["title"]),
        text=body.strip(),
        owner=str(meta.get("owner", "")),
        updated=str(meta.get("updated", "")),
        access=meta.get("access", "public"),
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise CorpusError(f"no .md documents in {corpus_dir}")
    docs = [parse_document(p) for p in paths]
    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        raise CorpusError("duplicate document ids in corpus")
    return docs


def corpus_hash(docs: list[Document]) -> str:
    h = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.doc_id):
        h.update(doc.model_dump_json().encode())
    return h.hexdigest()[:16]
