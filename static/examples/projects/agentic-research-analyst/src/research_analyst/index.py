"""Internal document index: ingest markdown -> chunks -> vectors, persisted to JSON.

Each internal markdown file starts with a small header block::

    title: Sodium-ion pilot memo
    published: 2025-11-04
    ---
    body text...
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from research_analyst.models import Origin, Source
from research_analyst.quality import quality_score, source_id_for

log = logging.getLogger(__name__)

CHUNK_CHARS = 900


def _parse(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_text(encoding="utf-8")
    header, sep, body = raw.partition("\n---\n")
    if not sep:
        return {"title": path.stem.replace("-", " ").title()}, raw
    meta = {}
    for line in header.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
    return meta, body.strip()


def chunk(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Paragraph-aware chunking: never split a paragraph unless it alone exceeds the limit."""
    chunks: list[str] = []
    current = ""
    for para in (p.strip() for p in text.split("\n\n") if p.strip()):
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
            continue
        if current:
            chunks.append(current)
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars) + 1 or max_chars
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        current = para
    if current:
        chunks.append(current)
    return chunks


def load_documents(folder: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(folder.glob("*.md")):
        meta, body = _parse(path)
        for i, text in enumerate(chunk(body)):
            url = f"internal://{path.stem}/c{i}"
            docs.append(Document(
                page_content=text,
                id=source_id_for(url),
                metadata={"url": url, "title": meta.get("title", path.stem),
                          "published": meta.get("published", ""), "file": path.name},
            ))
    return docs


class InternalIndex:
    def __init__(self, store: InMemoryVectorStore) -> None:
        self._store = store

    @classmethod
    def build(cls, folder: Path, embeddings: Embeddings) -> InternalIndex:
        docs = load_documents(folder)
        if not docs:
            raise ValueError(f"no markdown documents found in {folder}")
        store = InMemoryVectorStore(embeddings)
        store.add_documents(docs, ids=[d.id for d in docs])
        log.info("index built", extra={"chunks": len(docs), "folder": str(folder)})
        return cls(store)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._store.dump(str(path))

    @classmethod
    def load_or_build(cls, path: Path, folder: Path, embeddings: Embeddings) -> InternalIndex:
        if path.exists():
            return cls(InMemoryVectorStore.load(str(path), embeddings))
        index = cls.build(folder, embeddings)
        index.save(path)
        return index

    async def search(self, query: str, k: int) -> list[Source]:
        hits = await self._store.asimilarity_search_with_score(query, k=k)
        out = []
        for doc, _score in hits:
            meta = doc.metadata
            published = date.fromisoformat(meta["published"]) if meta.get("published") else None
            src = Source(id=source_id_for(meta["url"]), url=meta["url"], title=meta["title"],
                         origin=Origin.INTERNAL, content=doc.page_content, published=published)
            src.quality = quality_score(src)
            out.append(src)
        return out
