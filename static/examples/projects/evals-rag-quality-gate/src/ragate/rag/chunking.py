"""Sentence-aware chunking with overlap."""

from __future__ import annotations

from ragate.models import Chunk, Document
from ragate.pii import redact
from ragate.text import sentences


def chunk_document(
    doc: Document, chunk_size: int, chunk_overlap: int, *, pii_redaction: bool = True
) -> list[Chunk]:
    """Pack whole sentences into chunks of at most ``chunk_size`` characters.

    Never splits a sentence (a split sentence is the commonest cause of a policy
    number losing its subject). Overlap carries trailing sentences forward.
    """
    text = redact(doc.text) if pii_redaction else doc.text
    sents = sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks):02d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=" ".join(current),
                    position=len(chunks),
                )
            )

    for sent in sents:
        if current and len(" ".join([*current, sent])) > chunk_size:
            flush()
            carried: list[str] = []
            for prev in reversed(current):
                if len(" ".join([prev, *carried])) > chunk_overlap:
                    break
                carried.insert(0, prev)
            current = carried
        current.append(sent)
    flush()
    return chunks


def chunk_corpus(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    *,
    include_restricted: bool = False,
    pii_redaction: bool = True,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        if doc.access == "restricted" and not include_restricted:
            continue
        out.extend(chunk_document(doc, chunk_size, chunk_overlap, pii_redaction=pii_redaction))
    return out
