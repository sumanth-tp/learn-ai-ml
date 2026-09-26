"""Company-docs RAG MCP server (streamable HTTP).

Retrieval lives in the server; generation stays in the host. The server exposes the
corpus as resources (``docs://index``, ``docs://doc/{slug}``), a ``search`` tool with
BM25 ranking over heading-level chunks, and an ``answer_with_citations`` prompt.
The corpus is untrusted content: one seeded document carries a prompt injection, which
is exactly what the host's defences are tested against.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from demo_servers.common import (
    add_health_route,
    advertise_list_changed,
    log_level,
    serve_http,
    transport_security,
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "the",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
        "do",
        "does",
        "can",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    slug: str
    title: str
    section: str
    text: str


class Hit(BaseModel):
    slug: str
    title: str
    section: str
    score: float
    text: str


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75). Enough for a few hundred chunks, zero dependencies."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.title} {c.section} {c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for chunk, tf, length in zip(self.chunks, self.docs, self.lengths, strict=True):
            score = 0.0
            for term in terms:
                if term not in tf:
                    continue
                f = tf[term]
                norm = f + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += self.idf[term] * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


def load_corpus(root: Path) -> tuple[dict[str, tuple[str, str]], list[Chunk]]:
    """Read ``*.md``; the first ``# `` line is the title, ``## `` lines start chunks."""
    docs: dict[str, tuple[str, str]] = {}
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), path.stem)
        docs[path.stem] = (title, text)
        section, buf = "Overview", []
        for line in lines:
            if line.startswith("## "):
                if "".join(buf).strip():
                    chunks.append(Chunk(path.stem, title, section, "\n".join(buf).strip()))
                section, buf = line[3:].strip(), []
            elif not line.startswith("# "):
                buf.append(line)
        if "".join(buf).strip():
            chunks.append(Chunk(path.stem, title, section, "\n".join(buf).strip()))
    return docs, chunks


def create_server(docs_dir: Path | None = None) -> FastMCP:
    root = docs_dir or Path(os.environ.get("DOCS_DIR", "data/docs"))
    docs, chunks = load_corpus(root)
    index = BM25Index(chunks)

    mcp = FastMCP(
        "docs",
        instructions="Company handbook and policies. Search first, then cite slugs.",
        log_level=log_level(),
        transport_security=transport_security(),
    )
    advertise_list_changed(mcp)
    add_health_route(mcp)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search(query: str, k: int = 3) -> list[Hit]:
        """Search the company handbook and policies. Returns the best matching sections."""
        if not query.strip():
            raise ToolError("query must not be empty")
        k = max(1, min(k, 10))
        return [
            Hit(slug=c.slug, title=c.title, section=c.section, score=round(s, 3), text=c.text)
            for c, s in index.search(query, k)
        ]

    @mcp.resource("docs://index", mime_type="application/json")
    def docs_index() -> str:
        """Every document's slug and title."""
        return json.dumps([{"slug": s, "title": t} for s, (t, _) in sorted(docs.items())])

    @mcp.resource("docs://doc/{slug}", mime_type="text/markdown")
    def doc(slug: str) -> str:
        """The full text of one document."""
        if slug not in docs:
            raise ToolError(f"unknown document {slug!r}")
        return docs[slug][1]

    @mcp.prompt()
    def answer_with_citations(question: str) -> str:
        """Answer a policy question from the handbook, citing document slugs."""
        return (
            f"Answer this question using only the company docs: {question}\n"
            "Call docs__search first. Cite every claim as [slug]. If the docs do not "
            "answer it, say so."
        )

    return mcp


def main() -> None:
    serve_http(create_server(), default_port=8102)


if __name__ == "__main__":
    main()
