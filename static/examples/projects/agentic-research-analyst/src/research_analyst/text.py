"""Small, dependency-free text utilities shared by fakes, refinement and metrics."""

from __future__ import annotations

import re

_STOPWORD_TEXT = """
a an and are as at be been but by can could do does for from has have how if in into is it its
of on or our should than that the their them then there these they this to was we were what
when where which while who why will with would you your vs versus about instead also any all
not no so such via per
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def stem(token: str) -> str:
    """Tiny plural stemmer: costs -> cost, batteries -> battery. Enough for lexical matching."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    return {stem(t) for t in tokens(text) if t not in STOPWORDS and len(t) > 1}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if len(s.strip()) > 20]


def overlap(query: str, text: str) -> float:
    """Fraction of the query's content tokens found in ``text`` (recall-style)."""
    q = content_tokens(query)
    if not q:
        return 0.0
    t = content_tokens(text)
    return len(q & t) / len(q)


def shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    toks = tokens(text)
    return {tuple(toks[i : i + k]) for i in range(max(1, len(toks) - k + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
