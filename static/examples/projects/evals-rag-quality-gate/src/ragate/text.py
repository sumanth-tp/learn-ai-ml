"""Small, dependency-free text utilities used by chunking, BM25, fakes and heuristics."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been before being but by can could did do does for from
    had has have how i if in into is it its may me might must my no not of on or our
    per should so than that the their them then there these they this those to up
    us was we were what when where which while who why will with within would you your
    any all also each every other such via own same only just about after""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def stem(token: str) -> str:
    """A deliberately tiny suffix stripper: enough to match 'days'/'day', 'approved'/'approve'."""
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokens(text: str) -> list[str]:
    return [t.replace(",", "") for t in _TOKEN_RE.findall(text.lower())]


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


def sentences(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    body = " ".join(ln for ln in lines if ln and not ln.startswith("#"))
    return [s.strip() for s in _SENTENCE_RE.split(body) if s.strip()]


def coverage(needle: str, haystack: str) -> float:
    """Share of the needle's content tokens that appear in the haystack (0..1)."""
    need = set(content_tokens(needle))
    if not need:
        return 0.0
    return len(need & set(content_tokens(haystack))) / len(need)


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = tokens(text)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = content_tokens(prediction), content_tokens(reference)
    if not pred or not ref:
        return 0.0
    common = 0
    ref_counts: dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t in pred:
        if ref_counts.get(t, 0) > 0:
            common += 1
            ref_counts[t] -= 1
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)
