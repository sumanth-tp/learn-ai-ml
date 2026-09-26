"""Source identity, deduplication and quality scoring.

Three layers of dedup, cheapest first:
1. canonical URL -> deterministic source id (same page from two workers = one id);
2. near-duplicate content (syndicated copies on different URLs) by shingle Jaccard;
3. references are renumbered from the surviving canonical ids only.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from research_analyst.models import Origin, Source
from research_analyst.text import jaccard, shingles

_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "ref",
    "mc_cid",
    "mc_eid",
}

# Domain tiers. In production this is a reviewed, versioned allow/deny list.
HIGH_TRUST_SUFFIXES = (
    ".gov",
    ".edu",
    ".int",
    "iea.org",
    "irena.org",
    "nrel.gov",
    "nature.com",
    "sciencedirect.com",
    "reuters.com",
    "ft.com",
    "bloomberg.com",
)
MEDIUM_TRUST_SUFFIXES = (
    "energy-storage.news",
    "pv-magazine.com",
    "canarymedia.com",
    "carbonbrief.org",
    "electrek.co",
    "wikipedia.org",
)
LOW_TRUST_MARKERS = ("best-", "top10", "clickfarm", "deals", "coupon", "blogspot")


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = "https" if parts.scheme in ("http", "https", "") else parts.scheme
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    query = urlencode(
        sorted((k, v) for k, v in parse_qsl(parts.query) if k.lower() not in _TRACKING_PARAMS)
    )
    return urlunsplit((scheme, host, path, query, ""))


def source_id_for(url: str) -> str:
    return "S" + hashlib.sha256(canonical_url(url).encode()).hexdigest()[:10]


def domain_score(url: str, origin: Origin) -> float:
    if origin is Origin.INTERNAL:
        return 0.9
    host = urlsplit(canonical_url(url)).netloc
    if any(m in host for m in LOW_TRUST_MARKERS):
        return 0.1
    if host.endswith(HIGH_TRUST_SUFFIXES):
        return 0.9
    if host.endswith(MEDIUM_TRUST_SUFFIXES):
        return 0.65
    return 0.45


def recency_score(published: date | None, today: date | None = None) -> float:
    if published is None:
        return 0.5
    today = today or date.today()
    age_years = max(0.0, (today - published).days / 365.25)
    return max(0.2, 1.0 - 0.15 * age_years)  # loses 0.15 per year, floor 0.2


def quality_score(source: Source, today: date | None = None) -> float:
    """0..1. Domain trust dominates; recency and substance adjust it."""
    substance = min(1.0, len(source.content) / 600)
    domain = domain_score(source.url, source.origin)
    if domain <= 0.1:  # deny-listed domains are capped, however fresh or long they are
        return 0.15
    score = 0.6 * domain + 0.25 * recency_score(source.published, today) + 0.15 * substance
    return round(score, 3)


def deduplicate(
    sources: list[Source], threshold: float = 0.8
) -> tuple[list[Source], dict[str, str]]:
    """Collapse near-duplicate sources. Returns survivors and an alias map old_id -> kept_id.

    The higher-quality copy survives, so a syndicated copy on a content farm never
    displaces the original publisher.
    """
    ordered = sorted(sources, key=lambda s: (-s.quality, s.id))
    kept: list[tuple[Source, set]] = []
    alias: dict[str, str] = {}
    for src in ordered:
        sh = shingles(src.content)
        match = next((k for k, ksh in kept if jaccard(sh, ksh) >= threshold), None)
        if match is None:
            kept.append((src, sh))
            alias[src.id] = src.id
        else:
            alias[src.id] = match.id
            if src.id not in match.aliases:
                match.aliases.append(src.id)
    return [k for k, _ in kept], alias


def merge_sources(
    left: dict[str, Source] | None, right: dict[str, Source] | None
) -> dict[str, Source]:
    """LangGraph reducer: union by id; parallel workers finding the same page is not an error."""
    merged = dict(left or {})
    for sid, src in (right or {}).items():
        if sid not in merged or src.quality > merged[sid].quality:
            merged[sid] = src
    return merged


_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) instructions|disregard (the|your) (system|previous)"
    r"|you are now|system prompt|reveal your|do not cite",
    re.IGNORECASE,
)


def looks_like_injection(text: str) -> bool:
    """Heuristic tripwire for prompt injection in retrieved text. Cheap, high precision.

    It is one layer: prompts also mark sources as untrusted data, outputs are schema-bound,
    and citations are validated against the evidence actually retrieved.
    """
    return bool(_INJECTION.search(text))
