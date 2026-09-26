"""Web search behind one interface: Tavily in live mode, a corpus-backed stub offline."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from research_analyst.config import Settings
from research_analyst.providers.embeddings import HashingEmbeddings

log = logging.getLogger(__name__)


class WebResult(BaseModel):
    url: str
    title: str
    content: str
    published: date | None = None


class WebSearch(Protocol):
    async def search(self, query: str, k: int) -> list[WebResult]: ...


class SearchError(RuntimeError):
    """Raised when web search fails after retries. Workers treat it as a soft failure."""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (
        429, 500, 502, 503, 504)


class TavilySearch:
    """Tavily REST API. Retries 429/5xx/timeouts with jittered exponential backoff."""

    def __init__(self, api_key: str, url: str, timeout_s: float, attempts: int = 3) -> None:
        self._url = url
        self._attempts = attempts
        self._client = httpx.AsyncClient(
            timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"}
        )

    async def search(self, query: str, k: int) -> list[WebResult]:
        payload = {"query": query, "max_results": k, "search_depth": "basic",
                   "include_answer": False}
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._attempts),
                wait=wait_exponential_jitter(initial=0.5, max=8),
                retry=retry_if_exception(_is_transient),
                reraise=True,
            ):
                with attempt:
                    resp = await self._client.post(self._url, json=payload)
                    resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchError(f"web search failed: {exc}") from exc
        results = []
        for item in resp.json().get("results", []):
            published = None
            if raw := item.get("published_date"):
                try:
                    published = date.fromisoformat(raw[:10])
                except ValueError:
                    published = None
            results.append(WebResult(url=item["url"], title=item.get("title", item["url"]),
                                     content=item.get("content", ""), published=published))
        return results

    async def aclose(self) -> None:
        await self._client.aclose()


class StubWebSearch:
    """Offline 'internet': a JSONL corpus ranked by hashed bag-of-words cosine similarity."""

    def __init__(self, corpus_file: Path) -> None:
        self._docs = [WebResult.model_validate(json.loads(line))
                      for line in corpus_file.read_text().splitlines() if line.strip()]
        self._emb = HashingEmbeddings()
        self._vecs = self._emb.embed_documents([f"{d.title}. {d.content}" for d in self._docs])
        self.calls: list[str] = []

    async def search(self, query: str, k: int) -> list[WebResult]:
        self.calls.append(query)
        q = self._emb.embed_query(query)
        scored = sorted(
            ((sum(a * b for a, b in zip(q, v, strict=True)), d)
             for v, d in zip(self._vecs, self._docs, strict=True)),
            key=lambda x: -x[0],
        )
        return [d for score, d in scored[:k] if score > 0.05]

    async def aclose(self) -> None:
        return None


def build_web_search(settings: Settings) -> WebSearch:
    if settings.mode == "offline" or settings.tavily_api_key is None:
        if settings.mode == "live":
            log.warning("TAVILY_API_KEY not set; live mode is using the stub web search")
        return StubWebSearch(settings.corpus_dir / "web.jsonl")
    return TavilySearch(settings.tavily_api_key.get_secret_value(), settings.web_search_url,
                        settings.http_timeout_s)
