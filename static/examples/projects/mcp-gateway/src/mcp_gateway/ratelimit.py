"""Rate limiting (per-minute token buckets) and quotas (per-day counters).

Two mechanisms because they answer different questions:

* The **token bucket** protects upstreams from bursts: a runaway agent loop
  calling ``search`` 40 times a second is stopped within one second.
* The **daily quota** caps spend and blast radius: at most 50 refunds per
  user per day, even if each call is slow enough to pass the bucket.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from mcp_gateway.policy import Limit, Principal
from mcp_gateway.state import QuotaStore


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBuckets:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def try_take(self, key: str, per_minute: int) -> float | None:
        """Take one token. Returns None if allowed, else seconds until one is available."""
        rate = per_minute / 60.0
        now = self._clock()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = _Bucket(tokens=float(per_minute), updated=now)
            b.tokens = min(float(per_minute), b.tokens + (now - b.updated) * rate)
            b.updated = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return None
            return (1.0 - b.tokens) / rate if rate > 0 else 60.0


class RateLimitedError(Exception):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class Limiter:
    def __init__(self, quotas: QuotaStore, buckets: TokenBuckets | None = None) -> None:
        self.quotas = quotas
        self.buckets = buckets or TokenBuckets()

    def check(self, p: Principal, tool: str, user_limit: Limit, tool_limit: Limit | None) -> None:
        """Raise RateLimitedError when any limit is exhausted."""
        wait = self.buckets.try_take(f"u:{p.subject}", user_limit.per_minute)
        if wait is not None:
            raise RateLimitedError(f"rate limit: {user_limit.per_minute}/min per user", wait)
        if tool_limit is not None:
            wait = self.buckets.try_take(f"t:{p.subject}:{tool}", tool_limit.per_minute)
            if wait is not None:
                raise RateLimitedError(
                    f"rate limit: {tool_limit.per_minute}/min for {tool}", wait
                )
        scopes = [("*", user_limit.per_day)]
        if tool_limit is not None:
            scopes.append((tool, tool_limit.per_day))
        exhausted = self.quotas.try_consume(p.subject, scopes)
        if exhausted is not None:
            what = "all tools" if exhausted == "*" else exhausted
            raise RateLimitedError(f"daily quota exhausted for {what}")
