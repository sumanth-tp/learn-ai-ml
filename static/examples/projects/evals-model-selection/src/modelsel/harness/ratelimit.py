"""An asyncio token bucket, one per model, sized from the provider's requests-per-minute limit.

A semaphore caps how many calls are *in flight*; a token bucket caps how many
*start* per minute. You need both: 8 concurrent slow calls can still exceed a
60 RPM quota if each finishes in half a second.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


class AsyncTokenBucket:
    def __init__(
        self,
        rate_per_minute: float,
        burst: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self.rate = rate_per_minute / 60.0
        self.capacity = float(burst if burst is not None else max(1, int(rate_per_minute // 60) or 1))
        self._tokens = self.capacity
        self._clock = clock
        self._updated = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    async def acquire(self) -> float:
        """Wait for a token. Returns seconds waited (logged, so throttling is visible)."""
        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return waited
                delay = (1 - self._tokens) / self.rate
                waited += delay
                await asyncio.sleep(delay)
