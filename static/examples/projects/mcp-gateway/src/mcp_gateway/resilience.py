"""Result cache for read-only tools and a circuit breaker per upstream."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TTLCache:
    """Small LRU + TTL cache. Values are stored as-is (ToolResult objects)."""

    def __init__(
        self, ttl: float, max_entries: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < self._clock():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        if self.ttl <= 0:
            return
        with self._lock:
            self._data[key] = (self._clock() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def invalidate_prefix(self, prefix: str) -> None:
        with self._lock:
            for k in [k for k in self._data if k.startswith(prefix)]:
                del self._data[k]

    def __len__(self) -> int:
        return len(self._data)


class BreakerState(Enum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitOpenError(Exception):
    pass


@dataclass
class CircuitBreaker:
    """Consecutive-failure breaker: CLOSED -> OPEN after N failures,
    OPEN -> HALF_OPEN after ``reset_seconds``, one probe decides the rest."""

    name: str
    failure_threshold: int = 5
    reset_seconds: float = 30.0
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.state = BreakerState.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self._probe_in_flight = False
        self._lock = threading.Lock()

    def before_call(self) -> None:
        with self._lock:
            if self.state is BreakerState.OPEN:
                if self.clock() - self.opened_at < self.reset_seconds:
                    raise CircuitOpenError(f"upstream '{self.name}' circuit is open")
                self.state = BreakerState.HALF_OPEN
            if self.state is BreakerState.HALF_OPEN:
                if self._probe_in_flight:
                    raise CircuitOpenError(f"upstream '{self.name}' is being probed")
                self._probe_in_flight = True

    def on_success(self) -> None:
        with self._lock:
            self.state = BreakerState.CLOSED
            self.failures = 0
            self._probe_in_flight = False

    def release_probe(self) -> None:
        """End a half-open probe without a verdict (the call proved nothing)."""
        with self._lock:
            self._probe_in_flight = False
            if self.state is BreakerState.HALF_OPEN:
                self.state = BreakerState.OPEN
                self.opened_at = self.clock() - self.reset_seconds  # next call may probe

    def on_failure(self) -> None:
        with self._lock:
            self._probe_in_flight = False
            self.failures += 1
            if self.state is BreakerState.HALF_OPEN or self.failures >= self.failure_threshold:
                self.state = BreakerState.OPEN
                self.opened_at = self.clock()
