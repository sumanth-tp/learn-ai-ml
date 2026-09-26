"""Clocks. Production uses wall time; the simulator uses a clock it can move forward."""

from __future__ import annotations

import threading
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Seconds since the epoch."""
        ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class SimClock:
    """A manually driven clock. `sleep` advances time instantly, so retries and
    model latency cost simulated time, not wall time."""

    def __init__(self, start: float) -> None:
        self._t = start
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._t += max(0.0, seconds)

    def set(self, t: float) -> None:
        with self._lock:
            self._t = max(self._t, t)
