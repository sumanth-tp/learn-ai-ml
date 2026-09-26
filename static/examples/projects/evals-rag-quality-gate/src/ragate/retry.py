"""Retries with exponential backoff and jitter for remote calls (LLM, judge, embeddings)."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from ragate.log import get_logger

T = TypeVar("T")
log = get_logger(__name__)


class RetryExhaustedError(RuntimeError):
    def __init__(self, what: str, attempts: int, last: BaseException) -> None:
        super().__init__(f"{what} failed after {attempts} attempts: {last!r}")
        self.last = last


def call_with_retries(
    fn: Callable[[], T],
    *,
    what: str,
    attempts: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] | None = None,
    rng: random.Random | None = None,
) -> T:
    """Call ``fn`` up to ``attempts`` times. Full-jitter backoff: sleep U(0, min(cap, b*2^n))."""
    rng = rng or random.Random()
    last: BaseException | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fn()
        except retry_on as exc:
            last = exc
            if attempt == attempts:
                break
            delay = rng.uniform(0, min(max_delay_s, base_delay_s * 2 ** (attempt - 1)))
            log.warning("retrying", what=what, attempt=attempt, delay_s=round(delay, 3),
                        error=repr(exc))
            (sleep or time.sleep)(delay)
    assert last is not None
    raise RetryExhaustedError(what, attempts, last)
