"""Resilience primitives: circuit breaker, bulkhead, retry with exponential backoff + jitter."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

from orchestrator.errors import BulkheadFullError, CircuitOpenError, TransientError

T = TypeVar("T")


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Classic three-state breaker driven by consecutive failures.

    closed --(N consecutive failures)--> open --(recovery_timeout)--> half_open
    half_open --(success)--> closed ; half_open --(failure)--> open
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 10.0,
        half_open_max_calls: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._clock = clock
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_inflight = 0
        self.opened_count = 0
        self.rejected_count = 0

    @property
    def state(self) -> BreakerState:
        if self._state is BreakerState.OPEN and self._clock() - self._opened_at >= self.recovery_timeout:
            self._state = BreakerState.HALF_OPEN
            self._half_open_inflight = 0
        return self._state

    def allow(self) -> bool:
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.HALF_OPEN and self._half_open_inflight < self.half_open_max_calls:
            self._half_open_inflight += 1
            return True
        self.rejected_count += 1
        return False

    def record_success(self) -> None:
        if self._state is BreakerState.HALF_OPEN:
            self._state = BreakerState.CLOSED
        self._failures = 0
        self._half_open_inflight = 0

    def record_failure(self) -> None:
        if self._state is BreakerState.HALF_OPEN:
            self._trip()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
        self._half_open_inflight = 0
        self.opened_count += 1

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        if not self.allow():
            raise CircuitOpenError(f"circuit {self.name!r} is open")
        try:
            result = await fn()
        except TransientError:
            self.record_failure()
            raise
        except Exception:
            # non-transient errors (bad args, permission) say nothing about dependency health
            if self._state is BreakerState.HALF_OPEN:
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
            raise
        self.record_success()
        return result

    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self._failures,
            "opened_count": self.opened_count,
            "rejected_count": self.rejected_count,
        }


class BreakerRegistry:
    def __init__(self, failure_threshold: int, recovery_timeout: float, clock: Callable[[], float] = time.monotonic):
        self._args = (failure_threshold, recovery_timeout)
        self._clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, *self._args, clock=self._clock)
        return self._breakers[name]

    def all_stats(self) -> list[dict]:
        return [b.stats() for b in self._breakers.values()]


class Bulkhead:
    """Concurrency limiter with an optional bounded wait queue (sheds load when full)."""

    def __init__(self, name: str, max_concurrent: int, max_queue: int | None = None) -> None:
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self._sem = asyncio.Semaphore(max_concurrent)
        self.in_flight = 0
        self.waiting = 0
        self.peak = 0
        self.rejected = 0

    async def __aenter__(self) -> Bulkhead:
        if self.max_queue is not None and self._sem.locked() and self.waiting >= self.max_queue:
            self.rejected += 1
            raise BulkheadFullError(f"bulkhead {self.name!r} full ({self.in_flight} running, {self.waiting} queued)")
        self.waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self.waiting -= 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.in_flight -= 1
        self._sem.release()

    def stats(self) -> dict:
        return {
            "name": self.name,
            "max": self.max_concurrent,
            "in_flight": self.in_flight,
            "waiting": self.waiting,
            "peak": self.peak,
            "rejected": self.rejected,
        }


class BulkheadRegistry:
    def __init__(self, default_limit: int, limits: dict[str, int] | None = None) -> None:
        self._default = default_limit
        self._limits = limits or {}
        self._bulkheads: dict[str, Bulkhead] = {}

    def get(self, name: str) -> Bulkhead:
        if name not in self._bulkheads:
            self._bulkheads[name] = Bulkhead(name, self._limits.get(name, self._default))
        return self._bulkheads[name]

    def all_stats(self) -> list[dict]:
        return [b.stats() for b in self._bulkheads.values()]


@dataclass
class RetryPolicy:
    """Exponential backoff with full jitter: delay ~ U(0, min(cap, base * 2^(attempt-1)))."""

    max_attempts: int = 3
    base_delay: float = 0.05
    max_delay: float = 2.0
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)

    def delay_for(self, attempt: int) -> float:
        """Delay to sleep after failed attempt number ``attempt`` (1-based)."""
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return self.rng.uniform(0, ceiling) if self.jitter else ceiling


async def retry_async(
    fn: Callable[[int], Awaitable[T]],
    policy: RetryPolicy,
    *,
    retry_on: tuple[type[BaseException], ...] = (TransientError,),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``fn(attempt)`` until it succeeds, a non-retryable error is raised, or attempts run out."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn(attempt)
        except retry_on as exc:
            if attempt >= policy.max_attempts:
                raise
            delay = policy.delay_for(attempt)
            if on_retry:
                on_retry(attempt, exc, delay)
            await sleep(delay)
