import asyncio
import random

import pytest

from orchestrator.core.resilience import BreakerState, Bulkhead, CircuitBreaker, RetryPolicy, retry_async
from orchestrator.errors import BulkheadFullError, CircuitOpenError, FatalToolError, ToolExecutionError


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def _fail():
    raise ToolExecutionError("t", "boom")


async def _ok():
    return 1


async def test_breaker_opens_after_threshold_and_fails_fast():
    clock = FakeClock()
    b = CircuitBreaker("x", failure_threshold=3, recovery_timeout=10, clock=clock)
    for _ in range(3):
        with pytest.raises(ToolExecutionError):
            await b.call(_fail)
    assert b.state is BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        await b.call(_ok)
    assert b.rejected_count == 1


async def test_breaker_half_open_probe_closes_on_success():
    clock = FakeClock()
    b = CircuitBreaker("x", failure_threshold=1, recovery_timeout=5, clock=clock)
    with pytest.raises(ToolExecutionError):
        await b.call(_fail)
    clock.t = 6
    assert b.state is BreakerState.HALF_OPEN
    assert await b.call(_ok) == 1
    assert b.state is BreakerState.CLOSED


async def test_breaker_half_open_failure_reopens():
    clock = FakeClock()
    b = CircuitBreaker("x", failure_threshold=1, recovery_timeout=5, clock=clock)
    with pytest.raises(ToolExecutionError):
        await b.call(_fail)
    clock.t = 6
    with pytest.raises(ToolExecutionError):
        await b.call(_fail)
    assert b.state is BreakerState.OPEN and b.opened_count == 2


async def test_breaker_half_open_admits_limited_probes():
    clock = FakeClock()
    b = CircuitBreaker("x", failure_threshold=1, recovery_timeout=1, half_open_max_calls=1, clock=clock)
    with pytest.raises(ToolExecutionError):
        await b.call(_fail)
    clock.t = 2
    assert b.allow() is True
    assert b.allow() is False  # second concurrent probe is rejected


async def test_breaker_ignores_non_transient_errors():
    b = CircuitBreaker("x", failure_threshold=1)

    async def bad_args():
        raise FatalToolError("bad args")

    for _ in range(3):
        with pytest.raises(FatalToolError):
            await b.call(bad_args)
    assert b.state is BreakerState.CLOSED


async def test_bulkhead_caps_concurrency():
    bh = Bulkhead("b", max_concurrent=3)

    async def work():
        async with bh:
            await asyncio.sleep(0.01)

    await asyncio.gather(*(work() for _ in range(20)))
    assert bh.peak == 3 and bh.in_flight == 0


async def test_bulkhead_sheds_load_when_queue_full():
    bh = Bulkhead("b", max_concurrent=1, max_queue=1)
    gate = asyncio.Event()

    async def hold():
        async with bh:
            await gate.wait()

    t1 = asyncio.create_task(hold())
    await asyncio.sleep(0)
    t2 = asyncio.create_task(hold())  # queued
    await asyncio.sleep(0)
    with pytest.raises(BulkheadFullError):
        async with bh:
            pass
    gate.set()
    await asyncio.gather(t1, t2)
    assert bh.rejected == 1


def test_backoff_is_exponential_capped_and_jittered():
    p = RetryPolicy(5, base_delay=0.1, max_delay=1.0, jitter=False)
    assert [p.delay_for(i) for i in range(1, 6)] == [0.1, 0.2, 0.4, 0.8, 1.0]
    j = RetryPolicy(5, base_delay=0.1, max_delay=1.0, jitter=True, rng=random.Random(7))
    delays = [j.delay_for(4) for _ in range(200)]
    assert all(0 <= d <= 0.8 for d in delays) and len({round(d, 3) for d in delays}) > 50


async def test_retry_succeeds_after_transient_failures_and_sleeps_with_backoff():
    calls, slept = [], []

    async def flaky(attempt):
        calls.append(attempt)
        if attempt < 3:
            raise ToolExecutionError("t", "x")
        return "ok"

    async def fake_sleep(d):
        slept.append(d)

    r = await retry_async(flaky, RetryPolicy(4, 0.1, 1.0, jitter=False), sleep=fake_sleep)
    assert r == "ok" and calls == [1, 2, 3] and slept == [0.1, 0.2]


async def test_retry_does_not_retry_non_transient():
    n = 0

    async def fatal(_):
        nonlocal n
        n += 1
        raise FatalToolError("no")

    with pytest.raises(FatalToolError):
        await retry_async(fatal, RetryPolicy(5, 0, 0))
    assert n == 1


async def test_retry_gives_up_after_max_attempts():
    n = 0

    async def always(_):
        nonlocal n
        n += 1
        raise ToolExecutionError("t", "x")

    with pytest.raises(ToolExecutionError):
        await retry_async(always, RetryPolicy(3, 0, 0))
    assert n == 3
