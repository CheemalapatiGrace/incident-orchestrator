"""Pluggable work-queue interface for horizontal scaling.

Workers pull run ids from a :class:`WorkQueue`. The default :class:`InMemoryQueue` serves a single
process; :class:`RedisQueue` shows the same contract on Redis lists (reliable-queue pattern). Because
run state lives in the checkpoint store (not in the queue), any worker can pick up any run, and a
crashed worker's run is resumed from its last completed node once its lease expires.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Lease:
    item: str
    token: str
    deliveries: int = 1


class WorkQueue(Protocol):
    async def enqueue(self, item: str) -> None: ...
    async def dequeue(self, timeout: float | None = None) -> Lease | None: ...
    async def ack(self, lease: Lease) -> None: ...
    async def nack(self, lease: Lease) -> None: ...
    async def size(self) -> int: ...
    async def dead_letters(self) -> list[str]: ...


class InMemoryQueue:
    """asyncio queue with at-least-once delivery (visibility timeout), max deliveries and a DLQ."""

    def __init__(self, visibility_timeout_s: float = 60.0, max_deliveries: int = 3) -> None:
        self._q: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        self._inflight: dict[str, tuple[Lease, float]] = {}
        self._dlq: list[str] = []
        self.visibility_timeout_s = visibility_timeout_s
        self.max_deliveries = max_deliveries
        self._seq = 0

    async def enqueue(self, item: str) -> None:
        await self._q.put((item, 0))

    def _reap_expired(self) -> None:
        now = time.monotonic()
        for tok, (lease, deadline) in list(self._inflight.items()):
            if deadline <= now:
                del self._inflight[tok]
                self._requeue_or_dead(lease)

    def _requeue_or_dead(self, lease: Lease) -> None:
        if lease.deliveries >= self.max_deliveries:
            self._dlq.append(lease.item)
        else:
            self._q.put_nowait((lease.item, lease.deliveries))

    async def dequeue(self, timeout: float | None = None) -> Lease | None:
        self._reap_expired()
        try:
            item, deliveries = await asyncio.wait_for(self._q.get(), timeout) if timeout is not None else await self._q.get()
        except TimeoutError:
            return None
        self._seq += 1
        lease = Lease(item, f"l{self._seq}", deliveries + 1)
        self._inflight[lease.token] = (lease, time.monotonic() + self.visibility_timeout_s)
        return lease

    async def ack(self, lease: Lease) -> None:
        self._inflight.pop(lease.token, None)

    async def nack(self, lease: Lease) -> None:
        if self._inflight.pop(lease.token, None):
            self._requeue_or_dead(lease)

    async def size(self) -> int:
        return self._q.qsize()

    async def dead_letters(self) -> list[str]:
        return list(self._dlq)


class RedisQueue:
    """Redis adapter (reliable-queue pattern: LMOVE pending -> processing; ack = LREM).

    Documented stub: it implements the full :class:`WorkQueue` contract against any client exposing
    the handful of ``redis.asyncio`` commands used below (see ``tests/test_queue.py`` for a fake).
    To use it for real: ``pip install incident-orchestrator[redis]`` and pass ``redis.asyncio.from_url(...)``.
    Not load-tested; a production deployment would add a reaper that re-queues stale ``processing`` items.
    """

    def __init__(self, client: Any, name: str = "io:runs", max_deliveries: int = 3) -> None:
        self.r = client
        self.pending, self.processing, self.dlq, self.attempts = f"{name}:pending", f"{name}:processing", f"{name}:dlq", f"{name}:attempts"
        self.max_deliveries = max_deliveries

    async def enqueue(self, item: str) -> None:
        await self.r.lpush(self.pending, item)

    async def dequeue(self, timeout: float | None = None) -> Lease | None:
        item = await self.r.blmove(self.pending, self.processing, int(timeout or 0), "RIGHT", "LEFT")
        if item is None:
            return None
        item = item.decode() if isinstance(item, bytes) else item
        n = await self.r.hincrby(self.attempts, item, 1)
        return Lease(item, item, int(n))

    async def ack(self, lease: Lease) -> None:
        await self.r.lrem(self.processing, 1, lease.item)
        await self.r.hdel(self.attempts, lease.item)

    async def nack(self, lease: Lease) -> None:
        await self.r.lrem(self.processing, 1, lease.item)
        if lease.deliveries >= self.max_deliveries:
            await self.r.lpush(self.dlq, lease.item)
        else:
            await self.r.lpush(self.pending, lease.item)

    async def size(self) -> int:
        return int(await self.r.llen(self.pending))

    async def dead_letters(self) -> list[str]:
        return [x.decode() if isinstance(x, bytes) else x for x in await self.r.lrange(self.dlq, 0, -1)]
