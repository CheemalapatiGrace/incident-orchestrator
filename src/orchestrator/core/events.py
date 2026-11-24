"""In-process pub/sub event bus. Every event is persisted first, so streams can be replayed."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from orchestrator.core.store import Store


class Event(BaseModel):
    seq: int
    run_id: str
    ts: float
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class Subscription:
    def __init__(self, bus: EventBus, run_id: str | None) -> None:
        self._bus = bus
        self.run_id = run_id
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=10_000)
        self.dropped = 0

    def __enter__(self) -> Subscription:
        self._bus._subs.add(self)
        return self

    def __exit__(self, *exc: object) -> None:
        self._bus._subs.discard(self)

    async def get(self, timeout: float | None = None) -> Event | None:
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except TimeoutError:
            return None

    async def __aiter__(self) -> AsyncIterator[Event]:
        while True:
            yield await self.queue.get()


class EventBus:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._subs: set[Subscription] = set()

    def publish(self, run_id: str, type_: str, data: dict[str, Any] | None = None) -> Event:
        data = data or {}
        ts = time.time()
        seq = self.store.append_event(run_id, ts, type_, data)
        ev = Event(seq=seq, run_id=run_id, ts=ts, type=type_, data=data)
        for sub in list(self._subs):
            if sub.run_id is None or sub.run_id == run_id:
                try:
                    sub.queue.put_nowait(ev)
                except asyncio.QueueFull:
                    sub.dropped += 1  # slow consumer: it can resync from the store via Last-Event-ID
        return ev

    def subscribe(self, run_id: str | None = None) -> Subscription:
        return Subscription(self, run_id)
