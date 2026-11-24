"""Structured tracing: run -> task -> (llm | tool) spans with inputs, outputs, latency, cost.

Spans are redacted before they are persisted or streamed, so secrets/PII never reach the store.
"""

from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from orchestrator.core.events import EventBus
from orchestrator.core.store import Store
from orchestrator.security.redaction import redact_obj

_current: contextvars.ContextVar[SpanHandle | None] = contextvars.ContextVar("current_span", default=None)


def _bound(v: Any, limit: int) -> Any:
    if isinstance(v, str):
        return v if len(v) <= limit else v[:limit] + f"...[+{len(v) - limit} chars]"
    if isinstance(v, dict):
        return {k: _bound(x, limit) for k, x in v.items()}
    if isinstance(v, list):
        return [_bound(x, limit) for x in v[:200]]
    return v


def _clip(v: Any, limit: int) -> Any:
    """Redact (once, at the top) and bound the size of a payload."""
    return _bound(redact_obj(v), limit)


class SpanHandle:
    def __init__(self, run_id: str, kind: str, name: str, task_id: str | None, parent_id: str | None, input_: Any) -> None:
        self.span_id = uuid.uuid4().hex[:16]
        self.run_id = run_id
        self.kind = kind
        self.name = name
        self.task_id = task_id
        self.parent_id = parent_id
        self.input = input_
        self.output: Any = None
        self.attrs: dict[str, Any] = {}
        self.status = "ok"
        self.start = time.time()

    def set_output(self, value: Any) -> None:
        self.output = value

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


class Tracer:
    def __init__(self, store: Store, bus: EventBus, max_field_chars: int = 3000) -> None:
        self.store = store
        self.bus = bus
        self.max_field_chars = max_field_chars

    @asynccontextmanager
    async def span(
        self, run_id: str, kind: str, name: str, *, task_id: str | None = None, input: Any = None, attrs: dict | None = None
    ) -> AsyncIterator[SpanHandle]:
        parent = _current.get()
        sp = SpanHandle(run_id, kind, name, task_id or (parent.task_id if parent else None), parent.span_id if parent else None, input)
        if attrs:
            sp.attrs.update(attrs)
        token = _current.set(sp)
        self.bus.publish(
            run_id,
            "span.started",
            {"span_id": sp.span_id, "parent_id": sp.parent_id, "kind": kind, "name": name, "task_id": sp.task_id, "start": sp.start},
        )
        try:
            yield sp
        except BaseException as exc:
            if type(exc).__name__ == "GraphInterrupt":  # LangGraph parked the node at interrupt(): not a failure
                sp.attrs["interrupted"] = True
            else:
                sp.status = "cancelled" if _is_cancel(exc) else "error"
                sp.attrs["error"] = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            _current.reset(token)
            end = time.time()
            latency_ms = round((end - sp.start) * 1000, 2)
            row = {
                "span_id": sp.span_id, "run_id": run_id, "parent_id": sp.parent_id, "task_id": sp.task_id, "kind": kind,
                "name": name, "start_ts": sp.start, "end_ts": end, "latency_ms": latency_ms, "status": sp.status,
                "input": _clip(sp.input, self.max_field_chars), "output": _clip(sp.output, self.max_field_chars),
                "attrs": _clip(sp.attrs, 1000),
            }
            self.store.add_span(row)
            self.bus.publish(
                run_id,
                "span.ended",
                {
                    "span_id": sp.span_id, "parent_id": sp.parent_id, "kind": kind, "name": name, "task_id": sp.task_id,
                    "status": sp.status, "latency_ms": latency_ms, "start": sp.start, "end": end,
                    "attrs": _clip(sp.attrs, 300),
                },
            )


def _is_cancel(exc: BaseException) -> bool:
    import asyncio

    return isinstance(exc, asyncio.CancelledError)


def current_span() -> SpanHandle | None:
    return _current.get()
