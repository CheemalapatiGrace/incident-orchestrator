"""Budget enforcement (steps / tokens / cost / wall-clock) and loop detection."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from orchestrator.errors import BudgetExceeded, LoopDetected
from orchestrator.models import Budget


class BudgetTracker:
    """Tracks consumption for one run. Wall-clock excludes time spent waiting on humans."""

    def __init__(
        self,
        budget: Budget,
        clock: Callable[[], float] = time.monotonic,
        usage: dict | None = None,
    ) -> None:
        self.budget = budget
        self._clock = clock
        u = usage or {}
        self.steps = int(u.get("steps", 0))
        self.tokens_in = int(u.get("tokens_in", 0))
        self.tokens_out = int(u.get("tokens_out", 0))
        self.cost_usd = float(u.get("cost_usd", 0.0))
        self._elapsed_before = float(u.get("active_s", 0.0))  # carried across crash/resume
        self._seg_start = clock()
        self._wait_start: float | None = None
        self._human_wait = 0.0

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def active_seconds(self) -> float:
        waiting = self._human_wait + (self._clock() - self._wait_start if self._wait_start is not None else 0.0)
        return self._elapsed_before + (self._clock() - self._seg_start) - waiting

    def set_human_waiting(self, waiting: bool) -> None:
        if waiting and self._wait_start is None:
            self._wait_start = self._clock()
        elif not waiting and self._wait_start is not None:
            self._human_wait += self._clock() - self._wait_start
            self._wait_start = None

    def charge_step(self, n: int = 1) -> None:
        self.steps += n
        if self.steps > self.budget.max_steps:
            raise BudgetExceeded("steps", self.steps, self.budget.max_steps)
        self.check_wall_clock()

    def charge_llm(self, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += cost_usd
        if self.tokens > self.budget.max_tokens:
            raise BudgetExceeded("tokens", self.tokens, self.budget.max_tokens)
        if self.cost_usd > self.budget.max_cost_usd:
            raise BudgetExceeded("cost_usd", self.cost_usd, self.budget.max_cost_usd)

    def check_wall_clock(self) -> None:
        active = self.active_seconds()
        if active > self.budget.max_wall_clock_s:
            raise BudgetExceeded("wall_clock_s", active, self.budget.max_wall_clock_s)

    def snapshot(self) -> dict:
        return {
            "steps": self.steps,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "active_s": round(self.active_seconds(), 3),
            "limits": self.budget.model_dump(),
        }


class LoopDetector:
    """Detects runaway agent behaviour: the same call repeated, or a short cycle of calls repeated.

    One instance is scoped to a single agent invocation (a retry after a crash legitimately repeats
    calls, so counts must not leak across attempts).
    """

    def __init__(self, max_repeats: int = 3, window: int = 24) -> None:
        self.max_repeats = max_repeats
        self._history: deque[str] = deque(maxlen=window)

    def record(self, signature: str) -> None:
        self._history.append(signature)
        h = list(self._history)
        # identical call repeated N times in a row
        if len(h) >= self.max_repeats and len(set(h[-self.max_repeats :])) == 1:
            raise LoopDetected(f"call {signature!r} repeated {self.max_repeats}x consecutively")
        # cycle of length 2..3 repeated N times (A,B,A,B,A,B)
        for k in (2, 3):
            need = k * self.max_repeats
            if len(h) >= need:
                tail = h[-need:]
                block = tail[:k]
                if len(set(block)) > 1 and all(tail[i] == block[i % k] for i in range(need)):
                    raise LoopDetected(f"call cycle {block!r} repeated {self.max_repeats}x")
