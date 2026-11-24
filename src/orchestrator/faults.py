"""Fault-injection harness (chaos engineering for the orchestrator itself).

Rules target ``tool:<name>`` or ``llm:<role>`` (``*`` wildcards allowed) and fire a bounded number of
times (``times``) or probabilistically (``probability``, seeded for reproducibility).

Kinds: ``error`` (transient failure), ``timeout`` (hang until the caller's timeout fires),
``malformed`` (LLM returns garbage), ``latency`` (add delay), ``crash`` (hard process crash).
"""

from __future__ import annotations

import fnmatch
import random
from collections import Counter
from dataclasses import dataclass
from typing import Literal

Kind = Literal["error", "timeout", "malformed", "latency", "permanent_error"]


@dataclass
class FaultRule:
    target: str
    kind: Kind
    times: int | None = 1  # None = unlimited
    probability: float = 1.0
    delay_s: float = 0.0
    fired: int = 0

    def matches(self, target: str) -> bool:
        return fnmatch.fnmatch(target, self.target)


class FaultInjector:
    def __init__(self, rules: list[FaultRule] | None = None, *, seed: int = 0, crash_after_tasks: int | None = None) -> None:
        self.rules = rules or []
        self.rng = random.Random(seed)
        self.crash_after_tasks = crash_after_tasks
        self.stats: Counter[str] = Counter()

    def check(self, target: str) -> FaultRule | None:
        """Return the first matching rule that fires for this call (consuming one of its uses)."""
        for rule in self.rules:
            if not rule.matches(target):
                continue
            if rule.times is not None and rule.fired >= rule.times:
                continue
            if rule.probability < 1.0 and self.rng.random() > rule.probability:
                continue
            rule.fired += 1
            self.stats[f"{rule.kind}:{target}"] += 1
            return rule
        return None

    def total_injected(self) -> int:
        return sum(self.stats.values())

    @classmethod
    def random_chaos(cls, rate: float, seed: int = 0) -> FaultInjector:
        """Probabilistic soak-test profile: flaky tools, slow/garbled LLM output."""
        return cls(
            [
                FaultRule("tool:*", "error", times=None, probability=rate),
                FaultRule("llm:*", "malformed", times=None, probability=rate / 2),
                FaultRule("llm:*", "error", times=None, probability=rate / 2),
            ],
            seed=seed,
        )
