"""Run-time state shared by the executor, agents, tools and HITL controller."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orchestrator.config import Config
from orchestrator.core.budget import BudgetTracker
from orchestrator.core.events import EventBus
from orchestrator.core.resilience import BreakerRegistry, BulkheadRegistry
from orchestrator.core.store import Store
from orchestrator.core.tracing import Tracer
from orchestrator.faults import FaultInjector
from orchestrator.llm.base import LLM
from orchestrator.models import Alert, Plan, RunStatus, TaskStatus
from orchestrator.security.injection import InjectionScanner
from orchestrator.security.policy import PermissionPolicy

if TYPE_CHECKING:
    from orchestrator.tools.registry import ToolRegistry


@dataclass
class Deps:
    cfg: Config
    store: Store
    bus: EventBus
    tracer: Tracer
    llm: LLM
    registry: ToolRegistry
    policy: PermissionPolicy
    scanner: InjectionScanner
    breakers: BreakerRegistry
    bulkheads: BulkheadRegistry
    injector: FaultInjector | None = None
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


@dataclass
class TaskState:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    output: dict[str, Any] | None = None
    error: str | None = None
    reason: str | None = None
    input_overrides: dict[str, Any] = field(default_factory=dict)
    released: bool = False
    human_skipped: bool = False
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.SUCCEEDED or (self.status is TaskStatus.SKIPPED and self.human_skipped)

    def row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "status": self.status.value, "attempts": self.attempts, "output": self.output,
            "error": self.error, "reason": self.reason, "input_overrides": self.input_overrides,
            "released": self.released, "human_skipped": self.human_skipped,
            "started_at": self.started_at, "finished_at": self.finished_at,
        }

    @classmethod
    def from_row(cls, r: dict[str, Any]) -> TaskState:
        return cls(
            r["task_id"], TaskStatus(r["status"]), r["attempts"], r["output"], r["error"], r["reason"],
            r["input_overrides"] or {}, r["released"], r["human_skipped"], r["started_at"], r["finished_at"],
        )


class RunRuntime:
    """In-memory working state of one run; every mutation is mirrored to the store."""

    def __init__(self, run_id: str, incident: Alert, plan: Plan | None, budget: BudgetTracker, env: Any, deps: Deps) -> None:
        self.run_id = run_id
        self.incident = incident
        self.plan = plan
        self.budget = budget
        self.env = env
        self.deps = deps
        self.tasks: dict[str, TaskState] = {}
        self.guidance: list[dict[str, Any]] = []
        self.paused = False
        self.taint = False  # evidence contained suspected prompt injection
        self.cancelled = False
        self.status = RunStatus.PENDING
        self.breakpoints: set[str] = set()
        self.wake = asyncio.Event()
        self.approval_events: dict[str, asyncio.Event] = {}
        self.active = False  # an executor loop currently owns this runtime
        self.sem = asyncio.Semaphore(deps.cfg.max_parallel_tasks)
        self.completed_in_session = 0
        self.unpaused = asyncio.Event()
        self.unpaused.set()

    # ------------------------------------------------------------------- helpers
    def emit(self, type_: str, **data: Any) -> None:
        self.deps.bus.publish(self.run_id, type_, data)

    def persist_task(self, task_id: str) -> None:
        self.deps.store.save_task(self.run_id, self.tasks[task_id].row())

    def persist_run(self, **extra: Any) -> None:
        self.deps.store.update_run(
            self.run_id, status=self.status.value, plan=self.plan.model_dump(mode="json") if self.plan else None,
            usage=self.budget.snapshot(), guidance=self.guidance, paused=int(self.paused), taint=int(self.taint),
            breakpoints=sorted(self.breakpoints), **extra,
        )

    def set_status(self, status: RunStatus) -> None:
        if status != self.status:
            self.status = status
            self.persist_run()
            self.emit("run.status", status=status.value)

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        if paused:
            self.unpaused.clear()
        else:
            self.unpaused.set()

    def dependents(self, task_id: str) -> list[str]:
        assert self.plan
        return [t.id for t in self.plan.tasks if task_id in t.depends_on]

    def kick(self) -> None:
        self.wake.set()

    def now(self) -> float:
        return time.time()
