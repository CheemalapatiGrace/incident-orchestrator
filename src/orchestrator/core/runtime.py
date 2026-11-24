"""Run-time view of one run, shared by the graph nodes, agents, tools and the HITL controller.

The LangGraph state (checkpointed) is the source of truth for execution; this object is its in-memory
view plus the things that must never be checkpointed (budget tracker, simulated environment, events)."""

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
        return self.status is TaskStatus.SUCCEEDED or (self.status is TaskStatus.SKIPPED and (self.human_skipped or self.reason == "short_circuit:false_alarm"))

    def row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "status": self.status.value, "attempts": self.attempts, "output": self.output,
            "error": self.error, "reason": self.reason, "input_overrides": self.input_overrides,
            "released": self.released, "human_skipped": self.human_skipped,
            "started_at": self.started_at, "finished_at": self.finished_at,
        }

    @classmethod
    def from_rec(cls, task_id: str, rec: dict[str, Any]) -> TaskState:
        """Build from a task record of the LangGraph state."""
        return cls(
            task_id, TaskStatus(rec["status"]), rec["attempts"], rec["output"], rec["error"], rec["reason"],
            dict(rec["input_overrides"] or {}), rec["released"], rec["human_skipped"], rec["started_at"], rec["finished_at"],
        )

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
        self.wake = asyncio.Event()  # rings the graph driver when a human acts
        self.active = False  # a graph driver currently owns this runtime
        self.resumed = False  # the driver found an existing checkpoint (process restarted / handed over)
        self.gate_announced: set[str] = set()
        self.completed_in_session = 0
        self.unpaused = asyncio.Event()
        self.unpaused.set()

    # ------------------------------------------------------------------- helpers
    def emit(self, type_: str, **data: Any) -> None:
        self.deps.bus.publish(self.run_id, type_, data)

    def apply_patch(self, patch: dict[str, Any], *, operator_origin: bool = False) -> None:
        """Mirror a state patch into this view (and the read-model tables).

        Operator-origin fields (plan edits, pause, cancel, guidance) are applied to the view by the HITL
        controller the moment the human acts, so a patch echoing them back from the graph is ignored here
        unless ``operator_origin`` is set (used when rehydrating from a checkpoint)."""
        if "plan" in patch and (operator_origin or self.plan is None) and patch["plan"]:
            self.plan = Plan.model_validate(patch["plan"])
        for tid, p in (patch.get("tasks") or {}).items():
            if p is None:
                self.tasks.pop(tid, None)
                self.deps.store.delete_task(self.run_id, tid)
                continue
            ts = self.tasks.setdefault(tid, TaskState(tid))
            for k, v in p.items():
                setattr(ts, k, TaskStatus(v) if k == "status" else v)
            self.persist_task(tid)
        if patch.get("taint"):
            self.taint = True
        if operator_origin:
            if "paused" in patch:
                self.set_paused(bool(patch["paused"]))
            if patch.get("cancelled"):
                self.cancelled = True
            for g in patch.get("guidance") or []:
                if all(x["id"] != g["id"] for x in self.guidance):
                    self.guidance.append(g)

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
