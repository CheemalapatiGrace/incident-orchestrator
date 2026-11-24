"""Orchestrator facade: submit incidents, plan, execute/resume runs, build results, run workers."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from orchestrator.agents.supervisor import Supervisor
from orchestrator.config import Config
from orchestrator.core.budget import BudgetTracker
from orchestrator.core.events import EventBus
from orchestrator.core.executor import Executor
from orchestrator.core.queue import InMemoryQueue, WorkQueue
from orchestrator.core.resilience import BreakerRegistry, BulkheadRegistry
from orchestrator.core.runtime import Deps, RunRuntime, TaskState
from orchestrator.core.store import Store
from orchestrator.core.tracing import Tracer
from orchestrator.errors import BudgetExceeded, InvalidState
from orchestrator.faults import FaultInjector
from orchestrator.hitl.controller import HitlController
from orchestrator.llm.base import LLM
from orchestrator.llm.factory import build_llm
from orchestrator.models import TERMINAL_RUN_STATES, Alert, Budget, Plan, RunResult, RunStatus, TaskStatus
from orchestrator.security.injection import InjectionScanner
from orchestrator.security.policy import default_policy
from orchestrator.sim.environment import SimulatedEnvironment
from orchestrator.sim.scenarios import SCENARIO_IDS
from orchestrator.tools.registry import ToolRegistry

EnvFactory = Callable[[Alert], Any]
MAX_CACHED_RUNTIMES = 500


def default_env_factory(alert: Alert) -> SimulatedEnvironment:
    sid = alert.labels.get("scenario")
    if sid not in SCENARIO_IDS:
        sid = next((s for s in SCENARIO_IDS if alert.service in s or s.startswith(alert.service.split("-")[0])), SCENARIO_IDS[0])
    return SimulatedEnvironment.for_scenario(sid)


class Orchestrator:
    def __init__(
        self,
        config: Config | None = None,
        *,
        store: Store | None = None,
        llm: LLM | None = None,
        env_factory: EnvFactory = default_env_factory,
        injector: FaultInjector | None = None,
        queue: WorkQueue | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        worker_id: str | None = None,
    ) -> None:
        self.cfg = config or Config()
        self.store = store or Store(self.cfg.db_path)
        self.bus = EventBus(self.store)
        self.deps = Deps(
            cfg=self.cfg, store=self.store, bus=self.bus, tracer=Tracer(self.store, self.bus), llm=llm or build_llm(self.cfg),
            registry=ToolRegistry.default(), policy=default_policy(enforce=self.cfg.policy_enabled),
            scanner=InjectionScanner(self.cfg.scanner_enabled),
            breakers=BreakerRegistry(self.cfg.breaker_failure_threshold, self.cfg.breaker_recovery_s),
            bulkheads=BulkheadRegistry(self.cfg.tool_concurrency, {}), injector=injector, sleep=sleep,
        )
        # per-agent bulkheads are bounded tighter than tools
        for a in ("log_analyst", "metrics_analyst", "deploy_history", "runbook", "remediation", "comms"):
            self.deps.bulkheads._limits[f"agent:{a}"] = self.cfg.agent_concurrency
        self.env_factory = env_factory
        self.queue: WorkQueue = queue or InMemoryQueue()
        self.executor = Executor(self.deps)
        self.supervisor = Supervisor(self.deps)
        self.hitl = HitlController(self)
        self.runtimes: dict[str, RunRuntime] = {}
        self._envs: dict[str, Any] = {}
        self.worker_id = worker_id or f"w-{uuid.uuid4().hex[:6]}"
        self._workers: list[asyncio.Task[None]] = []
        self._bg: set[asyncio.Task[Any]] = set()
        self._stopping = False

    # ============================================================== submission
    def submit(self, alert: Alert, *, budget: Budget | None = None, breakpoints: list[str] | None = None, env: Any = None, run_id: str | None = None) -> str:
        """Create a run row (sanitising the untrusted alert text first). Does not execute."""
        run_id = run_id or f"run-{uuid.uuid4().hex[:10]}"
        scanner = self.deps.scanner
        title, f1 = scanner.sanitize_text(alert.title, "alert.title")
        desc, f2 = scanner.sanitize_text(alert.description, "alert.description")
        labels, f3 = scanner.sanitize(alert.labels, "alert.labels")
        clean = alert.model_copy(update={"title": title, "description": desc, "labels": labels})
        findings = f1 + f2 + f3
        b = budget or self.cfg.default_budget
        self.store.create_run(run_id, clean.model_dump(mode="json"), b.model_dump(), breakpoints or [])
        self.store.audit(run_id, "system", "run_submitted", alert.id, {"service": alert.service})
        self.bus.publish(run_id, "run.submitted", {"incident": clean.model_dump(mode="json")})
        if findings:
            self.store.update_run(run_id, taint=1)
            for f in findings:
                self.bus.publish(run_id, "guardrail.injection_detected", {**f.as_dict(), "agent": "ingest"})
            self.store.audit(run_id, "system:scanner", "injection_quarantined", "alert", {"count": len(findings)})
        if env is not None:
            self._envs[run_id] = env
        return run_id

    # ============================================================== runtime mgmt
    def get_runtime(self, run_id: str) -> RunRuntime:
        rt = self.runtimes.get(run_id)
        if rt is None:
            rt = self._load_runtime(run_id)
            self.runtimes[run_id] = rt
            self._evict()
        return rt

    def _evict(self) -> None:
        if len(self.runtimes) <= MAX_CACHED_RUNTIMES:
            return
        for rid in [r for r, x in self.runtimes.items() if not x.active and x.status in TERMINAL_RUN_STATES][: len(self.runtimes) - MAX_CACHED_RUNTIMES]:
            self.runtimes.pop(rid, None)

    def _load_runtime(self, run_id: str) -> RunRuntime:
        row = self.store.get_run(run_id)
        if row is None:
            raise InvalidState(f"unknown run {run_id!r}")
        alert = Alert.model_validate(row["incident"])
        env = self._envs.get(run_id) or self.env_factory(alert)
        self._envs[run_id] = env
        plan = Plan.model_validate(row["plan"]) if row["plan"] else None
        budget = BudgetTracker(Budget.model_validate(row["budget"]), usage=row["usage"])
        rt = RunRuntime(run_id, alert, plan, budget, env, self.deps)
        rt.status = RunStatus(row["status"])
        rt.guidance = row["guidance"] or []
        rt.taint = row["taint"]
        rt.breakpoints = set(row["breakpoints"] or [])
        rt.set_paused(row["paused"])
        if plan:
            saved = self.store.get_tasks(run_id)
            for spec in plan.tasks:
                ts = TaskState.from_row(saved[spec.id]) if spec.id in saved else TaskState(spec.id)
                if ts.status is TaskStatus.RUNNING:  # was in flight when the process died: re-run it
                    ts.status, ts.attempts = TaskStatus.PENDING, 0
                rt.tasks[spec.id] = ts
        return rt

    async def _ensure_plan(self, rt: RunRuntime) -> None:
        if rt.plan is not None:
            return
        async with self.deps.tracer.span(rt.run_id, "plan", "supervisor.plan", input={"service": rt.incident.service}) as sp:
            plan, fallback = await self.supervisor.plan(rt.run_id, rt.incident, rt.budget)
            for t in plan.tasks:
                if t.id in rt.breakpoints:
                    t.breakpoint = True
            sp.set(tasks=len(plan.tasks), fallback=fallback)
            sp.set_output(plan.model_dump(mode="json"))
        rt.plan = plan
        for spec in plan.tasks:
            rt.tasks[spec.id] = TaskState(spec.id)
            rt.persist_task(spec.id)
        rt.persist_run()
        rt.emit("plan.created", plan=plan.model_dump(mode="json"), fallback=fallback)

    # ================================================================ execution
    async def execute(self, run_id: str) -> RunResult | None:
        """Plan (if needed) and run a submitted run to completion, or resume it after a crash.

        Returns ``None`` if another worker holds the run's lease."""
        if not self.store.claim_run(run_id, self.worker_id, self.cfg.lease_s):
            return None
        hb = asyncio.create_task(self._heartbeat(run_id))
        rt = self.get_runtime(run_id)
        if rt.active:
            hb.cancel()
            return None
        rt.active = True
        try:
            if rt.status in TERMINAL_RUN_STATES and all(t.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.SKIPPED} for t in rt.tasks.values()) and rt.tasks:
                return self.build_result(rt)
            if rt.status in TERMINAL_RUN_STATES:
                rt.status = RunStatus.RUNNING  # reopened by a human action (retry)
            try:
                await self._ensure_plan(rt)
            except BudgetExceeded as e:
                rt.set_status(RunStatus.BUDGET_EXCEEDED)
                rt.emit("run.aborted", reason=str(e), dimension=e.dimension)
                return self._complete(rt)
            rt.active = False  # executor.run re-asserts it
            await self.executor.run(rt)
            return self._complete(rt)
        finally:
            rt.active = False
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await hb
            self.store.release_run(run_id, self.worker_id)

    def _complete(self, rt: RunRuntime) -> RunResult:
        result = self.build_result(rt)
        self.store.update_run(rt.run_id, result=result.model_dump(mode="json"), status=rt.status.value, usage=rt.budget.snapshot())
        rt.emit("run.completed", status=rt.status.value, root_cause=result.root_cause, confidence=result.confidence)
        return result

    async def _heartbeat(self, run_id: str) -> None:
        while True:
            await asyncio.sleep(max(self.cfg.lease_s / 3, 0.05))
            self.store.claim_run(run_id, self.worker_id, self.cfg.lease_s)

    async def run_incident(self, alert: Alert, **kw: Any) -> RunResult:
        run_id = self.submit(alert, **kw)
        res = await self.execute(run_id)
        assert res is not None
        return res

    def kick(self, run_id: str) -> None:
        """Re-run an already-known run (e.g. after a human retried a node on a finished run)."""
        if self._workers:
            t = asyncio.get_running_loop().create_task(self.queue.enqueue(run_id))
        else:
            t = asyncio.get_running_loop().create_task(self.execute(run_id))
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    # ================================================================== results
    def build_result(self, rt: RunRuntime) -> RunResult:
        rem = next((rt.tasks[t.id] for t in (rt.plan.tasks if rt.plan else []) if t.agent.value == "remediation" and rt.tasks[t.id].output), None)
        comms_task = next((rt.tasks[t.id] for t in (rt.plan.tasks if rt.plan else []) if t.agent.value == "comms" and t.params.get("kind", "update") == "update" and rt.tasks[t.id].output), None)
        out = rem.output if rem and rem.output else {}
        rc = out.get("root_cause", {})
        findings = sum(1 for e in self.store.get_events(rt.run_id, limit=100000) if e["type"] == "guardrail.injection_detected")
        failed = [t.task_id for t in rt.tasks.values() if t.status is TaskStatus.FAILED]
        return RunResult(
            run_id=rt.run_id, status=rt.status, root_cause=rc.get("category"), confidence=rc.get("confidence"),
            summary=out.get("summary", ""), actions=out.get("proposed_actions", []),
            comms=(comms_task.output if comms_task else None), usage=rt.budget.snapshot(),
            dead_letters=[d["task_id"] for d in self.store.get_dead_letters(rt.run_id)], failed_tasks=failed, security_findings=findings,
        )

    # ================================================================== workers
    def start_workers(self, n: int | None = None) -> None:
        for i in range(n or self.cfg.worker_concurrency):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))

    async def stop_workers(self) -> None:
        self._stopping = True
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._stopping = False

    async def _worker_loop(self, idx: int) -> None:
        while True:
            lease = await self.queue.dequeue(timeout=0.25)
            if lease is None:
                continue
            try:
                await self.execute(lease.item)
                await self.queue.ack(lease)
            except asyncio.CancelledError:
                await self.queue.nack(lease)
                raise
            except Exception:  # noqa: BLE001 - redeliver; the run resumes from its checkpoint
                await self.queue.nack(lease)

    async def wait_for_terminal(self, run_id: str, timeout: float = 30.0) -> RunStatus:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = self.store.get_run(run_id)
            rt = self.runtimes.get(run_id)
            if row and RunStatus(row["status"]) in TERMINAL_RUN_STATES and rt is not None and not rt.active:
                return RunStatus(row["status"])
            await asyncio.sleep(0.02)
        raise TimeoutError(f"run {run_id} did not finish within {timeout}s")

    def stats(self) -> dict[str, Any]:
        return {"breakers": self.deps.breakers.all_stats(), "bulkheads": self.deps.bulkheads.all_stats(), "runtimes": len(self.runtimes), "worker_id": self.worker_id}
