"""DAG executor.

Reliability features, in the order a task experiences them:
  ready-check -> run-level concurrency cap -> per-agent bulkhead -> per-agent circuit breaker ->
  per-task timeout -> retries with exponential backoff + full jitter -> dead-letter on exhaustion.
Every state transition is checkpointed to SQLite before anything depends on it, so ``resume`` after a
crash continues from the last completed node. Budgets (steps/tokens/cost/wall-clock) abort runaway runs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from orchestrator.agents.base import AgentContext
from orchestrator.agents.specialists import AGENTS
from orchestrator.core.resilience import RetryPolicy, retry_async
from orchestrator.core.runtime import Deps, RunRuntime, TaskState
from orchestrator.errors import BudgetExceeded, SimulatedCrash, TaskFailure, TaskTimeout
from orchestrator.hitl.approvals import ApprovalGate
from orchestrator.llm.structured import StructuredLLM
from orchestrator.models import TERMINAL_TASK_STATES, AgentName, RunStatus, TaskSpec, TaskStatus
from orchestrator.security.policy import GATE_PRINCIPAL
from orchestrator.security.redaction import redact_obj
from orchestrator.tools.toolbox import ToolBox

LOOP_POLL_S = 0.5


class Executor:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.gate = ApprovalGate(deps)

    # ================================================================== main loop
    async def run(self, rt: RunRuntime) -> None:
        d = self.deps
        rt.active = True
        rt.set_status(RunStatus.PAUSED if rt.paused else RunStatus.RUNNING)
        running: dict[str, asyncio.Task[None]] = {}
        try:
            async with d.tracer.span(rt.run_id, "run", rt.incident.id, input={"service": rt.incident.service, "severity": rt.incident.severity.value, "tasks": [t.id for t in (rt.plan.tasks if rt.plan else [])]}) as sp:
                status = await self._loop(rt, running)
                sp.set(final_status=status.value, **{k: v for k, v in rt.budget.snapshot().items() if k != "limits"})
        finally:
            for t in running.values():
                t.cancel()
            await asyncio.gather(*running.values(), return_exceptions=True)
            rt.active = False

    async def _loop(self, rt: RunRuntime, running: dict[str, asyncio.Task[None]]) -> RunStatus:
        assert rt.plan is not None
        while True:
            rt.wake.clear()
            try:
                self._reap(rt, running)
                self._cascade(rt)
                rt.budget.check_wall_clock()
            except BudgetExceeded as e:
                await self._abort(rt, running, RunStatus.BUDGET_EXCEEDED, f"budget exceeded: {e.dimension}", e)
                return rt.status
            if rt.cancelled:
                await self._abort(rt, running, RunStatus.CANCELLED, "cancelled by operator", None)
                return rt.status
            if not rt.paused:
                for spec in self._launchable(rt, running):
                    ts = rt.tasks[spec.id]
                    if spec.breakpoint and not ts.released:
                        if ts.status is not TaskStatus.HELD:
                            ts.status = TaskStatus.HELD
                            rt.persist_task(spec.id)
                            rt.emit("task.held", task_id=spec.id)
                        continue
                    self._launch(rt, spec, running)
            held = [t for t in rt.tasks.values() if t.status is TaskStatus.HELD]
            awaiting = [t for t in rt.tasks.values() if t.status is TaskStatus.AWAITING_APPROVAL]
            rt.budget.set_human_waiting(rt.paused or bool(held) or bool(awaiting))
            self._refresh_status(rt, awaiting)
            if all(t.status in TERMINAL_TASK_STATES for t in rt.tasks.values()):
                return self._finalize(rt)
            if not running and not rt.paused and not held and not self._launchable(rt, running):
                for t in rt.tasks.values():  # unreachable in a validated DAG; fail closed rather than hang
                    if t.status is TaskStatus.PENDING:
                        t.status, t.reason = TaskStatus.SKIPPED, "deadlock"
                        rt.persist_task(t.task_id)
                continue
            try:
                await asyncio.wait_for(rt.wake.wait(), LOOP_POLL_S)
            except TimeoutError:
                pass

    # ================================================================ scheduling
    def _dep_state(self, rt: RunRuntime, spec: TaskSpec) -> str:
        assert rt.plan is not None
        for dep in spec.depends_on:
            dts = rt.tasks[dep]
            if dts.status not in TERMINAL_TASK_STATES:
                return "wait"
            if dts.ok or rt.plan.get(dep).optional:
                continue
            return "blocked"
        return "ready"

    def _launchable(self, rt: RunRuntime, running: dict[str, Any]) -> list[TaskSpec]:
        assert rt.plan is not None
        out = []
        for spec in rt.plan.tasks:
            ts = rt.tasks[spec.id]
            if spec.id in running:
                continue
            if ts.status in (TaskStatus.PENDING, TaskStatus.HELD, TaskStatus.AWAITING_APPROVAL) and self._dep_state(rt, spec) == "ready":
                out.append(spec)
        return out

    def _cascade(self, rt: RunRuntime) -> None:
        """Skip pending tasks whose required (non-optional) upstream failed."""
        assert rt.plan is not None
        changed = True
        while changed:
            changed = False
            for spec in rt.plan.tasks:
                ts = rt.tasks[spec.id]
                if ts.status in (TaskStatus.PENDING, TaskStatus.HELD) and self._dep_state(rt, spec) == "blocked":
                    bad = next(dd for dd in spec.depends_on if not rt.tasks[dd].ok and not rt.plan.get(dd).optional)
                    ts.status, ts.reason, ts.finished_at = TaskStatus.SKIPPED, f"upstream_failed:{bad}", time.time()
                    rt.persist_task(spec.id)
                    rt.emit("task.skipped", task_id=spec.id, reason=ts.reason)
                    changed = True

    def _launch(self, rt: RunRuntime, spec: TaskSpec, running: dict[str, asyncio.Task[None]]) -> None:
        ts = rt.tasks[spec.id]
        if ts.status is not TaskStatus.AWAITING_APPROVAL:  # resumed gate keeps its status
            ts.status = TaskStatus.RUNNING
        rt.persist_task(spec.id)
        t = asyncio.create_task(self._task_main(rt, spec), name=f"{rt.run_id}:{spec.id}")
        t.add_done_callback(lambda _t: rt.wake.set())
        running[spec.id] = t

    def _reap(self, rt: RunRuntime, running: dict[str, asyncio.Task[None]]) -> None:
        for tid, t in list(running.items()):
            if not t.done():
                continue
            del running[tid]
            if t.cancelled():
                continue
            exc = t.exception()
            if exc is None:
                continue
            if isinstance(exc, BudgetExceeded):
                raise exc
            if not isinstance(exc, Exception):  # SimulatedCrash / KeyboardInterrupt: die like a real process
                raise exc
            ts = rt.tasks[tid]  # an unexpected bug inside the executor itself: contain it
            ts.status, ts.error, ts.finished_at = TaskStatus.FAILED, f"internal error: {type(exc).__name__}: {exc}", time.time()
            rt.persist_task(tid)
            rt.emit("task.failed", task_id=tid, error=ts.error)

    def _refresh_status(self, rt: RunRuntime, awaiting: list[TaskState]) -> None:
        if rt.paused:
            rt.set_status(RunStatus.PAUSED)
        elif awaiting:
            rt.set_status(RunStatus.AWAITING_APPROVAL)
        else:
            rt.set_status(RunStatus.RUNNING)

    def _finalize(self, rt: RunRuntime) -> RunStatus:
        assert rt.plan is not None
        failed_required = any(t.status is TaskStatus.FAILED and not rt.plan.get(t.task_id).optional for t in rt.tasks.values())
        degraded = any(t.status is TaskStatus.FAILED or (t.status is TaskStatus.SKIPPED and not t.human_skipped) for t in rt.tasks.values())
        status = RunStatus.FAILED if failed_required else RunStatus.DEGRADED if degraded else RunStatus.COMPLETED
        rt.budget.set_human_waiting(False)
        rt.set_status(status)
        return status

    async def _abort(self, rt: RunRuntime, running: dict[str, asyncio.Task[None]], status: RunStatus, why: str, exc: BudgetExceeded | None) -> None:
        for t in running.values():
            t.cancel()
        await asyncio.gather(*running.values(), return_exceptions=True)
        running.clear()
        for ts in rt.tasks.values():
            if ts.status in (TaskStatus.RUNNING, TaskStatus.AWAITING_APPROVAL, TaskStatus.HELD):
                ts.status, ts.reason, ts.finished_at = TaskStatus.FAILED, why, time.time()
            elif ts.status is TaskStatus.PENDING:
                ts.status, ts.reason, ts.finished_at = TaskStatus.SKIPPED, why, time.time()
            rt.persist_task(ts.task_id)
        for ap in self.deps.store.list_approvals(rt.run_id, "pending"):  # never leave a dangling approval
            self.deps.store.decide_approval(ap["approval_id"], "rejected", "system:run_aborted", why)
        rt.budget.set_human_waiting(False)
        rt.emit("run.aborted", reason=why, dimension=exc.dimension if exc else None)
        rt.set_status(status)

    # ================================================================ task execution
    def _make_ctx(self, rt: RunRuntime, spec: TaskSpec) -> AgentContext:
        d = self.deps
        assert rt.plan is not None
        ts = rt.tasks[spec.id]
        upstream = []
        for dep in spec.depends_on:
            dts = rt.tasks[dep]
            out = dts.output if dts.ok and dts.output is not None else {"_unavailable": True, "reason": dts.reason or dts.error or dts.status.value}
            upstream.append((dep, rt.plan.get(dep).agent.value, out))
        guidance = [g["text"] for g in rt.guidance if g.get("task_id") in (None, spec.id)]
        llm = StructuredLLM(llm=d.llm, cfg=d.cfg, tracer=d.tracer, bus=d.bus, budget=rt.budget, injector=d.injector, run_id=rt.run_id, agent=spec.agent.value)
        return AgentContext(d, rt, spec, {**spec.params, **ts.input_overrides}, upstream, guidance, llm, ToolBox(d, rt, spec.agent.value, spec.id))

    async def _task_main(self, rt: RunRuntime, spec: TaskSpec) -> None:
        d = self.deps
        ts = rt.tasks[spec.id]
        async with rt.sem:
            draft = ts.output if ts.status is TaskStatus.AWAITING_APPROVAL and ts.output is not None else None
            ts.started_at = ts.started_at or time.time()
            rt.persist_task(spec.id)
            rt.emit("task.started", task_id=spec.id, agent=spec.agent.value, resumed_at_gate=draft is not None)
            async with d.tracer.span(rt.run_id, "task", spec.id, task_id=spec.id, input={"agent": spec.agent.value, "params": {**spec.params, **ts.input_overrides}}, attrs={"agent": spec.agent.value}) as sp:
                try:
                    if draft is None:
                        draft = await self._with_retries(rt, spec, ts)
                    out = await self.gate.process(rt, spec, ts, draft) if spec.agent is AgentName.REMEDIATION else draft
                except (BudgetExceeded, asyncio.CancelledError):
                    raise
                except TaskFailure as tf:
                    self._dead_letter(rt, spec, ts, tf)
                    sp.status = "error"
                    sp.set(error=str(tf)[:300])
                    return
                except Exception as e:  # noqa: BLE001 - e.g. a gate bug must not wedge the run
                    self._dead_letter(rt, spec, ts, TaskFailure(spec.id, e, ts.attempts))
                    sp.status = "error"
                    return
                ts.output = redact_obj(out)
                ts.status, ts.error, ts.reason, ts.finished_at = TaskStatus.SUCCEEDED, None, None, time.time()
                sp.set_output(self._brief(ts.output))
                rt.persist_task(spec.id)
                rt.persist_run()
                rt.emit("task.succeeded", task_id=spec.id, attempts=ts.attempts, summary=str(ts.output.get("summary", ""))[:300])
                rt.completed_in_session += 1
                inj = d.injector
                if inj and inj.crash_after_tasks is not None and rt.completed_in_session >= inj.crash_after_tasks:
                    raise SimulatedCrash(f"injected crash after {rt.completed_in_session} completed tasks")

    @staticmethod
    def _brief(out: dict[str, Any]) -> dict[str, Any]:
        return {"summary": out.get("summary"), "keys": sorted(out)}

    async def _with_retries(self, rt: RunRuntime, spec: TaskSpec, ts: TaskState) -> dict[str, Any]:
        d, cfg = self.deps, self.deps.cfg
        agent = AGENTS[spec.agent]
        policy = RetryPolicy(spec.max_attempts or cfg.max_task_attempts, cfg.retry_base_delay_s, cfg.retry_max_delay_s)
        breaker = d.breakers.get(f"agent:{spec.agent.value}")
        bulkhead = d.bulkheads.get(f"agent:{spec.agent.value}")
        timeout = spec.timeout_s or cfg.task_timeout_s

        async def attempt(_n: int) -> dict[str, Any]:
            ts.attempts += 1
            rt.persist_task(spec.id)
            rt.emit("task.attempt", task_id=spec.id, attempt=ts.attempts)
            ctx = self._make_ctx(rt, spec)  # rebuilt per attempt: fresh loop detector, latest edits/guidance
            async with bulkhead:
                async def call() -> dict[str, Any]:
                    try:
                        return await asyncio.wait_for(agent.run(ctx), timeout)
                    except TimeoutError as e:
                        raise TaskTimeout(f"task {spec.id!r} exceeded {timeout}s") from e

                return await breaker.call(call)

        def on_retry(attempt_no: int, exc: BaseException, delay: float) -> None:
            rt.emit("task.retry", task_id=spec.id, attempt=attempt_no, error=f"{type(exc).__name__}: {exc}"[:300], delay_s=round(delay, 4))

        try:
            return await retry_async(attempt, policy, on_retry=on_retry, sleep=d.sleep)
        except BudgetExceeded:
            raise
        except Exception as e:
            raise TaskFailure(spec.id, e, ts.attempts) from e

    def _dead_letter(self, rt: RunRuntime, spec: TaskSpec, ts: TaskState, tf: TaskFailure) -> None:
        err = f"{type(tf.cause).__name__}: {tf.cause}"[:500]
        ts.status, ts.error, ts.finished_at = TaskStatus.FAILED, err, time.time()
        rt.persist_task(spec.id)
        self.deps.store.add_dead_letter(rt.run_id, spec.id, err, {"agent": spec.agent.value, "params": {**spec.params, **ts.input_overrides}, "attempts": ts.attempts})
        rt.emit("task.dead_lettered", task_id=spec.id, error=err, attempts=ts.attempts)
        rt.emit("task.failed", task_id=spec.id, error=err)


__all__ = ["Executor", "GATE_PRINCIPAL"]
