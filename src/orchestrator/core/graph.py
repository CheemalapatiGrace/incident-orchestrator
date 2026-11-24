"""The LangGraph orchestration core.

Topology (a ``StateGraph`` over :class:`~orchestrator.core.state.RunState`)::

    START -> supervisor -> dispatch --Send--> {log_analyst | metrics_analyst | deploy_history | runbook |
                              ^   |                       remediation | comms}   (one Send per ready task)
                              |   |--Send--> approval_gate        (interrupt(): a human decides)
                              |   |--------> hold                 (interrupt_before: paused / breakpoint)
                              |   `--------> finalize -> END
       every specialist, approval_gate and hold route back to dispatch

``dispatch`` is the scheduler: it folds queued human commands into the state, enforces the wall-clock
budget / cancellation, skips tasks whose required upstream failed, marks breakpoints as held, and the
conditional edge ``route`` fans the plan out with ``Send`` - every task whose dependencies are done
runs in the *same superstep*, concurrently. LangGraph's checkpointer persists the state after every
superstep, so a crashed process resumes exactly there.

What we add on top of LangGraph (``run_task`` / ``ApprovalGate``): per-agent circuit breaker and
bulkhead, per-attempt timeout, retries with exponential backoff and jitter, dead-lettering, injection
scanning/redaction, risk scoring and the approval-before-mutation rule.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, Send

from orchestrator.agents.base import AgentContext
from orchestrator.agents.specialists import AGENTS
from orchestrator.core.events import set_node_writer
from orchestrator.core.resilience import RetryPolicy as BackoffPolicy
from orchestrator.core.resilience import retry_async
from orchestrator.core.runtime import RunRuntime
from orchestrator.core.state import (
    AWAITING,
    FAILED,
    HELD,
    PENDING,
    SHORT_CIRCUIT,
    SKIPPED,
    SUCCEEDED,
    TERMINAL,
    RunState,
    deliberate_skip,
    dep_state,
    false_alarm,
    fold,
    is_ok,
    new_task,
    plan_of,
    upstream_of,
)
from orchestrator.errors import BudgetExceeded, SimulatedCrash, TaskFailure, TaskTimeout
from orchestrator.llm.structured import StructuredLLM
from orchestrator.models import AgentName, RunStatus, TaskSpec, TaskStatus
from orchestrator.security.redaction import redact_obj
from orchestrator.tools.toolbox import ToolBox

if TYPE_CHECKING:
    from orchestrator.engine import Orchestrator

AGENT_NODES = [a.value for a in AgentName]
# infrastructure faults (a locked SQLite file) are retried by LangGraph's own RetryPolicy; agent/tool
# faults are handled by our reliability layer inside the node
INFRA_RETRY = RetryPolicy(max_attempts=3, initial_interval=0.05, max_interval=0.5, retry_on=sqlite3.OperationalError)


def _now() -> float:
    return time.time()


def abort_patch(status: RunStatus, reason: str, dimension: str | None = None) -> dict[str, Any]:
    return {"abort": {"status": status.value, "reason": reason, "dimension": dimension}}


class GraphNodes:
    """The node functions, closed over the orchestrator's services."""

    def __init__(self, eng: Orchestrator) -> None:
        self.e = eng
        self.d = eng.deps

    # ------------------------------------------------------------------------------ plumbing
    def rt(self, config: RunnableConfig) -> RunRuntime:
        return self.e.get_runtime(config["configurable"]["thread_id"])

    @staticmethod
    def _bind_events() -> None:
        """Route every event published in this node through LangGraph's custom stream, so the driver
        feeds the event bus in superstep order (``stream_mode=['updates', 'custom']``)."""
        try:
            set_node_writer(get_stream_writer())
        except RuntimeError:  # called outside a graph run (unit tests of a node)
            set_node_writer(None)

    # ------------------------------------------------------------------------------ supervisor
    async def supervisor(self, state: RunState, config: RunnableConfig) -> dict[str, Any]:
        self._bind_events()
        rt = self.rt(config)
        if state.get("plan"):
            return {}
        try:
            async with self.d.tracer.span(rt.run_id, "plan", "supervisor.plan", input={"service": rt.incident.service}) as sp:
                plan, fallback = await self.e.supervisor.plan(rt.run_id, rt.incident, rt.budget)
                for t in plan.tasks:
                    if t.id in rt.breakpoints:
                        t.breakpoint = True
                sp.set(tasks=len(plan.tasks), fallback=fallback)
                sp.set_output(plan.model_dump(mode="json"))
        except BudgetExceeded as e:
            patch = abort_patch(RunStatus.BUDGET_EXCEEDED, str(e), e.dimension)
            return patch
        patch = {"plan": plan.model_dump(mode="json"), "tasks": {t.id: new_task() for t in plan.tasks}}
        rt.apply_patch(patch, operator_origin=True)
        rt.emit("plan.created", plan=plan.model_dump(mode="json"), fallback=fallback)
        return patch

    # --------------------------------------------------------------------------------- dispatch
    async def dispatch(self, state: RunState, config: RunnableConfig) -> dict[str, Any]:
        """Scheduler + join. Folds human commands into the state, then resolves everything that does
        not need an agent: cancellation, budgets, upstream-failure cascade, breakpoints, deadlocks."""
        self._bind_events()
        rt, d = self.rt(config), self.d
        work: dict[str, Any] = dict(state)
        upd: dict[str, Any] = {}

        def apply(patch: dict[str, Any]) -> None:
            fold(work, patch)
            fold(upd, patch)

        cmds = d.store.pending_commands(rt.run_id)
        for c in cmds:
            apply(c["patch"])
        if cmds:
            upd["cmd_ids"] = [c["id"] for c in cmds]

        if not work.get("abort"):
            try:
                rt.budget.check_wall_clock()
            except BudgetExceeded as e:
                apply(abort_patch(RunStatus.BUDGET_EXCEEDED, f"budget exceeded: {e.dimension}", e.dimension))
            if not work.get("abort") and (rt.cancelled or work.get("cancelled")):
                apply(abort_patch(RunStatus.CANCELLED, "cancelled by operator"))
        inj = d.injector
        if inj and inj.crash_after_tasks is not None and rt.completed_in_session >= inj.crash_after_tasks and not work.get("abort"):
            plan0 = plan_of(work)
            if plan0 and not all(work["tasks"][t.id]["status"] in TERMINAL for t in plan0.tasks):
                raise SimulatedCrash(f"injected crash after {rt.completed_in_session} completed tasks")

        plan = plan_of(work)
        if plan is not None and not work.get("abort"):
            tasks = work["tasks"]
            changed = True
            while changed:  # skip pending tasks whose required (non-optional) upstream failed
                changed = False
                for spec in plan.tasks:
                    rec = tasks[spec.id]
                    if rec["status"] in (PENDING, HELD) and dep_state(plan, tasks, spec) == "blocked":
                        bad = next(dd for dd in spec.depends_on if not is_ok(tasks[dd]) and not plan.get(dd).optional)
                        reason = f"upstream_failed:{bad}"
                        apply({"tasks": {spec.id: {"status": SKIPPED, "reason": reason, "finished_at": _now()}}})
                        tasks = work["tasks"]
                        rt.emit("task.skipped", task_id=spec.id, reason=reason)
                        changed = True
            if false_alarm(plan, tasks, rt.incident.severity.value, bool(work.get("taint") or rt.taint)):
                # conditional routing: nothing anomalous anywhere on a sev3 -> do not spend a runbook lookup on it
                for spec in plan.tasks:
                    if spec.agent is AgentName.RUNBOOK and tasks[spec.id]["status"] in (PENDING, HELD):
                        apply({"tasks": {spec.id: {"status": SKIPPED, "reason": SHORT_CIRCUIT, "finished_at": _now(), "output": {"_skipped": True, "reason": "false_alarm"}}}})
                        tasks = work["tasks"]
                        rt.emit("task.skipped", task_id=spec.id, reason=SHORT_CIRCUIT, by="router")
            if not work.get("paused"):
                for spec in plan.tasks:  # breakpoints: a ready task with an unreleased breakpoint is held
                    rec = tasks[spec.id]
                    if rec["status"] in (PENDING, HELD) and spec.breakpoint and not rec["released"] and dep_state(plan, tasks, spec) == "ready" and rec["status"] != HELD:
                        apply({"tasks": {spec.id: {"status": HELD}}})
                        tasks = work["tasks"]
                        rt.emit("task.held", task_id=spec.id)
                runnable = self._ready(plan, tasks)
                waiting = any(r["status"] == HELD for r in tasks.values())
                if not runnable and not waiting:
                    for spec in plan.tasks:  # unreachable in a validated DAG; fail closed rather than hang
                        if tasks[spec.id]["status"] == PENDING:
                            apply({"tasks": {spec.id: {"status": SKIPPED, "reason": "deadlock"}}})
                            tasks = work["tasks"]
        if rt.taint and not work.get("taint"):
            upd["taint"] = True
        rt.apply_patch(upd)
        return upd

    @staticmethod
    def _ready(plan: Any, tasks: dict[str, dict]) -> list[tuple[str, Any]]:
        """(node, spec) for every task that can run right now."""
        out = []
        for spec in plan.tasks:
            rec = tasks[spec.id]
            if rec["status"] == AWAITING and spec.agent is AgentName.REMEDIATION:
                out.append(("approval_gate", spec))
            elif rec["status"] in (PENDING, HELD) and dep_state(plan, tasks, spec) == "ready" and not (spec.breakpoint and not rec["released"]):
                out.append((spec.agent.value, spec))
        return out

    def route(self, state: RunState) -> list[Send] | str:
        """Conditional edge out of ``dispatch``: fan the plan out with ``Send``."""
        plan = plan_of(state)
        if state.get("abort") or plan is None:
            return "finalize"
        tasks = state["tasks"]
        if all(tasks[t.id]["status"] in TERMINAL for t in plan.tasks):
            return "finalize"
        if state.get("paused"):
            return "hold"
        guidance = state.get("guidance", [])
        sends: list[Send] = []
        for node, spec in self._ready(plan, tasks):
            rec = tasks[spec.id]
            base = {"task_id": spec.id, "spec": spec.model_dump(mode="json")}
            if node == "approval_gate":
                sends.append(Send(node, {**base, "draft": rec["output"]}))
            else:
                texts = [g["text"] for g in guidance if g.get("task_id") in (None, spec.id)]
                sends.append(Send(node, {**base, "params": {**spec.params, **rec["input_overrides"]}, "upstream": upstream_of(plan, tasks, spec), "guidance": texts}))
        return sends or "hold"

    async def hold(self, state: RunState) -> dict[str, Any]:
        return {}  # the graph is compiled with interrupt_before=['hold']; edits arrive via update_state

    # ------------------------------------------------------------------------------ specialists
    def agent_node(self, agent: AgentName) -> Callable[..., Any]:
        async def node(payload: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            self._bind_events()
            return await self.run_task(self.rt(config), TaskSpec.model_validate(payload["spec"]), payload)

        node.__name__ = agent.value
        return node

    async def run_task(self, rt: RunRuntime, spec: TaskSpec, payload: dict[str, Any]) -> dict[str, Any]:
        d = self.d
        ts = rt.tasks[spec.id]
        ts.status, ts.attempts = TaskStatus.RUNNING, 0
        ts.started_at = ts.started_at or _now()
        rt.persist_task(spec.id)
        rt.emit("task.started", task_id=spec.id, agent=spec.agent.value, resumed_at_gate=False)
        async with d.tracer.span(rt.run_id, "task", spec.id, task_id=spec.id, input={"agent": spec.agent.value, "params": payload["params"]}, attrs={"agent": spec.agent.value}) as sp:
            try:
                draft = await self._with_retries(rt, spec, ts, payload)
            except BudgetExceeded as e:
                sp.status = "error"
                sp.set(error=str(e)[:300])
                return self._abort_task(rt, spec, ts, e)
            except TaskFailure as tf:
                sp.status = "error"
                sp.set(error=str(tf)[:300])
                return self._dead_letter(rt, spec, ts, tf)
            except Exception as e:  # noqa: BLE001 - an unexpected bug in a node must not wedge the run
                sp.status = "error"
                return self._dead_letter(rt, spec, ts, TaskFailure(spec.id, e, ts.attempts))
            if spec.agent is AgentName.REMEDIATION and draft.get("proposed_actions"):
                # the draft is checkpointed *before* the human gate, so a crash while waiting resumes at
                # the gate - never back at the LLM
                sp.set_output({"summary": draft.get("summary"), "keys": sorted(draft), "awaiting_approval": True})
                return self._patch(rt, spec.id, {"status": AWAITING, "output": draft, "attempts": ts.attempts})
            return self._succeed(rt, spec, ts, draft, sp)

    def _succeed(self, rt: RunRuntime, spec: TaskSpec, ts: Any, out: dict[str, Any], sp: Any) -> dict[str, Any]:
        out = redact_obj(out)
        sp.set_output({"summary": out.get("summary"), "keys": sorted(out)})
        patch = self._patch(rt, spec.id, {"status": SUCCEEDED, "output": out, "attempts": ts.attempts, "error": None, "reason": None, "finished_at": _now()})
        rt.persist_run()
        rt.emit("task.succeeded", task_id=spec.id, attempts=ts.attempts, summary=str(out.get("summary", ""))[:300])
        rt.completed_in_session += 1
        return patch

    def _patch(self, rt: RunRuntime, tid: str, fields: dict[str, Any]) -> dict[str, Any]:
        patch: dict[str, Any] = {"tasks": {tid: {"started_at": rt.tasks[tid].started_at, **fields}}}
        if rt.taint:
            patch["taint"] = True
        rt.apply_patch(patch)
        return patch

    def _abort_task(self, rt: RunRuntime, spec: TaskSpec, ts: Any, e: BudgetExceeded) -> dict[str, Any]:
        why = f"budget exceeded: {e.dimension}"
        patch = self._patch(rt, spec.id, {"status": FAILED, "reason": why, "attempts": ts.attempts, "finished_at": _now()})
        return {**patch, **abort_patch(RunStatus.BUDGET_EXCEEDED, why, e.dimension)}

    def _dead_letter(self, rt: RunRuntime, spec: TaskSpec, ts: Any, tf: TaskFailure) -> dict[str, Any]:
        err = f"{type(tf.cause).__name__}: {tf.cause}"[:500]
        patch = self._patch(rt, spec.id, {"status": FAILED, "error": err, "attempts": ts.attempts, "finished_at": _now()})
        self.d.store.add_dead_letter(rt.run_id, spec.id, err, {"agent": spec.agent.value, "params": {**spec.params, **ts.input_overrides}, "attempts": ts.attempts})
        rt.emit("task.dead_lettered", task_id=spec.id, error=err, attempts=ts.attempts)
        rt.emit("task.failed", task_id=spec.id, error=err)
        return patch

    async def _with_retries(self, rt: RunRuntime, spec: TaskSpec, ts: Any, payload: dict[str, Any]) -> dict[str, Any]:
        d, cfg = self.d, self.d.cfg
        agent = AGENTS[spec.agent]
        policy = BackoffPolicy(spec.max_attempts or cfg.max_task_attempts, cfg.retry_base_delay_s, cfg.retry_max_delay_s)
        breaker = d.breakers.get(f"agent:{spec.agent.value}")
        bulkhead = d.bulkheads.get(f"agent:{spec.agent.value}")
        timeout = spec.timeout_s or cfg.task_timeout_s

        async def attempt(_n: int) -> dict[str, Any]:
            ts.attempts += 1
            rt.persist_task(spec.id)
            rt.emit("task.attempt", task_id=spec.id, attempt=ts.attempts)
            llm = StructuredLLM(llm=d.llm, cfg=cfg, tracer=d.tracer, bus=d.bus, budget=rt.budget, injector=d.injector, run_id=rt.run_id, agent=spec.agent.value)
            ctx = AgentContext(d, rt, spec, payload["params"], [tuple(u) for u in payload["upstream"]], payload["guidance"], llm, ToolBox(d, rt, spec.agent.value, spec.id))
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

    # --------------------------------------------------------------------------- approval gate
    async def approval_gate(self, payload: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        """Human approval of the remediation proposals. Calls ``interrupt()``: the graph parks here with the
        request in its checkpoint (so it survives a restart) until the driver resumes it with
        ``Command(resume=...)``. Mutating actions execute only after this node's interrupt returns."""
        self._bind_events()
        rt, d = self.rt(config), self.d
        spec = TaskSpec.model_validate(payload["spec"])
        ts = rt.tasks[spec.id]
        ts.status = TaskStatus.AWAITING_APPROVAL
        if rt.resumed and spec.id not in rt.gate_announced:
            rt.gate_announced.add(spec.id)
            rt.emit("task.started", task_id=spec.id, agent=spec.agent.value, resumed_at_gate=True)
        async with d.tracer.span(rt.run_id, "task", f"{spec.id}.gate", task_id=spec.id, input={"agent": spec.agent.value, "gate": True}, attrs={"agent": spec.agent.value}) as sp:
            try:
                out = await self.e.gate.process(rt, spec, ts, payload["draft"])
            except GraphInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 - a gate bug must not wedge the run
                sp.status = "error"
                return self._dead_letter(rt, spec, ts, TaskFailure(spec.id, e, ts.attempts))
            if rt.cancelled:
                return self._patch(rt, spec.id, {"status": FAILED, "reason": "cancelled by operator", "finished_at": _now()})
            return self._succeed(rt, spec, ts, out, sp)

    # ------------------------------------------------------------------------------- finalize
    async def finalize(self, state: RunState, config: RunnableConfig) -> dict[str, Any]:
        self._bind_events()
        rt, d = self.rt(config), self.d
        plan = plan_of(state)
        tasks = state.get("tasks", {})
        abort = state.get("abort")
        patches: dict[str, dict] = {}
        if abort:
            why, status = abort["reason"], RunStatus(abort["status"])
            for tid, rec in tasks.items():
                if rec["status"] in (AWAITING, HELD, TaskStatus.RUNNING.value):
                    patches[tid] = {"status": FAILED, "reason": why, "finished_at": _now()}
                elif rec["status"] == PENDING:
                    patches[tid] = {"status": SKIPPED, "reason": why, "finished_at": _now()}
            for ap in d.store.list_approvals(rt.run_id, "pending"):  # never leave a dangling approval
                d.store.decide_approval(ap["approval_id"], "rejected", "system:run_aborted", why)
            rt.emit("run.aborted", reason=why, dimension=abort.get("dimension"))
        else:
            assert plan is not None
            failed_required = any(r["status"] == FAILED and not plan.get(tid).optional for tid, r in tasks.items())
            degraded = any(r["status"] == FAILED or (r["status"] == SKIPPED and not deliberate_skip(r)) for r in tasks.values())
            status = RunStatus.FAILED if failed_required else RunStatus.DEGRADED if degraded else RunStatus.COMPLETED
        patch: dict[str, Any] = {"tasks": patches, "status": status.value}
        rt.apply_patch(patch)
        return patch


def configure_tracing() -> None:
    """Offline by default: LangGraph/LangChain tracing hooks (LangSmith) stay off - even if the shell exports
    ``LANGSMITH_TRACING`` - unless ``IO_LANGSMITH=1`` opts in (then LANGSMITH_API_KEY etc. apply as usual)."""
    if os.environ.get("IO_LANGSMITH") == "1":
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    else:
        for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING"):
            os.environ[var] = "false"


def build_graph(eng: Orchestrator, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    configure_tracing()
    n = GraphNodes(eng)
    g = StateGraph(RunState)
    g.add_node("supervisor", n.supervisor, retry_policy=INFRA_RETRY)
    g.add_node("dispatch", n.dispatch, retry_policy=INFRA_RETRY)
    for agent in AgentName:
        g.add_node(agent.value, n.agent_node(agent))
    g.add_node("approval_gate", n.approval_gate)
    g.add_node("hold", n.hold)
    g.add_node("finalize", n.finalize, retry_policy=INFRA_RETRY)
    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", "dispatch")
    g.add_conditional_edges("dispatch", n.route, [*AGENT_NODES, "approval_gate", "hold", "finalize"])
    for name in [*AGENT_NODES, "approval_gate", "hold"]:
        g.add_edge(name, "dispatch")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=["hold"], name="incident_orchestrator")
