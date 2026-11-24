"""Human-in-the-loop controller. Every action is validated, applied to the live run, persisted, emitted
as an event and written to the audit log with the acting principal.

Supported interventions: approve / reject / edit-and-approve an action, pause / resume, retry or skip
a node, release a held node (breakpoint), edit a node's input or the plan before it executes, inject
operator guidance, cancel.

Approve / reject act on the durable ``approvals`` row (any process may decide); the graph driver notices
and resumes the interrupted gate with ``Command(resume=...)``. Every other intervention is validated
against the run's live view, applied to that view at once, and queued as an absolute *state patch*
(``Store.add_command``). The driver folds queued patches into the LangGraph state with
``graph.update_state(...)`` while the graph is parked, or the ``dispatch`` node folds them at the next
superstep boundary - so a human never races a running superstep.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from orchestrator.core.dag import descendants, validate_plan
from orchestrator.core.runtime import RunRuntime, TaskState
from orchestrator.core.state import new_task
from orchestrator.errors import ApprovalError, InvalidState, PlanValidationError
from orchestrator.hitl.approvals import assess_for_run
from orchestrator.models import (
    ACTIVE_TASK_STATES,
    TERMINAL_RUN_STATES,
    AgentName,
    Plan,
    RunStatus,
    TaskSpec,
    TaskStatus,
)
from orchestrator.security.redaction import redact_obj
from orchestrator.tools.registry import canonical

if TYPE_CHECKING:
    from orchestrator.engine import Orchestrator

EDITABLE_TASK_FIELDS = {"description", "params", "timeout_s", "max_attempts", "optional", "breakpoint"}
MAX_GUIDANCE_CHARS = 1000


class HitlController:
    def __init__(self, engine: Orchestrator) -> None:
        self.e = engine

    # ------------------------------------------------------------------ helpers
    def _log(self, rt: RunRuntime, actor: str, action: str, target: str = "", **detail: Any) -> None:
        detail = redact_obj(detail)
        self.e.store.audit(rt.run_id, actor, action, target, detail)
        rt.emit("human.action", actor=actor, action=action, target=target, detail=detail)

    def _rt(self, run_id: str) -> RunRuntime:
        return self.e.get_runtime(run_id)

    def _task(self, rt: RunRuntime, task_id: str) -> TaskState:
        if task_id not in rt.tasks:
            raise InvalidState(f"unknown task {task_id!r}")
        return rt.tasks[task_id]

    def _queue(self, rt: RunRuntime, op: str, patch: dict[str, Any]) -> None:
        self.e.store.add_command(rt.run_id, op, patch)

    def _reopen(self, rt: RunRuntime) -> None:
        """A human action created new work on a finished run: reopen it and re-schedule."""
        if rt.status in TERMINAL_RUN_STATES:
            rt.set_status(RunStatus.RUNNING)
        if not rt.active:
            self.e.kick(rt.run_id)
        rt.kick()

    # ------------------------------------------------------------ run-level controls
    def pause(self, run_id: str, actor: str, reason: str = "") -> None:
        rt = self._rt(run_id)
        if rt.status in TERMINAL_RUN_STATES:
            raise InvalidState("run already finished")
        rt.set_paused(True)
        rt.persist_run()
        self._queue(rt, "pause", {"paused": True})
        self._log(rt, actor, "pause", run_id, reason=reason)
        rt.emit("run.paused")
        rt.kick()

    def resume(self, run_id: str, actor: str) -> None:
        rt = self._rt(run_id)
        if not rt.paused:
            raise InvalidState("run is not paused")
        rt.set_paused(False)
        rt.persist_run()
        self._queue(rt, "resume", {"paused": False})
        self._log(rt, actor, "resume", run_id)
        rt.emit("run.resumed")
        if not rt.active and rt.status not in TERMINAL_RUN_STATES:
            self.e.kick(run_id)  # e.g. paused, then the worker restarted
        rt.kick()

    def cancel(self, run_id: str, actor: str) -> None:
        rt = self._rt(run_id)
        if rt.status in TERMINAL_RUN_STATES:
            raise InvalidState("run already finished")
        rt.cancelled = True
        self._queue(rt, "cancel", {"cancelled": True})
        self._log(rt, actor, "cancel", run_id)
        if rt.active:
            rt.kick()
        else:
            rt.set_status(RunStatus.CANCELLED)

    # ---------------------------------------------------------------- approvals
    def _pending_approval(self, approval_id: str) -> dict[str, Any]:
        row = self.e.store.get_approval(approval_id)
        if row is None:
            raise ApprovalError(f"unknown approval {approval_id!r}")
        if row["status"] != "pending":
            raise ApprovalError(f"approval {approval_id!r} already {row['status']} by {row['decided_by']}")
        return row

    def approve(self, approval_id: str, actor: str, note: str = "", edits: dict[str, Any] | None = None) -> dict[str, Any]:
        row = self._pending_approval(approval_id)
        rt = self._rt(row["run_id"])
        proposal = row["proposal"]
        params = {**proposal["params"], **(edits or {})}
        edited = canonical(params) != canonical(proposal["params"])
        confidence = self._confidence(rt, row["task_id"])
        assessment = assess_for_run(rt, proposal["action"], params, confidence)
        if assessment.blocked:
            self._log(rt, actor, "approve_refused", approval_id, reason=assessment.block_reason)
            raise ApprovalError(f"edited action violates guardrails: {assessment.block_reason}")
        if (row["risk"].get("requires_note") or assessment.requires_note) and not note.strip():
            raise ApprovalError("high-risk action: a justification note is required to approve")
        status = "edited" if edited else "approved"
        if not self.e.store.decide_approval(approval_id, status, actor, note, params):
            raise ApprovalError(f"approval {approval_id!r} was decided concurrently")
        self._log(rt, actor, "approve_edited" if edited else "approve", approval_id, note=note, params=params, original=proposal["params"] if edited else None)
        rt.emit("approval.decided", approval_id=approval_id, status=status, actor=actor, note=note, params=params)
        self._poke(rt, approval_id)
        return {"approval_id": approval_id, "status": status, "params": params}

    def reject(self, approval_id: str, actor: str, note: str = "") -> dict[str, Any]:
        row = self._pending_approval(approval_id)
        rt = self._rt(row["run_id"])
        if not self.e.store.decide_approval(approval_id, "rejected", actor, note):
            raise ApprovalError(f"approval {approval_id!r} was decided concurrently")
        self._log(rt, actor, "reject", approval_id, note=note)
        rt.emit("approval.decided", approval_id=approval_id, status="rejected", actor=actor, note=note)
        self._poke(rt, approval_id)
        return {"approval_id": approval_id, "status": "rejected"}

    def _poke(self, rt: RunRuntime, approval_id: str) -> None:
        rt.kick()  # wake the driver; another process's driver polls the shared approvals table

    def _confidence(self, rt: RunRuntime, task_id: str) -> float:
        ts = rt.tasks.get(task_id)
        return float((ts.output or {}).get("root_cause", {}).get("confidence", 0.0)) if ts else 0.0

    # ------------------------------------------------------------- node controls
    def retry_task(self, run_id: str, task_id: str, actor: str, reset_downstream: bool = True) -> list[str]:
        rt = self._rt(run_id)
        assert rt.plan
        self._task(rt, task_id)
        targets = {task_id} | (descendants(rt.plan, task_id) if reset_downstream else set())
        busy = [t for t in targets if rt.tasks[t].status in ACTIVE_TASK_STATES]
        if busy:
            raise InvalidState(f"cannot retry while {busy} are running; pause or wait first")
        reset = {"status": TaskStatus.PENDING.value, "attempts": 0, "output": None, "error": None, "reason": None, "finished_at": None, "human_skipped": False}
        for t in sorted(targets):
            ts = rt.tasks[t]
            ts.status, ts.attempts, ts.output, ts.error, ts.reason, ts.finished_at, ts.human_skipped = TaskStatus.PENDING, 0, None, None, None, None, False
            rt.persist_task(t)
        self._queue(rt, "retry_task", {"tasks": {t: dict(reset) for t in sorted(targets)}, "abort": {}})
        self._log(rt, actor, "retry_task", task_id, reset=sorted(targets))
        rt.emit("task.reset", tasks=sorted(targets))
        self._reopen(rt)
        return sorted(targets)

    def skip_task(self, run_id: str, task_id: str, actor: str, output: dict[str, Any] | None = None, reason: str = "") -> None:
        rt = self._rt(run_id)
        ts = self._task(rt, task_id)
        if ts.status in ACTIVE_TASK_STATES or ts.status is TaskStatus.SUCCEEDED:
            raise InvalidState(f"cannot skip a task that is {ts.status.value}")
        ts.status, ts.human_skipped, ts.reason, ts.finished_at = TaskStatus.SKIPPED, True, reason or "skipped by operator", time.time()
        ts.output = redact_obj(output) if output else {"_skipped": True, "reason": ts.reason}
        rt.persist_task(task_id)
        self._queue(rt, "skip_task", {"tasks": {task_id: {"status": TaskStatus.SKIPPED.value, "human_skipped": True, "reason": ts.reason, "finished_at": ts.finished_at, "output": ts.output}}})
        self._log(rt, actor, "skip_task", task_id, reason=reason, substitute_output=bool(output))
        rt.emit("task.skipped", task_id=task_id, reason=ts.reason, by="human")
        self._reopen(rt)

    def release_task(self, run_id: str, task_id: str, actor: str) -> None:
        rt = self._rt(run_id)
        ts = self._task(rt, task_id)
        if ts.status is not TaskStatus.HELD:
            raise InvalidState(f"task is {ts.status.value}, not held")
        ts.released = True
        rt.persist_task(task_id)
        self._queue(rt, "release_task", {"tasks": {task_id: {"released": True}}})
        self._log(rt, actor, "release_task", task_id)
        rt.kick()

    def edit_task_input(self, run_id: str, task_id: str, actor: str, params: dict[str, Any]) -> dict[str, Any]:
        rt = self._rt(run_id)
        ts = self._task(rt, task_id)
        if ts.status in ACTIVE_TASK_STATES:
            raise InvalidState("task is executing; edit before it starts or after it finishes (then retry)")
        if len(canonical(params)) > 4000:
            raise InvalidState("params too large")
        ts.input_overrides = {**ts.input_overrides, **redact_obj(params)}
        rt.persist_task(task_id)
        self._queue(rt, "edit_task_input", {"tasks": {task_id: {"input_overrides": dict(ts.input_overrides)}}})
        self._log(rt, actor, "edit_task_input", task_id, params=params)
        rt.emit("task.input_edited", task_id=task_id, overrides=ts.input_overrides)
        return ts.input_overrides

    def edit_plan(self, run_id: str, actor: str, ops: list[dict[str, Any]]) -> Plan:
        rt = self._rt(run_id)
        if rt.plan is None:
            raise InvalidState("run has no plan yet")
        specs = {t.id: t.model_copy(deep=True) for t in rt.plan.tasks}
        order = [t.id for t in rt.plan.tasks]
        added: list[str] = []
        removed: list[str] = []

        def mutable(tid: str) -> None:
            if tid not in specs:
                raise PlanValidationError([f"unknown task {tid!r}"])
            if rt.tasks[tid].status not in (TaskStatus.PENDING, TaskStatus.HELD):
                raise InvalidState(f"task {tid!r} is {rt.tasks[tid].status.value}; only not-yet-started tasks can be edited")

        for op in ops:
            kind = op.get("op")
            if kind == "add_task":
                spec = TaskSpec.model_validate(op["task"])
                if spec.id in specs:
                    raise PlanValidationError([f"task {spec.id!r} already exists"])
                specs[spec.id] = spec
                order.append(spec.id)
                added.append(spec.id)
            elif kind == "remove_task":
                mutable(op["id"])
                del specs[op["id"]]
                order.remove(op["id"])
                removed.append(op["id"])
                for s in specs.values():
                    s.depends_on = [d for d in s.depends_on if d != op["id"]]
            elif kind == "set_dependencies":
                mutable(op["id"])
                specs[op["id"]].depends_on = list(op["depends_on"])
            elif kind == "update_task":
                mutable(op["id"])
                fields = op.get("fields", {})
                bad = set(fields) - EDITABLE_TASK_FIELDS
                if bad:
                    raise PlanValidationError([f"fields not editable: {sorted(bad)}"])
                specs[op["id"]] = TaskSpec.model_validate({**specs[op["id"]].model_dump(), **fields})
            else:
                raise PlanValidationError([f"unknown plan operation {kind!r}"])
        new_plan = Plan(tasks=[specs[i] for i in order], rationale=rt.plan.rationale)
        validate_plan(new_plan)  # raises -> nothing has been applied
        # a task that is already done must not gain unfinished prerequisites it never saw
        for tid in set(specs) & {t for t, s in rt.tasks.items() if s.status in (TaskStatus.SUCCEEDED, TaskStatus.RUNNING, TaskStatus.AWAITING_APPROVAL)}:
            old = next((t for t in rt.plan.tasks if t.id == tid), None)
            if old and set(specs[tid].depends_on) != set(old.depends_on) and tid not in removed:
                raise InvalidState(f"task {tid!r} already started; its dependencies cannot change")
        rt.plan = new_plan
        for tid in removed:
            rt.tasks.pop(tid, None)
            self.e.store.delete_task(rt.run_id, tid)
        for tid in added:
            rt.tasks[tid] = TaskState(tid)
            rt.persist_task(tid)
        rt.persist_run()
        self._queue(rt, "edit_plan", {"plan": new_plan.model_dump(mode="json"), "tasks": {**dict.fromkeys(removed), **{tid: new_task() for tid in added}}})
        self._log(rt, actor, "edit_plan", run_id, ops=ops)
        rt.emit("plan.edited", plan=new_plan.model_dump(mode="json"), added=added, removed=removed)
        self._reopen(rt) if added else rt.kick()
        return new_plan

    def set_breakpoint(self, run_id: str, task_id: str, actor: str, on: bool = True) -> None:
        self.edit_plan(run_id, actor, [{"op": "update_task", "id": task_id, "fields": {"breakpoint": on}}])

    def inject_guidance(self, run_id: str, actor: str, text: str, task_id: str | None = None) -> dict[str, Any]:
        rt = self._rt(run_id)
        text = redact_obj(text.strip())[:MAX_GUIDANCE_CHARS]
        if not text:
            raise InvalidState("guidance is empty")
        if task_id is not None:
            self._task(rt, task_id)
        g = {"id": uuid.uuid4().hex[:8], "actor": actor, "text": text, "task_id": task_id, "ts": time.time()}
        rt.guidance.append(g)
        rt.persist_run()
        self._queue(rt, "inject_guidance", {"guidance": [g]})
        self._log(rt, actor, "inject_guidance", task_id or "*", text=text)
        rt.emit("guidance.added", guidance=g)
        return g


__all__ = ["HitlController", "AgentName"]
