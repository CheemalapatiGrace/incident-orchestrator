"""Risk-based approval gate.

The RemediationAgent only *proposes*. For every proposal the gate (1) scores risk and validates scope,
(2) blocks anything forbidden/out-of-scope outright, (3) persists an approval request and pauses the
node until a human approves / rejects / edits it, and (4) executes approved actions through the one and
only mutation path (:meth:`ToolBox.execute_approved`), which re-verifies the approval record.

Fail-safe defaults: no decision (timeout) means rejection; a rejected/blocked action is never executed.
The gate is resumable: approval ids are deterministic, decisions are persisted, and execution is
idempotent, so a crash mid-wait neither loses a pending request nor double-executes an action.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from orchestrator.core.runtime import Deps, RunRuntime, TaskState
from orchestrator.errors import ApprovalError, FatalToolError, TransientError
from orchestrator.models import TaskSpec, TaskStatus
from orchestrator.security import risk as risk_mod
from orchestrator.security.policy import GATE_PRINCIPAL
from orchestrator.tools.registry import args_hash
from orchestrator.tools.toolbox import ToolBox


def assess_for_run(rt: RunRuntime, action: str, params: dict[str, Any], confidence: float) -> risk_mod.RiskAssessment:
    env = rt.env
    target = str(params.get("service", ""))
    return risk_mod.assess(
        action, params, incident_service=rt.incident.service, service_info=env.services.get(target, {}),
        related_services=env.related_services(rt.incident.service), severity=rt.incident.severity.value,
        confidence=confidence, tainted=rt.taint,
    )


class ApprovalGate:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps

    def approval_id(self, rt: RunRuntime, spec: TaskSpec, idx: int, proposal: dict[str, Any]) -> str:
        """Deterministic id (survives restarts); a re-run that proposes something different gets a new id."""
        base = f"{rt.run_id}:{spec.id}:{idx}"
        n = 0
        while True:
            aid = base if n == 0 else f"{base}:r{n}"
            row = self.deps.store.get_approval(aid)
            if row is None or (
                row["proposal"].get("action") == proposal["action"]
                and args_hash(row["proposal"].get("params", {})) == args_hash(proposal.get("params", {}))
            ):
                return aid
            n += 1

    async def process(self, rt: RunRuntime, spec: TaskSpec, ts: TaskState, draft: dict[str, Any]) -> dict[str, Any]:
        proposals = list(draft.get("proposed_actions", []))
        if not proposals:
            return draft
        confidence = float(draft.get("root_cause", {}).get("confidence", 0.0))
        # persist the pre-gate draft so a crash while waiting resumes at the gate, not at the LLM
        ts.output = draft
        out_actions = []
        for idx, p in enumerate(proposals):
            out_actions.append(await self._handle(rt, spec, ts, idx, p, confidence))
        if ts.status is TaskStatus.AWAITING_APPROVAL:
            ts.status = TaskStatus.RUNNING
            rt.persist_task(spec.id)
        return {**draft, "proposed_actions": out_actions}

    async def _handle(self, rt: RunRuntime, spec: TaskSpec, ts: TaskState, idx: int, p: dict[str, Any], confidence: float) -> dict[str, Any]:
        d = self.deps
        action, params = p["action"], dict(p.get("params", {}))
        risk = assess_for_run(rt, action, params, confidence)
        result: dict[str, Any] = {**p, "params": params, "risk": risk.as_dict(), "approval_id": None}
        if risk.blocked:
            d.bus.publish(rt.run_id, "guardrail.action_blocked", {"action": action, "params": params, "reason": risk.block_reason, "task_id": spec.id})
            d.store.audit(rt.run_id, "system:risk_engine", "action_blocked", action, {"reason": risk.block_reason, "params": params})
            return {**result, "status": "blocked", "decided_by": "system:risk_engine", "note": risk.block_reason}

        aid = self.approval_id(rt, spec, idx, p)
        result["approval_id"] = aid
        if d.store.create_approval(aid, rt.run_id, spec.id, {"action": action, "params": params, "rationale": p.get("rationale", ""), "expected_effect": p.get("expected_effect", "")}, risk.as_dict()):
            d.bus.publish(rt.run_id, "approval.requested", {"approval_id": aid, "task_id": spec.id, "action": action, "params": params, "risk": risk.as_dict(), "rationale": p.get("rationale", "")})
            d.store.audit(rt.run_id, "system:gate", "approval_requested", aid, {"action": action, "risk": risk.level.value})
        row = d.store.get_approval(aid)
        assert row is not None
        if row["status"] == "pending":
            ts.status = TaskStatus.AWAITING_APPROVAL
            rt.persist_task(spec.id)
            rt.persist_run()
            rt.kick()
            row = await self._wait(rt, aid)
            ts.status = TaskStatus.RUNNING
            rt.persist_task(spec.id)
        result["decided_by"] = row["decided_by"]
        result["note"] = row["note"]
        if row["status"] == "rejected":
            return {**result, "status": "rejected"}

        final_params = row["final_params"] if row["final_params"] is not None else params
        # an edit is re-validated by the same guardrails (the controller also checks, this is the backstop)
        recheck = assess_for_run(rt, action, final_params, confidence)
        if recheck.blocked:
            d.bus.publish(rt.run_id, "guardrail.action_blocked", {"action": action, "params": final_params, "reason": recheck.block_reason, "task_id": spec.id})
            return {**result, "status": "blocked", "note": recheck.block_reason}
        result["params"] = final_params
        result["risk"] = recheck.as_dict()
        await rt.unpaused.wait()  # a paused run never mutates production
        box = ToolBox(d, rt, GATE_PRINCIPAL, spec.id)
        try:
            res = await box.execute_approved(aid, action, final_params)
        except (ApprovalError, FatalToolError, TransientError, TimeoutError) as e:
            d.bus.publish(rt.run_id, "action.failed", {"approval_id": aid, "action": action, "error": str(e)[:300]})
            return {**result, "status": "failed", "error": str(e)[:300]}
        d.bus.publish(rt.run_id, "action.executed", {"approval_id": aid, "action": action, "params": final_params, "result": res})
        return {**result, "status": "executed", "result": res}

    async def _wait(self, rt: RunRuntime, aid: str) -> dict[str, Any]:
        d = self.deps
        ev = rt.approval_events.setdefault(aid, asyncio.Event())
        timeout = d.cfg.approval_timeout_s
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            row = d.store.get_approval(aid)
            assert row is not None
            if row["status"] != "pending":
                return row
            if deadline is not None and time.monotonic() >= deadline:
                if d.store.decide_approval(aid, "rejected", "system:timeout", "no human decision before the approval timeout (fail-safe reject)"):
                    d.store.audit(rt.run_id, "system:gate", "approval_timeout", aid, {"timeout_s": timeout})
                    d.bus.publish(rt.run_id, "approval.decided", {"approval_id": aid, "status": "rejected", "actor": "system:timeout"})
                continue
            if rt.cancelled:
                d.store.decide_approval(aid, "rejected", "system:cancelled", "run cancelled")
                continue
            ev.clear()
            try:
                await asyncio.wait_for(ev.wait(), d.cfg.approval_poll_s)
            except TimeoutError:
                pass
