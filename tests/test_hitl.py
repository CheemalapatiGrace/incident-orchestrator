"""Human-in-the-loop: approval gates, pause/resume, retry/skip, input/plan edits, guidance, audit."""

import asyncio

import pytest

from conftest import (
    alert_for,
    approve_when_pending,
    events,
    make_engine,
    run_with_approval,
    wait_pending,
    wait_until,
)
from orchestrator.errors import ApprovalError, InvalidState, PlanValidationError
from orchestrator.faults import FaultInjector, FaultRule
from orchestrator.models import RunStatus, TaskStatus


async def start(eng, sid, **kw):
    rid = eng.submit(alert_for(sid), **kw)
    return rid, asyncio.create_task(eng.execute(rid))


# ------------------------------------------------------------------------------ approval gate
async def test_gate_pauses_the_run_until_a_human_decides():
    eng = make_engine()
    rid, task = await start(eng, "search-traffic-spike")
    (ap,) = await wait_pending(eng, rid)
    rt = eng.runtimes[rid]
    await asyncio.sleep(0.05)
    assert rt.status is RunStatus.AWAITING_APPROVAL
    assert rt.tasks["remediation"].status is TaskStatus.AWAITING_APPROVAL
    assert rt.tasks["comms"].status is TaskStatus.PENDING  # downstream waits for the human
    assert rt.env.mutations == [] and not task.done()
    eng.hitl.approve(ap["approval_id"], "alice", "capacity needed")
    res = await task
    assert res.status is RunStatus.COMPLETED and res.actions[0]["status"] == "executed"
    assert res.actions[0]["decided_by"] == "alice"
    assert rt.env.mutations[0]["params"]["replicas"] == 12


async def test_high_risk_approval_requires_a_justification_note():
    eng = make_engine()
    rid, task = await start(eng, "checkout-bad-deploy")  # sev1 + critical tier => high risk
    (ap,) = await wait_pending(eng, rid)
    assert ap["risk"]["level"] == "high" and ap["risk"]["requires_note"]
    with pytest.raises(ApprovalError, match="justification"):
        eng.hitl.approve(ap["approval_id"], "alice", "")
    assert eng.store.get_approval(ap["approval_id"])["status"] == "pending"
    eng.hitl.approve(ap["approval_id"], "alice", "deploy dep-9812 matches the error onset")
    assert (await task).status is RunStatus.COMPLETED


async def test_reject_prevents_execution_and_is_reflected_in_comms():
    eng = make_engine()
    rid, task = await start(eng, "payments-memory-leak")
    (ap,) = await wait_pending(eng, rid)
    eng.hitl.reject(ap["approval_id"], "bob", "restart during settlement window is unacceptable")
    res = await task
    assert res.status is RunStatus.COMPLETED and eng._envs[rid].mutations == []
    assert res.actions[0]["status"] == "rejected" and res.actions[0]["decided_by"] == "bob"
    assert "rejected by bob" in res.comms["body"]


async def test_edited_approval_executes_the_edited_parameters_and_is_audited():
    eng = make_engine()
    rid, task = await start(eng, "search-traffic-spike")
    (ap,) = await wait_pending(eng, rid)
    out = eng.hitl.approve(ap["approval_id"], "alice", "10 is enough", edits={"replicas": 10})
    res = await task
    assert out["status"] == "edited" and res.actions[0]["status"] == "executed"
    assert eng._envs[rid].mutations[0]["params"]["replicas"] == 10
    entry = next(a for a in eng.store.get_audit(rid) if a["action"] == "approve_edited")
    assert entry["detail"]["original"]["replicas"] == 12 and entry["detail"]["params"]["replicas"] == 10


async def test_edit_that_violates_guardrails_is_refused_and_request_stays_pending():
    eng = make_engine()
    rid, task = await start(eng, "search-traffic-spike")
    (ap,) = await wait_pending(eng, rid)
    with pytest.raises(ApprovalError, match="guardrails"):
        eng.hitl.approve(ap["approval_id"], "mallory", "more!", edits={"replicas": 5000})
    with pytest.raises(ApprovalError, match="guardrails"):
        eng.hitl.approve(ap["approval_id"], "mallory", "wrong target", edits={"service": "payments-db"})
    assert eng.store.get_approval(ap["approval_id"])["status"] == "pending"
    eng.hitl.approve(ap["approval_id"], "alice", "ok")
    assert (await task).status is RunStatus.COMPLETED


async def test_an_approval_can_only_be_decided_once():
    eng = make_engine()
    rid, task = await start(eng, "search-traffic-spike")
    (ap,) = await wait_pending(eng, rid)
    eng.hitl.approve(ap["approval_id"], "alice", "ok")
    with pytest.raises(ApprovalError, match="already"):
        eng.hitl.reject(ap["approval_id"], "bob", "too late")
    await task
    assert len(eng._envs[rid].mutations) == 1


async def test_unanswered_approval_times_out_to_a_safe_rejection():
    eng = make_engine(approval_timeout_s=0.1)
    rid = eng.submit(alert_for("inventory-disk-full"))
    res = await eng.execute(rid)
    assert res.status is RunStatus.COMPLETED and eng._envs[rid].mutations == []
    ap = eng.store.list_approvals(rid)[0]
    assert ap["status"] == "rejected" and ap["decided_by"] == "system:timeout"


async def test_approval_works_across_processes_via_the_shared_database(tmp_path):
    db = str(tmp_path / "x.db")
    e1 = make_engine(db_path=db)
    rid, task = await start(e1, "inventory-disk-full")
    (ap,) = await wait_pending(e1, rid)
    e2 = make_engine(db_path=db)  # e.g. `python -m orchestrator approvals approve ...` in another shell
    e2.hitl.approve(ap["approval_id"], "cli:alice", "checked volume growth")
    res = await asyncio.wait_for(task, 5)
    assert res.status is RunStatus.COMPLETED and e1._envs[rid].mutations[0]["action"] == "expand_volume"


# ---------------------------------------------------------------------------- pause / resume
async def test_pause_stops_new_work_and_resume_continues_correctly():
    eng = make_engine()
    rid = eng.submit(alert_for("payments-memory-leak"))
    eng.hitl.pause(rid, "alice", "change freeze")
    task = asyncio.create_task(run_with_approval(eng, rid))
    await asyncio.sleep(0.15)
    assert not events(eng, rid, "task.started") and eng.runtimes[rid].status is RunStatus.PAUSED
    eng.hitl.resume(rid, "alice")
    res = await asyncio.wait_for(task, 5)
    assert res.status is RunStatus.COMPLETED and res.root_cause == "memory_leak"
    actions = [(a["actor"], a["action"]) for a in eng.store.get_audit(rid) if a["actor"] == "alice"]
    assert actions == [("alice", "pause"), ("alice", "resume")]


async def test_paused_run_never_mutates_even_if_approved_meanwhile():
    eng = make_engine()
    rid, task = await start(eng, "inventory-disk-full")
    (ap,) = await wait_pending(eng, rid)
    eng.hitl.pause(rid, "alice")
    eng.hitl.approve(ap["approval_id"], "bob", "approved while paused")
    await asyncio.sleep(0.15)
    assert eng._envs[rid].mutations == [] and not task.done()
    eng.hitl.resume(rid, "alice")
    res = await asyncio.wait_for(task, 5)
    assert res.status is RunStatus.COMPLETED and len(eng._envs[rid].mutations) == 1


async def test_wall_clock_budget_does_not_burn_while_paused():
    from orchestrator.models import Budget

    eng = make_engine()
    rid = eng.submit(alert_for("payments-memory-leak"), budget=Budget(max_wall_clock_s=2.0))
    eng.hitl.pause(rid, "alice")
    task = asyncio.create_task(run_with_approval(eng, rid))
    await asyncio.sleep(2.3)  # longer than the whole wall-clock budget
    eng.hitl.resume(rid, "alice")
    res = await asyncio.wait_for(task, 5)
    assert res.status is RunStatus.COMPLETED


async def test_cancel_aborts_run_and_rejects_dangling_approvals():
    eng = make_engine()
    rid, task = await start(eng, "search-traffic-spike")
    await wait_pending(eng, rid)
    eng.hitl.cancel(rid, "alice")
    res = await asyncio.wait_for(task, 5)
    assert res.status is RunStatus.CANCELLED and eng._envs[rid].mutations == []
    assert eng.store.list_approvals(rid)[0]["status"] == "rejected"


# ------------------------------------------------------------------------ retry / skip / edit
async def test_retry_a_dead_lettered_node_and_run_recovers_without_double_execution():
    inj = FaultInjector([FaultRule("llm:metrics_analyst", "error", 3)])  # exhausts all 3 task attempts
    eng = make_engine(injector=inj)
    rid = eng.submit(alert_for("payments-memory-leak"))
    res = await run_with_approval(eng, rid)
    assert res.status is RunStatus.DEGRADED and eng.store.get_tasks(rid)["metrics"]["status"] == "failed"
    approver = asyncio.create_task(approve_when_pending(eng, rid))
    reset = eng.hitl.retry_task(rid, "metrics", "alice")
    assert {"metrics", "runbook", "remediation", "comms"} <= set(reset)
    await eng.wait_for_terminal(rid)
    approver.cancel()
    tasks = eng.store.get_tasks(rid)
    assert eng.store.get_run(rid)["status"] == "completed"
    assert tasks["metrics"]["status"] == "succeeded" and tasks["remediation"]["status"] == "succeeded"
    assert len(eng._envs[rid].mutations) == 1  # re-running remediation did not restart the service twice
    assert any(a["action"] == "retry_task" and a["actor"] == "alice" for a in eng.store.get_audit(rid))


async def test_cannot_retry_a_running_node():
    eng = make_engine(latency_s=0.05)
    rid, task = await start(eng, "search-traffic-spike")
    await wait_until(lambda: eng.runtimes.get(rid) and eng.runtimes[rid].tasks.get("logs") and eng.runtimes[rid].tasks["logs"].status is TaskStatus.RUNNING)
    with pytest.raises(InvalidState):
        eng.hitl.retry_task(rid, "logs", "alice")
    eng.hitl.cancel(rid, "alice")
    await task


async def test_skip_a_node_with_substitute_output_and_run_continues():
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"), breakpoints=["deploys"])
    task = asyncio.create_task(run_with_approval(eng, rid))
    await wait_until(lambda: eng.runtimes.get(rid) and eng.runtimes[rid].tasks["deploys"].status is TaskStatus.HELD)
    eng.hitl.skip_task(rid, "deploys", "alice", output={"suspected_causes": [{"category": "bad_deploy", "confidence": 0.9, "reason": "operator: dep-9812"}], "suspect_change": {"id": "dep-9812", "previous_version": "v2.13.2"}}, reason="deploy tool is down")
    res = await asyncio.wait_for(task, 5)
    assert res.status is RunStatus.COMPLETED  # a deliberate skip is not a degradation
    assert res.root_cause == "bad_deploy" and eng._envs[rid].mutations[0]["params"]["to_version"] == "v2.13.2"
    assert eng.store.get_tasks(rid)["deploys"]["status"] == "skipped"


async def test_breakpoint_holds_node_until_input_is_edited_and_released():
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"), breakpoints=["logs"])
    task = asyncio.create_task(run_with_approval(eng, rid))
    await wait_until(lambda: eng.runtimes.get(rid) and eng.runtimes[rid].tasks["logs"].status is TaskStatus.HELD)
    assert not any(c["tool"] == "search_logs" for c in eng.runtimes[rid].env.calls)  # not executed yet
    eng.hitl.edit_task_input(rid, "logs", "alice", {"query": "NullPointerException"})
    eng.hitl.release_task(rid, "logs", "alice")
    res = await asyncio.wait_for(task, 5)
    assert res.status is RunStatus.COMPLETED
    custom = [c for c in eng._envs[rid].calls if c["tool"] == "search_logs" and c["args"]["query"] == "NullPointerException"]
    assert len(custom) == 1  # the human-edited input was used


async def test_edit_plan_before_execution_adds_and_removes_nodes_and_run_completes():
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"))
    rt = eng.get_runtime(rid)
    await eng._ensure_plan(rt)
    eng.hitl.pause(rid, "alice")
    plan = eng.hitl.edit_plan(rid, "alice", [
        {"op": "remove_task", "id": "runbook"},
        {"op": "add_task", "task": {"id": "logs_timeouts", "agent": "log_analyst", "params": {"service": "checkout-api", "query": "timeout"}}},
        {"op": "set_dependencies", "id": "remediation", "depends_on": ["logs", "metrics", "deploys", "logs_timeouts"]},
    ])
    assert "runbook" not in [t.id for t in plan.tasks] and "logs_timeouts" in [t.id for t in plan.tasks]
    assert events(eng, rid, "plan.edited")
    eng.hitl.resume(rid, "alice")
    res = await run_with_approval(eng, rid)
    tasks = eng.store.get_tasks(rid)
    assert res.status is RunStatus.COMPLETED and "runbook" not in tasks and tasks["logs_timeouts"]["status"] == "succeeded"
    assert res.root_cause == "bad_deploy"


async def test_invalid_or_late_plan_edits_are_rejected_and_leave_the_plan_untouched():
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"))
    rt = eng.get_runtime(rid)
    await eng._ensure_plan(rt)
    before = rt.plan.model_dump()
    with pytest.raises(PlanValidationError, match="cycle"):
        eng.hitl.edit_plan(rid, "alice", [{"op": "set_dependencies", "id": "logs", "depends_on": ["comms"]}])
    with pytest.raises(PlanValidationError):
        eng.hitl.edit_plan(rid, "alice", [{"op": "add_task", "task": {"id": "x", "agent": "log_analyst", "depends_on": ["ghost"]}}])
    assert rt.plan.model_dump() == before
    await run_with_approval(eng, rid)
    with pytest.raises(InvalidState, match="not-yet-started"):
        eng.hitl.edit_plan(rid, "alice", [{"op": "remove_task", "id": "logs"}])


async def test_operator_guidance_changes_the_remediation_and_reaches_comms():
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"), breakpoints=["remediation"])
    task = asyncio.create_task(run_with_approval(eng, rid))
    await wait_until(lambda: eng.runtimes.get(rid) and eng.runtimes[rid].tasks["remediation"].status is TaskStatus.HELD)
    eng.hitl.inject_guidance(rid, "alice", "Avoid rollback: the release contains a schema migration")
    eng.hitl.release_task(rid, "remediation", "alice")
    res = await asyncio.wait_for(task, 5)
    assert res.root_cause == "bad_deploy"
    assert [a["action"] for a in res.actions] == ["restart_service"]  # rollback excluded by the human
    assert "Operator note: Avoid rollback" in res.comms["body"]
    assert any(a["action"] == "inject_guidance" and a["actor"] == "alice" for a in eng.store.get_audit(rid))


async def test_reopening_a_finished_run_with_retry_works():
    eng = make_engine()
    rid = eng.submit(alert_for("auth-config-regression"))
    await run_with_approval(eng, rid)
    approver = asyncio.create_task(approve_when_pending(eng, rid))
    eng.hitl.edit_task_input(rid, "logs", "alice", {"window_minutes": 30})
    eng.hitl.retry_task(rid, "logs", "alice")
    await wait_until(lambda: eng.runtimes[rid].status is RunStatus.RUNNING or len(events(eng, rid, "run.completed")) >= 2, 3)
    await eng.wait_for_terminal(rid)
    approver.cancel()
    assert eng.store.get_run(rid)["status"] == "completed" and len(events(eng, rid, "run.completed")) == 2
    assert len(eng._envs[rid].mutations) == 1


async def test_every_human_action_is_audited_with_actor_and_streamed():
    eng = make_engine()
    rid, task = await start(eng, "search-traffic-spike")
    (ap,) = await wait_pending(eng, rid)
    eng.hitl.inject_guidance(rid, "carol", "keep an eye on cost")
    eng.hitl.pause(rid, "dave")
    eng.hitl.resume(rid, "dave")
    eng.hitl.approve(ap["approval_id"], "erin", "go")
    await task
    audit = [(a["actor"], a["action"]) for a in eng.store.get_audit(rid) if not a["actor"].startswith(("system", "gate", "agent"))]
    assert audit == [("carol", "inject_guidance"), ("dave", "pause"), ("dave", "resume"), ("erin", "approve")]
    assert len(events(eng, rid, "human.action")) == 4
