import asyncio

import pytest

from conftest import alert_for, events, make_engine, run_with_approval
from orchestrator.faults import FaultInjector, FaultRule
from orchestrator.models import RunStatus
from orchestrator.sim.scenarios import SCENARIO_IDS, build_scenario


@pytest.mark.parametrize("sid", SCENARIO_IDS)
async def test_every_scenario_finds_root_cause_and_applies_approved_fix(sid):
    s = build_scenario(sid)
    eng = make_engine()
    rid = eng.submit(s.alert())
    res = await run_with_approval(eng, rid)
    assert res.status is RunStatus.COMPLETED
    assert res.root_cause == s.root_cause
    env = eng._envs[rid]
    assert [(m["action"], m["params"]["service"]) for m in env.mutations] == [(a["action"], a["service"]) for a in s.correct_actions]
    assert res.comms and "update" in res.comms["title"]


async def test_triage_tasks_run_in_parallel():
    eng = make_engine(latency_s=0.05)
    rid = eng.submit(alert_for("search-traffic-spike"))
    await run_with_approval(eng, rid)
    spans = {s["name"]: s for s in eng.store.get_spans(rid) if s["kind"] == "task"}
    trio = [spans["logs"], spans["metrics"], spans["deploys"]]
    assert max(s["start_ts"] for s in trio) < min(s["end_ts"] for s in trio)  # all three overlap
    assert spans["remediation"]["start_ts"] >= max(s["end_ts"] for s in trio + [spans["runbook"]]) - 1e-3


async def test_dependencies_are_respected():
    eng = make_engine()
    rid = eng.submit(alert_for("auth-config-regression"))
    await run_with_approval(eng, rid)
    order = [(e["data"]["task_id"], e["type"]) for e in events(eng, rid, "task") if e["type"] in {"task.started", "task.succeeded"}]
    idx = {x: i for i, x in enumerate(order)}
    for dep, child in [("logs", "runbook"), ("metrics", "runbook"), ("runbook", "remediation"), ("remediation", "comms")]:
        assert idx[(dep, "task.succeeded")] < idx[(child, "task.started")]


async def test_sev1_plan_includes_parallel_ack_task():
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"))  # sev1
    res = await run_with_approval(eng, rid)
    plan = eng.get_runtime(rid).plan
    ack = plan.get("comms_ack")
    assert ack.depends_on == [] and res.status is RunStatus.COMPLETED


async def test_slow_task_hits_timeout_then_retries_successfully():
    inj = FaultInjector([FaultRule("llm:log_analyst", "latency", 1, delay_s=2.0)])
    eng = make_engine(injector=inj, task_timeout_s=0.2, llm_timeout_s=5.0)
    rid = eng.submit(alert_for("payments-memory-leak"))
    res = await run_with_approval(eng, rid)
    retries = events(eng, rid, "task.retry")
    assert res.status is RunStatus.COMPLETED and len(retries) == 1 and "TaskTimeout" in retries[0]["data"]["error"]


async def test_exhausted_retries_dead_letter_the_task_and_degrade_the_run():
    inj = FaultInjector([FaultRule("llm:metrics_analyst", "error", None)])
    eng = make_engine(injector=inj)
    rid = eng.submit(alert_for("payments-memory-leak"))
    res = await run_with_approval(eng, rid)
    assert res.status is RunStatus.DEGRADED and res.dead_letters == ["metrics"] and res.root_cause == "memory_leak"
    dl = eng.store.get_dead_letters(rid)
    assert len(dl) == 1 and dl[0]["task_id"] == "metrics" and dl[0]["payload"]["attempts"] == 3
    assert events(eng, rid, "task.dead_lettered")


async def test_required_task_failure_cascades_and_fails_the_run():
    inj = FaultInjector([FaultRule("llm:remediation", "error", None)])
    eng = make_engine(injector=inj)
    rid = eng.submit(alert_for("inventory-disk-full"))
    res = await run_with_approval(eng, rid)
    tasks = eng.store.get_tasks(rid)
    assert res.status is RunStatus.FAILED
    assert tasks["remediation"]["status"] == "failed" and tasks["comms"]["status"] == "skipped"
    assert tasks["comms"]["reason"] == "upstream_failed:remediation"
    assert eng._envs[rid].mutations == []


async def test_malformed_llm_output_is_repaired_without_failing_the_task():
    inj = FaultInjector([FaultRule("llm:deploy_history", "malformed", 1)])
    eng = make_engine(injector=inj)
    rid = eng.submit(alert_for("auth-config-regression"))
    res = await run_with_approval(eng, rid)
    assert res.status is RunStatus.COMPLETED
    assert events(eng, rid, "guardrail.output_invalid") and events(eng, rid, "guardrail.output_repaired")
    assert not events(eng, rid, "task.retry")  # repaired inside the LLM layer, no task retry


async def test_planner_failure_falls_back_to_default_plan():
    inj = FaultInjector([FaultRule("llm:planner", "malformed", None)])
    eng = make_engine(injector=inj)
    rid = eng.submit(alert_for("search-traffic-spike"))
    res = await run_with_approval(eng, rid)
    assert res.status is RunStatus.COMPLETED and events(eng, rid, "plan.fallback")


async def test_agent_bulkhead_limits_cross_run_concurrency():
    eng = make_engine(llm=None, agent_concurrency=2, latency_s=0.0)
    eng.deps.llm.latency_s = 0.02
    ids = [eng.submit(alert_for(SCENARIO_IDS[i % 6], id=f"a{i}")) for i in range(8)]
    stop = asyncio.create_task(_approve_all(eng))
    await asyncio.gather(*(eng.execute(r) for r in ids))
    stop.cancel()
    assert eng.deps.bulkheads.get("agent:log_analyst").peak <= 2
    assert eng.deps.bulkheads.get("agent:log_analyst").peak == 2  # it did saturate


async def _approve_all(eng):
    while True:
        for ap in eng.store.list_approvals(status="pending"):
            eng.hitl.approve(ap["approval_id"], "t", "ok")
        await asyncio.sleep(0.005)


async def test_per_agent_circuit_breaker_opens_and_fast_fails_later_runs():
    inj = FaultInjector([FaultRule("llm:metrics_analyst", "error", None)])
    eng = make_engine(injector=inj, breaker_failure_threshold=2, max_task_attempts=1, breaker_recovery_s=60)
    for i in range(4):
        rid = eng.submit(alert_for("payments-memory-leak", id=f"b{i}"))
        res = await run_with_approval(eng, rid)
        assert res.status is RunStatus.DEGRADED
    b = eng.deps.breakers.get("agent:metrics_analyst")
    assert b.opened_count == 1 and b.rejected_count >= 2
    llm_errors = [e for e in inj.stats.items() if e[0].startswith("error:llm:metrics")]
    assert llm_errors[0][1] == 2  # after opening, the broken dependency was no longer being hammered


async def test_cost_and_token_accounting_is_attached_to_llm_spans():
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"))
    res = await run_with_approval(eng, rid)
    llm = [s for s in eng.store.get_spans(rid) if s["kind"] == "llm"]
    assert llm and all(s["attrs"]["cost_usd"] > 0 and s["attrs"]["tokens_in"] > 0 for s in llm)
    assert res.usage["cost_usd"] == pytest.approx(sum(s["attrs"]["cost_usd"] for s in llm), rel=1e-3)
    assert res.usage["steps"] > len(llm)


async def test_trace_is_a_proper_run_task_leaf_tree():
    eng = make_engine()
    rid = eng.submit(alert_for("notifications-dependency-outage"))
    await run_with_approval(eng, rid)
    spans = {s["span_id"]: s for s in eng.store.get_spans(rid)}
    runs = [s for s in spans.values() if s["kind"] == "run"]
    assert len(runs) == 1
    for s in spans.values():
        if s["kind"] in {"llm", "tool"} and s["task_id"]:
            chain, cur = [], s
            while cur["parent_id"]:
                cur = spans[cur["parent_id"]]
                chain.append(cur["kind"])
            assert chain[-1] == "run" and "task" in chain
    assert any(s["name"].startswith("exec:") for s in spans.values())  # the approved mutation is traced too
