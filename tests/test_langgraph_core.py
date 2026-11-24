"""Properties that exist because the orchestration core is a LangGraph ``StateGraph``:
topology, Send fan-out compiled from the plan, superstep checkpoints and crash resume, ``interrupt()`` /
``Command(resume=...)``, ``update_state`` edits, and restart-survives-approval."""

import asyncio
import pathlib

import pytest

from conftest import (
    alert_for,
    approve_when_pending,
    events,
    fast_cfg,
    make_engine,
    run_with_approval,
    tstatus,
    wait_pending,
    wait_until,
)
from orchestrator.agents.specialists import AGENTS
from orchestrator.core import runner as runner_mod
from orchestrator.core.dag import topo_levels
from orchestrator.core.events import EventBus, set_node_writer
from orchestrator.core.graph import AGENT_NODES
from orchestrator.engine import Orchestrator
from orchestrator.errors import SimulatedCrash
from orchestrator.llm.mock import MockLLM
from orchestrator.models import AgentName, Budget, RunStatus, TaskStatus
from orchestrator.sim.environment import SimulatedEnvironment


async def graph_and_cfg(eng, rid):
    g = await eng.runner.graph()
    return g, eng.runner.config(eng.get_runtime(rid), 8)


def engine_on(db, wid="w", **kw):
    return Orchestrator(fast_cfg(db_path=db, **kw), llm=MockLLM(), worker_id=wid, env_factory=lambda a: SimulatedEnvironment.for_scenario(a.labels["scenario"]))


# ----------------------------------------------------------------------------- topology
async def test_compiled_graph_topology_has_supervisor_scheduler_six_specialists_gate_and_hold():
    eng = make_engine()
    g = (await eng.runner.graph()).get_graph()
    assert {"supervisor", "dispatch", "approval_gate", "hold", "finalize", *AGENT_NODES} <= set(g.nodes)
    edges = {(e.source, e.target) for e in g.edges}
    assert ("__start__", "supervisor") in edges and ("supervisor", "dispatch") in edges and ("finalize", "__end__") in edges
    for n in (*AGENT_NODES, "approval_gate", "hold"):
        assert (n, "dispatch") in edges  # fan-in: every branch joins back at the scheduler
        assert ("dispatch", n) in edges  # fan-out: the scheduler can Send to it
    assert {s for s, t in edges if t == "dispatch"} >= {"supervisor"}


def test_docs_graph_md_matches_the_compiled_graph():
    async def mermaid():
        eng = make_engine()
        return (await eng.runner.graph()).get_graph().draw_mermaid()

    doc = pathlib.Path(__file__).parent.parent.joinpath("docs", "graph.md").read_text()
    assert asyncio.run(mermaid()).strip() in doc, "docs/graph.md is stale: run `make graph`"


async def test_send_fan_out_waves_are_compiled_from_the_plan():
    """Each superstep fans out exactly the plan's next topological layer with ``Send``."""
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"))  # sev1: 4 independent roots (incl. a parallel comms ack)
    await run_with_approval(eng, rid)
    g, cfg = await graph_and_cfg(eng, rid)
    history = [s async for s in g.aget_state_history(cfg)]
    waves = [sorted(set(s.next) & {*AGENT_NODES, "approval_gate"}) for s in reversed(history) if set(s.next) & {*AGENT_NODES, "approval_gate"}]
    plan = eng.get_runtime(rid).plan
    expected = [sorted({plan.get(t).agent.value for t in lvl}) for lvl in topo_levels(plan)]
    # a plan layer == a superstep (the human gate is one extra step inside the remediation layer)
    assert [w for w in waves if w != ["approval_gate"]] == expected
    assert ["approval_gate"] in waves and waves.index(["approval_gate"]) == waves.index(["remediation"]) + 1
    assert waves[0] == ["comms", "deploy_history", "log_analyst", "metrics_analyst"]  # one superstep, four concurrent branches


async def test_parallel_branches_really_overlap_in_one_superstep():
    eng = make_engine(latency_s=0.05)
    rid = eng.submit(alert_for("search-traffic-spike"))
    await run_with_approval(eng, rid)
    spans = {s["name"]: s for s in eng.store.get_spans(rid) if s["kind"] == "task"}
    trio = [spans["logs"], spans["metrics"], spans["deploys"]]
    first_end = min(s["end_ts"] for s in trio)
    assert all(s["start_ts"] < first_end for s in trio)  # all three started before any finished
    wall = max(s["end_ts"] for s in trio) - min(s["start_ts"] for s in trio)
    assert wall < 0.75 * sum(s["end_ts"] - s["start_ts"] for s in trio)  # concurrent, not sequential


async def test_every_superstep_is_checkpointed_and_state_reducers_merge_parallel_branches():
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"))
    await run_with_approval(eng, rid)
    g, cfg = await graph_and_cfg(eng, rid)
    snap = await g.aget_state(cfg)
    assert snap.next == () and snap.values["status"] == "completed"
    assert {t: r["status"] for t, r in snap.values["tasks"].items()} == {t: "succeeded" for t in snap.values["tasks"]}
    history = [s async for s in g.aget_state_history(cfg)]
    assert len(history) >= 12  # one checkpoint per superstep (supervisor, 6+ dispatch rounds, specialists, gate, finalize)
    # four specialists finished in ONE superstep: their `tasks` patches were folded by the reducer
    wave1 = next(s for s in reversed(history) if set(s.next) >= {"log_analyst", "deploy_history"})
    after = next(s for s in reversed(history) if s.metadata["step"] == wave1.metadata["step"] + 1)
    assert {"logs", "metrics", "deploys", "comms_ack"} <= {t for t, r in after.values["tasks"].items() if r["status"] == "succeeded"}


# ------------------------------------------------------------------- checkpointer resume
async def test_crash_inside_a_superstep_resumes_without_rerunning_finished_siblings(tmp_path, monkeypatch):
    db = str(tmp_path / "mid.db")
    e1 = engine_on(db, "w1")
    rid = e1.submit(alert_for("checkout-bad-deploy"))
    metrics = AGENTS[AgentName.METRICS_ANALYST]
    orig = metrics.run

    async def dying(ctx):
        await asyncio.sleep(0.15)  # logs / deploys / comms_ack have finished and been checkpointed by now
        raise SimulatedCrash("kill -9 in the middle of a superstep")

    monkeypatch.setattr(metrics, "run", dying)
    with pytest.raises(SimulatedCrash):
        await e1.execute(rid)
    monkeypatch.setattr(metrics, "run", orig)
    e1.store.close()

    e2 = engine_on(db, "w2")
    res = await run_with_approval(e2, rid)
    assert res.status is RunStatus.COMPLETED and res.root_cause == "bad_deploy"
    called = {c["tool"] for c in e2._envs[rid].calls}
    assert "query_metrics" in called  # the interrupted branch re-ran...
    assert not {"search_logs", "list_deploys"} & called  # ...its finished siblings did not
    assert len(e2._envs[rid].mutations) == 1


async def test_crash_at_a_superstep_boundary_resumes_at_dispatch(tmp_path):
    from orchestrator.faults import FaultInjector

    db = str(tmp_path / "b.db")
    e1 = Orchestrator(fast_cfg(db_path=db), llm=MockLLM(), injector=FaultInjector(crash_after_tasks=1), worker_id="w1", env_factory=lambda a: SimulatedEnvironment.for_scenario(a.labels["scenario"]))
    rid = e1.submit(alert_for("auth-config-regression"))
    with pytest.raises(SimulatedCrash):
        await e1.execute(rid)
    g, cfg = await graph_and_cfg(e1, rid)
    snap = await g.aget_state(cfg)
    assert snap.next == ("dispatch",) and snap.values["tasks"]["logs"]["status"] == "succeeded"
    e1.store.close()
    e2 = engine_on(db, "w2")
    res = await run_with_approval(e2, rid)
    assert res.status is RunStatus.COMPLETED and res.root_cause == "config_error"


# ------------------------------------------------------------------ interrupt / Command(resume)
async def test_approval_gate_parks_the_graph_with_interrupt_and_resumes_with_command():
    eng = make_engine()
    rid = eng.submit(alert_for("search-traffic-spike"))
    task = asyncio.create_task(eng.execute(rid))
    (ap,) = await wait_pending(eng, rid)
    g, cfg = await graph_and_cfg(eng, rid)
    await wait_until(lambda: eng.runtimes[rid].status is RunStatus.AWAITING_APPROVAL)
    snap = await g.aget_state(cfg)
    assert snap.next == ("approval_gate",)
    (intr,) = snap.interrupts
    assert intr.value["kind"] == "approval" and intr.value["approval_ids"] == [ap["approval_id"]]
    assert eng._envs[rid].mutations == []  # nothing mutates while parked
    eng.hitl.approve(ap["approval_id"], "alice", "traffic is real")
    res = await asyncio.wait_for(task, 5)
    assert res.status is RunStatus.COMPLETED and len(eng._envs[rid].mutations) == 1
    assert (await g.aget_state(cfg)).interrupts == ()


async def test_node_events_travel_through_the_custom_stream_in_order():
    eng = make_engine()
    rid = eng.submit(alert_for("auth-config-regression"))
    await run_with_approval(eng, rid)
    evs = [e for e in events(eng, rid) if not e["type"].startswith("span.")]
    seqs = [e["seq"] for e in events(eng, rid)]
    assert seqs == sorted(seqs) and evs[0]["type"] == "run.submitted" and evs[-1]["type"] == "run.completed"
    started = {e["data"]["task_id"]: e["seq"] for e in evs if e["type"] == "task.started"}
    done = {e["data"]["task_id"]: e["seq"] for e in evs if e["type"] == "task.succeeded"}
    assert all(started[t] < done[t] for t in done)


def test_event_bus_routes_through_the_node_writer_only_inside_a_node():
    eng = make_engine()
    got = []
    set_node_writer(got.append)
    try:
        assert EventBus.publish(eng.bus, "r1", "x.y", {"a": 1}) is None  # handed to the graph's custom stream
    finally:
        set_node_writer(None)
    assert got[0]["type"] == "x.y" and eng.store.get_events("r1") == []
    assert eng.bus.publish("r1", "x.y", {"a": 1}).seq > 0  # outside a node: persisted directly


# ------------------------------------------------------------------------ update_state edits
async def test_update_state_edit_while_parked_at_a_breakpoint_changes_what_the_node_runs():
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"), breakpoints=["logs"])
    task = asyncio.create_task(run_with_approval(eng, rid))
    await wait_until(lambda: tstatus(eng, rid, "logs") is TaskStatus.HELD and eng.runtimes[rid].status is RunStatus.RUNNING and eng.runtimes[rid].budget._wait_start is not None)
    g, cfg = await graph_and_cfg(eng, rid)
    snap = await g.aget_state(cfg)
    assert snap.next == ("hold",)  # interrupt_before=['hold']: parked, nothing in flight
    # an operator (or any script) edits the checkpointed state directly...
    await g.aupdate_state(cfg, {"tasks": {"logs": {"input_overrides": {"query": "OutOfMemoryError"}, "released": True}}}, as_node="hold")
    res = await asyncio.wait_for(task, 5)  # ...and the driver notices the graph moved on
    assert res.status is RunStatus.COMPLETED
    assert any(c["tool"] == "search_logs" and c["args"]["query"] == "OutOfMemoryError" for c in eng._envs[rid].calls)


async def test_human_edits_are_state_patches_folded_into_the_checkpoint():
    eng = make_engine()
    rid = eng.submit(alert_for("payments-memory-leak"), breakpoints=["logs"])
    task = asyncio.create_task(run_with_approval(eng, rid))
    await wait_until(lambda: tstatus(eng, rid, "logs") is TaskStatus.HELD)
    eng.hitl.edit_task_input(rid, "logs", "alice", {"window_minutes": 45})
    eng.hitl.inject_guidance(rid, "alice", "customer impact is high")
    assert len(eng.store.pending_commands(rid)) in (0, 2)  # either folded already or still queued, never lost
    eng.hitl.release_task(rid, "logs", "alice")
    await asyncio.wait_for(task, 5)
    g, cfg = await graph_and_cfg(eng, rid)
    v = (await g.aget_state(cfg)).values
    assert v["tasks"]["logs"]["input_overrides"] == {"window_minutes": 45} and v["tasks"]["logs"]["released"]
    assert [x["text"] for x in v["guidance"]] == ["customer impact is high"]
    assert eng.store.pending_commands(rid) == []


async def test_queued_human_edit_survives_a_restart_while_the_gate_is_parked(tmp_path):
    db = str(tmp_path / "q.db")
    e1 = engine_on(db, "w1")
    rid = e1.submit(alert_for("checkout-bad-deploy"))
    t = asyncio.create_task(e1.execute(rid))
    await wait_pending(e1, rid)
    e1.hitl.inject_guidance(rid, "alice", "Avoid rollback: the release contains a schema migration")
    assert e1.store.pending_commands(rid)  # parked at the gate: queued, not yet in the checkpoint
    t.cancel()
    await asyncio.gather(t, return_exceptions=True)
    e1.store.close()
    e2 = engine_on(db, "w2")
    t2 = asyncio.create_task(e2.execute(rid))
    (ap,) = await wait_pending(e2, rid)
    e2.hitl.approve(ap["approval_id"], "alice", "ok, proceed")
    res = await asyncio.wait_for(t2, 5)
    assert res.status is RunStatus.COMPLETED and "Operator note: Avoid rollback" in res.comms["body"]


# --------------------------------------------------------------- restart survives approval
async def test_pending_approval_survives_two_restarts_and_executes_exactly_once(tmp_path):
    db = str(tmp_path / "r.db")
    e = engine_on(db, "w0")
    rid = e.submit(alert_for("inventory-disk-full"))
    t = asyncio.create_task(e.execute(rid))
    (first,) = await wait_pending(e, rid)
    ids = [first["approval_id"]]
    for i in (1, 2):  # kill -9 twice while the human has not decided
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        e.store.close()
        e = engine_on(db, f"w{i}")
        t = asyncio.create_task(e.execute(rid))
        (again,) = await wait_pending(e, rid)
        ids.append(again["approval_id"])
        g, cfg = await graph_and_cfg(e, rid)
        await wait_until(lambda eng=e: eng.runtimes[rid].status is RunStatus.AWAITING_APPROVAL)
        snap = await g.aget_state(cfg)
        assert snap.interrupts and snap.interrupts[0].value["approval_ids"] == [first["approval_id"]]  # same interrupt, from the checkpoint
        assert e._envs[rid].mutations == []
    assert len(set(ids)) == 1 and len(e.store.list_approvals(rid)) == 1
    e.hitl.approve(ids[0], "alice", "volume at 100%")
    res = await asyncio.wait_for(t, 5)
    assert res.status is RunStatus.COMPLETED and len(e._envs[rid].mutations) == 1
    again = await e.execute(rid)  # a finished run is never re-executed
    assert again.status is RunStatus.COMPLETED and len(e._envs[rid].mutations) == 1
    assert len(e._envs[rid].mutations) == 1 and e.store.get_run(rid)["status"] == "completed"


async def test_approving_while_no_process_runs_executes_once_on_the_next_start(tmp_path):
    db = str(tmp_path / "off.db")
    e1 = engine_on(db, "w1")
    rid = e1.submit(alert_for("inventory-disk-full"))
    t = asyncio.create_task(e1.execute(rid))
    (ap,) = await wait_pending(e1, rid)
    t.cancel()
    await asyncio.gather(t, return_exceptions=True)
    e1.store.close()
    cli = engine_on(db, "cli")  # `python -m orchestrator approvals approve ...` while the server is down
    cli.hitl.approve(ap["approval_id"], "cli:alice", "checked")
    e2 = engine_on(db, "w2")
    res = await asyncio.wait_for(e2.execute(rid), 5)
    assert res.status is RunStatus.COMPLETED and len(e2._envs[rid].mutations) == 1 and res.actions[0]["decided_by"] == "cli:alice"


# ----------------------------------------------------------------------- backstops on top
async def test_recursion_limit_is_an_extra_backstop(monkeypatch):
    monkeypatch.setattr(runner_mod, "RECURSION_LIMIT", 4)
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"))
    res = await run_with_approval(eng, rid)
    assert res.status is RunStatus.BUDGET_EXCEEDED and eng._envs[rid].mutations == []
    assert events(eng, rid, "run.aborted")[0]["data"]["dimension"] == "supersteps"


async def test_budget_abort_inside_a_parallel_superstep_fails_the_run_cleanly():
    eng = make_engine()
    rid = eng.submit(alert_for("payments-memory-leak"), budget=Budget(max_steps=3))
    res = await run_with_approval(eng, rid)
    assert res.status is RunStatus.BUDGET_EXCEEDED and eng._envs[rid].mutations == []
    assert not eng.store.list_approvals(rid, "pending") and approve_when_pending is not None


async def test_langgraph_tracing_is_off_by_default_so_no_network_calls(monkeypatch):
    from langsmith.utils import tracing_is_enabled

    monkeypatch.setenv("LANGSMITH_TRACING", "true")  # even a hostile shell environment is overridden
    eng = make_engine()
    await eng.runner.graph()
    assert tracing_is_enabled() is False


async def test_false_alarm_short_circuit_skips_the_runbook_lookup_but_still_diagnoses_and_reports():
    eng = make_engine()
    rid = eng.submit(alert_for("reporting-false-alarm"))  # sev3, nothing anomalous anywhere
    res = await run_with_approval(eng, rid)
    t = eng.store.get_tasks(rid)
    assert t["runbook"]["status"] == "skipped" and t["runbook"]["reason"] == "short_circuit:false_alarm"
    assert not any(c["tool"] == "search_runbooks" for c in eng._envs[rid].calls)
    assert res.status is RunStatus.COMPLETED and res.root_cause == "unknown" and res.actions == [] and res.comms  # a deliberate skip is no degradation


async def test_short_circuit_never_applies_to_a_real_incident_or_a_tainted_run():
    eng = make_engine()
    rid = eng.submit(alert_for("payments-memory-leak"))  # sev2 with a real signal
    await run_with_approval(eng, rid)
    assert eng.store.get_tasks(rid)["runbook"]["status"] == "succeeded"


async def test_reaper_adopts_a_run_whose_worker_died_and_the_checkpoint_resumes_it(tmp_path):
    db = str(tmp_path / "reap.db")
    e1 = engine_on(db, "w1", lease_s=0.3)
    rid = e1.submit(alert_for("inventory-disk-full"))
    t = asyncio.create_task(e1.execute(rid))
    (ap,) = await wait_pending(e1, rid)
    t.cancel()  # the worker "dies" holding a lease...
    await asyncio.gather(t, return_exceptions=True)
    e1.store.claim_run(rid, "dead-worker", 0.05)  # (a real kill -9 never releases it)
    e1.store.close()
    e2 = engine_on(db, "w2", lease_s=0.3)
    e2.hitl.approve(ap["approval_id"], "alice", "volume at 100%")  # decided while nobody was running the graph
    e2.start_workers(1)
    await wait_until(lambda: e2.store.get_run(rid)["status"] == "completed", 8)
    await e2.stop_workers()
    assert len(e2._envs[rid].mutations) == 1 and len(e2.store.list_approvals(rid)) == 1
