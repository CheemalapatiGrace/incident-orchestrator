import asyncio

import pytest

from conftest import alert_for, make_engine, run_with_approval, wait_pending
from orchestrator.core.store import Store
from orchestrator.errors import SimulatedCrash
from orchestrator.faults import FaultInjector
from orchestrator.models import RunStatus

TOOL_OF = {"logs": "search_logs", "metrics": "query_metrics", "deploys": "list_deploys", "runbook": "search_runbooks"}


@pytest.mark.parametrize("crash_after", [1, 3, 5])
async def test_crashed_run_resumes_from_last_completed_node(tmp_path, crash_after):
    db = str(tmp_path / "run.db")
    from conftest import fast_cfg
    from orchestrator.engine import Orchestrator
    from orchestrator.llm.mock import MockLLM
    from orchestrator.sim.environment import SimulatedEnvironment

    def build(inj=None, wid="w"):
        return Orchestrator(fast_cfg(db_path=db), llm=MockLLM(), injector=inj, worker_id=wid,
                            env_factory=lambda a: SimulatedEnvironment.for_scenario(a.labels["scenario"]))

    e1 = build(FaultInjector(crash_after_tasks=crash_after), "w1")
    rid = e1.submit(alert_for("checkout-bad-deploy"))
    with pytest.raises(SimulatedCrash):
        await e1.execute(rid)
    done = {t for t, s in e1.store.get_tasks(rid).items() if s["status"] == "succeeded"}
    assert len(done) >= crash_after
    assert e1.store.get_run(rid)["status"] == "running"  # died mid-flight
    steps_before = e1.store.get_run(rid)["usage"]["steps"]
    e1.store.close()

    e2 = build(wid="w2")  # a brand-new process on the same database file
    res = await run_with_approval(e2, rid)
    assert res.status is RunStatus.COMPLETED and res.root_cause == "bad_deploy"
    called = {c["tool"] for c in e2._envs[rid].calls}
    for t in done & set(TOOL_OF):
        assert TOOL_OF[t] not in called, f"completed node {t} was re-executed"
    assert len(e2._envs[rid].mutations) == 1
    assert e2.get_runtime(rid).budget.steps > steps_before  # usage is carried across the crash


async def test_crash_while_awaiting_human_keeps_the_pending_approval(tmp_path):
    db = str(tmp_path / "gate.db")
    e1 = make_engine(db_path=db, worker_id="w1")
    rid = e1.submit(alert_for("search-traffic-spike"))
    t = asyncio.create_task(e1.execute(rid))
    before = await wait_pending(e1, rid)
    t.cancel()  # kill -9 while a human decision is outstanding
    await asyncio.gather(t, return_exceptions=True)
    assert e1.store.get_tasks(rid)["remediation"]["status"] == "awaiting_approval"
    e1.store.close()

    e2 = make_engine(db_path=db, worker_id="w2")
    t2 = asyncio.create_task(e2.execute(rid))
    after = await wait_pending(e2, rid)
    assert [a["approval_id"] for a in after] == [a["approval_id"] for a in before]  # same request, not a duplicate
    e2.hitl.approve(after[0]["approval_id"], "alice", "approved after restart")
    res = await t2
    assert res.status is RunStatus.COMPLETED
    assert len(e2._envs[rid].mutations) == 1 and len(e2.store.list_approvals(rid)) == 1


async def test_run_lease_is_exclusive_and_expires(tmp_path):
    s = Store(str(tmp_path / "l.db"))
    s.create_run("r1", {"x": 1}, {}, [])
    assert s.claim_run("r1", "w1", 60)
    assert not s.claim_run("r1", "w2", 60)  # held
    assert s.claim_run("r1", "w1", 60)  # owner can renew
    s.update_run("r1", status="running")
    s._x("UPDATE runs SET lease_expires=0 WHERE run_id='r1'")  # w1 died and its lease lapsed
    assert s.claim_run("r1", "w2", 60)


async def test_execute_returns_none_when_another_worker_owns_the_run():
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"))
    assert eng.store.claim_run(rid, "someone-else", 60)
    assert await eng.execute(rid) is None
    assert eng.store.get_run(rid)["status"] == "pending"


async def test_finished_run_is_not_re_executed():
    eng = make_engine()
    rid = eng.submit(alert_for("inventory-disk-full"))
    r1 = await run_with_approval(eng, rid)
    calls = len(eng._envs[rid].calls)
    r2 = await eng.execute(rid)
    assert r2.status == r1.status and len(eng._envs[rid].calls) == calls
