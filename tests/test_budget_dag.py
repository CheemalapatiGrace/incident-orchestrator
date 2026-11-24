import pytest

from conftest import alert_for, make_engine, run_with_approval
from orchestrator.agents.supervisor import default_plan
from orchestrator.core.budget import BudgetTracker, LoopDetector
from orchestrator.core.dag import descendants, topo_levels, validate_plan
from orchestrator.errors import BudgetExceeded, LoopDetected, PlanValidationError
from orchestrator.models import AgentName, Budget, Plan, TaskSpec


def test_step_budget():
    b = BudgetTracker(Budget(max_steps=3))
    b.charge_step()
    b.charge_step()
    b.charge_step()
    with pytest.raises(BudgetExceeded) as e:
        b.charge_step()
    assert e.value.dimension == "steps"


def test_token_and_cost_budget():
    b = BudgetTracker(Budget(max_tokens=100, max_cost_usd=0.01))
    b.charge_llm(40, 40, 0.001)
    with pytest.raises(BudgetExceeded) as e:
        b.charge_llm(20, 20, 0.001)
    assert e.value.dimension == "tokens"
    b2 = BudgetTracker(Budget(max_tokens=10**9, max_cost_usd=0.01))
    with pytest.raises(BudgetExceeded) as e2:
        b2.charge_llm(1, 1, 0.02)
    assert e2.value.dimension == "cost_usd"


def test_wall_clock_budget_excludes_human_wait():
    t = [0.0]
    b = BudgetTracker(Budget(max_wall_clock_s=10), clock=lambda: t[0])
    t[0] = 6
    b.set_human_waiting(True)  # a human takes their time...
    t[0] = 106
    b.set_human_waiting(False)
    b.check_wall_clock()  # ...but only 6s of *active* time was consumed
    assert 5.9 < b.active_seconds() < 6.1
    t[0] = 111
    with pytest.raises(BudgetExceeded):
        b.check_wall_clock()


def test_budget_usage_survives_restore():
    b = BudgetTracker(Budget(max_steps=10))
    for _ in range(4):
        b.charge_step()
    b2 = BudgetTracker(Budget(max_steps=10), usage=b.snapshot())
    assert b2.steps == 4


def test_loop_detector_identical_calls():
    d = LoopDetector(3)
    d.record("a")
    d.record("a")
    with pytest.raises(LoopDetected):
        d.record("a")


def test_loop_detector_cycles():
    d = LoopDetector(3)
    for s in ["a", "b", "a", "b", "a"]:
        d.record(s)
    with pytest.raises(LoopDetected):
        d.record("b")


def test_loop_detector_tolerates_varied_calls():
    d = LoopDetector(3)
    for s in ["a", "b", "c", "a", "b", "d", "a"]:
        d.record(s)


async def test_tiny_step_budget_aborts_run_and_executes_nothing():
    from orchestrator.models import Budget as B

    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"), budget=B(max_steps=5))
    res = await run_with_approval(eng, rid)
    assert res.status.value == "budget_exceeded"
    assert eng._envs[rid].mutations == []
    assert any(e["type"] == "run.aborted" for e in eng.store.get_events(rid))
    assert not eng.store.list_approvals(rid, "pending")


def test_default_plan_is_valid_dag_with_parallel_triage():
    plan = default_plan(alert_for("checkout-bad-deploy"))
    validate_plan(plan, required_agents=frozenset({AgentName.REMEDIATION}))
    levels = topo_levels(plan)
    assert set(levels[0]) == {"logs", "metrics", "deploys"}  # fan-out
    assert levels[-1] == ["comms"]
    assert descendants(plan, "logs") == {"runbook", "remediation", "comms"}


@pytest.mark.parametrize(
    "tasks,msg",
    [
        ([TaskSpec(id="a", agent=AgentName.LOG_ANALYST, depends_on=["b"]), TaskSpec(id="b", agent=AgentName.RUNBOOK, depends_on=["a"])], "cycle"),
        ([TaskSpec(id="a", agent=AgentName.LOG_ANALYST, depends_on=["zzz"])], "unknown task"),
        ([TaskSpec(id="a", agent=AgentName.LOG_ANALYST), TaskSpec(id="a", agent=AgentName.RUNBOOK)], "duplicate"),
        ([TaskSpec(id="a", agent=AgentName.LOG_ANALYST, depends_on=["a"])], "itself"),
        ([], "no tasks"),
    ],
)
def test_invalid_plans_rejected(tasks, msg):
    with pytest.raises(PlanValidationError, match=msg):
        validate_plan(Plan(tasks=tasks))


def test_required_agent_enforced():
    with pytest.raises(PlanValidationError, match="remediation"):
        validate_plan(Plan(tasks=[TaskSpec(id="a", agent=AgentName.LOG_ANALYST)]), required_agents=frozenset({AgentName.REMEDIATION}))
