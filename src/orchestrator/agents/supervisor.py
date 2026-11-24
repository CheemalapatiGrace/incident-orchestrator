"""Supervisor/planner: decomposes an alert into a validated task DAG."""

from __future__ import annotations

import json

from orchestrator.core.budget import BudgetTracker
from orchestrator.core.dag import validate_plan
from orchestrator.core.runtime import Deps
from orchestrator.errors import BudgetExceeded, OrchestratorError, PlanValidationError
from orchestrator.llm.structured import StructuredLLM
from orchestrator.models import AgentName, Alert, Plan, TaskSpec

REQUIRED = frozenset({AgentName.REMEDIATION})


def default_plan(alert: Alert) -> Plan:
    """Deterministic safety-net plan used when the planner model fails or returns an invalid DAG."""
    s = alert.service
    tasks = [
        TaskSpec(id="logs", agent=AgentName.LOG_ANALYST, description="Analyse recent logs", params={"service": s}),
        TaskSpec(id="metrics", agent=AgentName.METRICS_ANALYST, description="Analyse metrics", params={"service": s}),
        TaskSpec(id="deploys", agent=AgentName.DEPLOY_HISTORY, description="Recent changes", params={"service": s}),
        TaskSpec(id="runbook", agent=AgentName.RUNBOOK, description="Find runbook", depends_on=["logs", "metrics"]),
        TaskSpec(id="remediation", agent=AgentName.REMEDIATION, description="Root cause and proposals", depends_on=["logs", "metrics", "deploys", "runbook"], optional=False),
        TaskSpec(id="comms", agent=AgentName.COMMS, description="Status update", depends_on=["remediation"], params={"kind": "update"}),
    ]
    return Plan(tasks=tasks, rationale="fallback plan")


class Supervisor:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps

    async def plan(self, run_id: str, alert: Alert, budget: BudgetTracker) -> tuple[Plan, bool]:
        """Returns (plan, used_fallback)."""
        d = self.deps
        llm = StructuredLLM(llm=d.llm, cfg=d.cfg, tracer=d.tracer, bus=d.bus, budget=budget, injector=d.injector, run_id=run_id, agent="supervisor")
        ctx = {"incident": {"id": alert.id, "service": alert.service, "severity": alert.severity.value, "title": alert.title}}
        prompt = f"<context>\n{json.dumps(ctx)}\n</context>\nTask: decompose the incident into a DAG of specialist tasks. Agents: {[a.value for a in AgentName]}."
        try:
            plan = await llm.call(role="planner", system="You are the incident supervisor.", prompt=prompt, model=Plan)
            validate_plan(plan, required_agents=REQUIRED)
            return plan, False
        except BudgetExceeded:
            raise
        except (PlanValidationError, OrchestratorError) as e:
            d.bus.publish(run_id, "plan.fallback", {"reason": str(e)[:300]})
            plan = default_plan(alert)
            validate_plan(plan, required_agents=REQUIRED)
            return plan, True
