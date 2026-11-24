"""The six specialist agents. Each declares which tools it gathers from; the permission policy
(``security/policy.py``) independently constrains what it may call."""

from __future__ import annotations

from typing import Any

from orchestrator.agents.base import AgentContext, BaseAgent
from orchestrator.agents.schemas import (
    CommsOutput,
    DeployOutput,
    LogOutput,
    MetricsOutput,
    RemediationOutput,
    RunbookOutput,
)
from orchestrator.models import AgentName


class LogAnalyst(BaseAgent):
    name = AgentName.LOG_ANALYST
    OutputModel = LogOutput
    instruction = "Identify error signatures, their onset time and likely root-cause categories from the log lines."

    async def gather(self, ctx: AgentContext) -> dict[str, Any]:
        win = int(ctx.params.get("window_minutes", 120))
        base = {"service": ctx.service, "since_minutes": win}
        ev = {
            "tool:search_logs:errors": await ctx.tools.call("search_logs", {**base, "level": "ERROR", "limit": 100}),
            "tool:search_logs:warnings": await ctx.tools.call("search_logs", {**base, "level": "WARN", "limit": 40}),
        }
        if ctx.params.get("query"):  # a human can steer the analysis by editing this param before the node runs
            ev["tool:search_logs:custom"] = await ctx.tools.call("search_logs", {**base, "query": str(ctx.params["query"]), "limit": 60})
        return ev


class MetricsAnalyst(BaseAgent):
    name = AgentName.METRICS_ANALYST
    OutputModel = MetricsOutput
    instruction = "Detect anomalous metrics versus baseline and map them to root-cause categories."

    async def gather(self, ctx: AgentContext) -> dict[str, Any]:
        return {"tool:query_metrics:all": await ctx.tools.call("query_metrics", {"service": ctx.service, "metric": "all", "window_minutes": int(ctx.params.get("window_minutes", 120))})}


class DeployHistoryAgent(BaseAgent):
    name = AgentName.DEPLOY_HISTORY
    OutputModel = DeployOutput
    instruction = "Correlate recent deploys/config changes with the alert time and flag the most likely culprit."

    async def gather(self, ctx: AgentContext) -> dict[str, Any]:
        return {"tool:list_deploys:recent": await ctx.tools.call("list_deploys", {"service": ctx.service, "since_minutes": int(ctx.params.get("since_minutes", 240))})}


class RunbookAgent(BaseAgent):
    name = AgentName.RUNBOOK
    OutputModel = RunbookOutput
    instruction = "Select the most relevant runbook and summarise its steps."

    async def gather(self, ctx: AgentContext) -> dict[str, Any]:
        terms: list[str] = []
        for _tid, _agent, out in ctx.upstream:
            if out.get("_unavailable"):
                continue
            for c in out.get("suspected_causes", []):
                terms.append(c["category"])
            for s in out.get("error_signatures", []):
                terms.append(s["pattern"])
            for a in out.get("anomalies", []):
                terms += [a["kind"], a["metric"]]
        if ctx.params.get("query"):
            terms.append(str(ctx.params["query"]))
        query = " ".join(terms) or f"{ctx.service} incident"
        return {"tool:search_runbooks:query": await ctx.tools.call("search_runbooks", {"query": query[:900], "top_k": 3})}


class RemediationAgent(BaseAgent):
    name = AgentName.REMEDIATION
    OutputModel = RemediationOutput
    instruction = "Fuse the specialists' evidence into one root cause and PROPOSE (never execute) remediation actions."

    async def gather(self, ctx: AgentContext) -> dict[str, Any]:
        return {"tool:get_service_info:service": await ctx.tools.call("get_service_info", {"service": ctx.service})}

    async def finalize(self, ctx: AgentContext, out: dict[str, Any]) -> dict[str, Any]:
        if out["proposed_actions"]:
            ctx.deps.policy.check_propose(self.name.value)  # only this agent may propose mutations
        return out


class CommsAgent(BaseAgent):
    name = AgentName.COMMS
    OutputModel = CommsOutput
    instruction = "Draft a concise, factual stakeholder status update. Do not quote raw log lines."

    async def gather(self, ctx: AgentContext) -> dict[str, Any]:
        return {"tool:get_service_info:service": await ctx.tools.call("get_service_info", {"service": ctx.service})}


AGENTS: dict[AgentName, BaseAgent] = {
    a.name: a for a in (LogAnalyst(), MetricsAnalyst(), DeployHistoryAgent(), RunbookAgent(), RemediationAgent(), CommsAgent())
}
