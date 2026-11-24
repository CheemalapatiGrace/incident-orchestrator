"""Least-privilege tool permission policy.

* Read-only tools are granted per agent.
* Mutating actions can only be *proposed* by the RemediationAgent; they are executed exclusively by
  the approval gate (principal ``approval_gate``) after a human approval record exists. No agent can
  call a mutating tool directly - not even the RemediationAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orchestrator.errors import PermissionDenied
from orchestrator.models import AgentName

GATE_PRINCIPAL = "approval_gate"

MUTATING_ACTIONS = frozenset(
    {"rollback_deploy", "restart_service", "scale_service", "revert_config", "expand_volume", "failover_dependency"}
)


@dataclass
class PermissionPolicy:
    read_tools: dict[str, frozenset[str]] = field(default_factory=dict)
    may_propose: frozenset[str] = frozenset()  # agents allowed to propose mutating actions
    enforce: bool = True

    def check_tool(self, agent: str, tool: str) -> None:
        if tool in MUTATING_ACTIONS:
            # always enforced, independent of `enforce` (hard invariant)
            if agent != GATE_PRINCIPAL:
                raise PermissionDenied(agent, tool, "mutating actions can only run via the approval gate")
            return
        if not self.enforce:
            return
        if tool not in self.read_tools.get(agent, frozenset()):
            raise PermissionDenied(agent, tool, "tool not in agent allow-list")

    def check_propose(self, agent: str) -> None:
        if agent not in self.may_propose:
            raise PermissionDenied(agent, "propose_action", "agent may not propose mutating actions")

    def matrix(self) -> dict[str, list[str]]:
        return {a: sorted(t) for a, t in self.read_tools.items()}


def default_policy(enforce: bool = True) -> PermissionPolicy:
    A = AgentName
    return PermissionPolicy(
        read_tools={
            A.LOG_ANALYST.value: frozenset({"search_logs", "get_service_info"}),
            A.METRICS_ANALYST.value: frozenset({"query_metrics", "get_service_info"}),
            A.DEPLOY_HISTORY.value: frozenset({"list_deploys", "get_service_info"}),
            A.RUNBOOK.value: frozenset({"search_runbooks"}),
            A.REMEDIATION.value: frozenset({"get_service_info"}),
            A.COMMS.value: frozenset({"get_service_info"}),
        },
        may_propose=frozenset({A.REMEDIATION.value}),
        enforce=enforce,
    )
