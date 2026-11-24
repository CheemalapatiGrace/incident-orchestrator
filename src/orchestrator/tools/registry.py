"""Typed tool definitions. Read tools are callable by agents (subject to policy); mutating tools are
only reachable through :meth:`ToolRegistry.execute_approved`, which demands an approval record."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from orchestrator.errors import ApprovalError, FatalToolError
from orchestrator.security.policy import MUTATING_ACTIONS


class SearchLogsArgs(BaseModel):
    service: str = Field(max_length=128)
    query: str = Field(default="", max_length=200)
    level: str | None = Field(default=None, max_length=10)
    since_minutes: int = Field(default=120, ge=1, le=1440)
    limit: int = Field(default=100, ge=1, le=500)


class QueryMetricsArgs(BaseModel):
    service: str = Field(max_length=128)
    metric: str = Field(default="all", max_length=64)
    window_minutes: int = Field(default=120, ge=5, le=1440)


class ListDeploysArgs(BaseModel):
    service: str = Field(max_length=128)
    since_minutes: int = Field(default=240, ge=1, le=20000)


class ServiceInfoArgs(BaseModel):
    service: str = Field(max_length=128)


class SearchRunbooksArgs(BaseModel):
    query: str = Field(max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class RollbackArgs(BaseModel):
    service: str
    to_version: str


class RestartArgs(BaseModel):
    service: str
    strategy: str = "rolling"


class ScaleArgs(BaseModel):
    service: str
    replicas: int


class RevertArgs(BaseModel):
    service: str
    key: str
    to_revision: str


class ExpandArgs(BaseModel):
    service: str
    size_gb: int


class FailoverArgs(BaseModel):
    service: str
    dependency: str
    target: str


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Awaitable[Any]]  # (env, args) for reads; (env, args, approval_id) for mutations
    mutating: bool = False
    timeout_s: float | None = None  # None -> config default


def canonical(args: dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))


def args_hash(args: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(args).encode()).hexdigest()[:16]


def _read(name: str, desc: str, model: type[BaseModel], method: str) -> Tool:
    return Tool(name, desc, model, lambda env, a, _m=method: getattr(env, _m)(**a.model_dump()))


def _mut(name: str, desc: str, model: type[BaseModel]) -> Tool:
    return Tool(name, desc, model, lambda env, a, approval_id=None, _m=name: getattr(env, _m)(approval_id=approval_id, **a.model_dump()), mutating=True)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise FatalToolError(f"unknown tool {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    @classmethod
    def default(cls) -> ToolRegistry:
        r = cls()
        r.register(_read("search_logs", "Search recent log lines for a service", SearchLogsArgs, "search_logs"))
        r.register(_read("query_metrics", "Fetch metric time series for a service", QueryMetricsArgs, "query_metrics"))
        r.register(_read("list_deploys", "List recent deploys and config changes", ListDeploysArgs, "list_deploys"))
        r.register(_read("get_service_info", "Service metadata: tier, replicas, version, dependencies", ServiceInfoArgs, "get_service_info"))
        r.register(_read("search_runbooks", "BM25 search over bundled runbooks", SearchRunbooksArgs, "search_runbooks"))
        r.register(_mut("rollback_deploy", "Roll a service back to a previous version", RollbackArgs))
        r.register(_mut("restart_service", "Rolling restart of a service", RestartArgs))
        r.register(_mut("scale_service", "Set replica count", ScaleArgs))
        r.register(_mut("revert_config", "Revert a config key to a previous revision", RevertArgs))
        r.register(_mut("expand_volume", "Grow a data volume", ExpandArgs))
        r.register(_mut("failover_dependency", "Switch a dependency to its configured alternate", FailoverArgs))
        assert {t for t, v in r._tools.items() if v.mutating} == set(MUTATING_ACTIONS)
        return r


def verify_approval(approval: dict | None, action: str, params: dict[str, Any]) -> None:
    """Hard check performed immediately before any mutation: a matching, human-granted approval exists."""
    if not approval:
        raise ApprovalError("no approval record: mutating action refused")
    if approval["status"] not in {"approved", "edited"}:
        raise ApprovalError(f"approval status is {approval['status']!r}: mutating action refused")
    if not approval.get("decided_by"):
        raise ApprovalError("approval has no deciding principal")
    if approval["proposal"].get("action") != action:
        raise ApprovalError("approval was granted for a different action")
    final = approval.get("final_params") if approval.get("final_params") is not None else approval["proposal"].get("params")
    if canonical(final) != canonical(params):
        raise ApprovalError("parameters differ from the approved parameters")
