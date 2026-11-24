"""Pydantic schemas for every structured LLM output (validated + repaired by the structured caller)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.models import ActionProposal

CATEGORIES = ("bad_deploy", "memory_leak", "traffic_spike", "config_error", "disk_full", "dependency_outage", "unknown")


class ToolRequest(BaseModel):
    tool: str = Field(max_length=64)
    args: dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = Field(default="", max_length=1500)
    tool_requests: list[ToolRequest] = Field(default_factory=list, max_length=8)
    final: bool = True
    # populated by the agent runtime (never by the model):
    blocked_requests: list[dict[str, Any]] = Field(default_factory=list)
    loop_detected: bool = False


class Cause(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class Signature(BaseModel):
    pattern: str
    category: str
    count: int
    first_seen: str | None = None
    sample: str = ""


class LogOutput(AgentOutput):
    error_line_count: int = 0
    onset: str | None = None
    error_signatures: list[Signature] = Field(default_factory=list)
    suspected_causes: list[Cause] = Field(default_factory=list)


class Anomaly(BaseModel):
    metric: str
    kind: str
    baseline: float
    last: float
    ratio: float
    onset: str | None = None


class MetricsOutput(AgentOutput):
    anomalies: list[Anomaly] = Field(default_factory=list)
    suspected_causes: list[Cause] = Field(default_factory=list)


class Change(BaseModel):
    id: str
    service: str
    type: str
    version: str | None = None
    previous_version: str | None = None
    minutes_before_alert: float
    summary: str = ""
    author: str = ""
    config_key: str | None = None
    previous_revision: str | None = None


class DeployOutput(AgentOutput):
    recent_changes: list[Change] = Field(default_factory=list)
    suspect_change: Change | None = None
    suspected_causes: list[Cause] = Field(default_factory=list)


class RunbookMatch(BaseModel):
    doc_id: str
    title: str
    score: float
    category: str = "unknown"


class RunbookOutput(AgentOutput):
    matches: list[RunbookMatch] = Field(default_factory=list)
    recommended_steps: list[str] = Field(default_factory=list)
    suspected_causes: list[Cause] = Field(default_factory=list)


class RootCause(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)
    contributions: dict[str, float] = Field(default_factory=dict)


class RemediationOutput(AgentOutput):
    root_cause: RootCause
    proposed_actions: list[ActionProposal] = Field(default_factory=list, max_length=6)


class CommsOutput(AgentOutput):
    title: str = ""
    body: str = Field(default="", max_length=3000)
    audience: str = "internal"
    next_update_minutes: int = 30
