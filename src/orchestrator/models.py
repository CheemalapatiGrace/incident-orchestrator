"""Core pydantic models shared across the package."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class Severity(StrEnum):
    SEV1 = "sev1"
    SEV2 = "sev2"
    SEV3 = "sev3"


class Alert(BaseModel):
    """The incoming alert payload that starts a run."""

    id: str = Field(min_length=1, max_length=128)
    service: str = Field(min_length=1, max_length=128)
    title: str = Field(max_length=500)
    severity: Severity = Severity.SEV2
    fired_at: datetime = Field(default_factory=utcnow)
    description: str = Field(default="", max_length=4000)
    labels: dict[str, str] = Field(default_factory=dict)


class AgentName(StrEnum):
    LOG_ANALYST = "log_analyst"
    METRICS_ANALYST = "metrics_analyst"
    DEPLOY_HISTORY = "deploy_history"
    RUNBOOK = "runbook"
    REMEDIATION = "remediation"
    COMMS = "comms"


class TaskStatus(StrEnum):
    PENDING = "pending"
    HELD = "held"  # breakpoint: waiting for a human to release it
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


TERMINAL_TASK_STATES = frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.SKIPPED})
ACTIVE_TASK_STATES = frozenset({TaskStatus.RUNNING, TaskStatus.AWAITING_APPROVAL})


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    DEGRADED = "degraded"  # finished, but some optional task failed
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"


TERMINAL_RUN_STATES = frozenset(
    {RunStatus.COMPLETED, RunStatus.DEGRADED, RunStatus.FAILED, RunStatus.BUDGET_EXCEEDED, RunStatus.CANCELLED}
)


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    agent: AgentName
    description: str = Field(default="", max_length=300)
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float | None = Field(default=None, gt=0, le=600)
    max_attempts: int | None = Field(default=None, ge=1, le=10)
    optional: bool = True  # may the run proceed if this task fails?
    breakpoint: bool = False  # hold for a human before executing


class Plan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tasks: list[TaskSpec]
    rationale: str = ""

    def get(self, task_id: str) -> TaskSpec:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(task_id)


class Budget(BaseModel):
    max_steps: int = 80
    max_tokens: int = 80_000
    max_cost_usd: float = 0.50
    max_wall_clock_s: float = 120.0


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionProposal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: str = Field(max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=600)
    expected_effect: str = Field(default="", max_length=300)


class RunResult(BaseModel):
    run_id: str
    status: RunStatus
    root_cause: str | None = None
    confidence: float | None = None
    summary: str = ""
    actions: list[dict[str, Any]] = Field(default_factory=list)
    comms: dict[str, Any] | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    dead_letters: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)
    security_findings: int = 0
