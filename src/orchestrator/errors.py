"""Exception taxonomy.

The split between *transient* (retryable) and everything else drives the retry logic:
only :class:`TransientError` subclasses are retried by the node wrapper and the tool layer.
"""

from __future__ import annotations


class OrchestratorError(Exception):
    """Base class for all orchestrator errors."""


class TransientError(OrchestratorError):
    """A failure that may succeed if retried (timeouts, flaky tools, open breakers)."""


class ToolExecutionError(TransientError):
    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"tool {tool!r} failed: {message}")
        self.tool = tool


class ToolTimeout(TransientError):
    pass


class TaskTimeout(TransientError):
    pass


class LLMError(TransientError):
    pass


class OutputValidationError(TransientError):
    """The model produced output that does not satisfy the expected schema."""


class CircuitOpenError(TransientError):
    pass


class BulkheadFullError(TransientError):
    pass


class FatalToolError(OrchestratorError):
    """Non-retryable tool failure (bad arguments, unknown tool)."""


class PermissionDenied(OrchestratorError):
    """An agent attempted a tool/action outside its least-privilege policy."""

    def __init__(self, agent: str, tool: str, reason: str = "not permitted") -> None:
        super().__init__(f"agent {agent!r} may not use {tool!r}: {reason}")
        self.agent = agent
        self.tool = tool
        self.reason = reason


class BudgetExceeded(OrchestratorError):
    def __init__(self, dimension: str, used: float, limit: float) -> None:
        super().__init__(f"budget exceeded: {dimension} used={used:.4g} limit={limit:.4g}")
        self.dimension = dimension
        self.used = used
        self.limit = limit


class LoopDetected(OrchestratorError):
    pass


class PlanValidationError(OrchestratorError):
    def __init__(self, issues: list[str]) -> None:
        super().__init__("invalid plan: " + "; ".join(issues))
        self.issues = issues


class ApprovalError(OrchestratorError):
    pass


class InvalidState(OrchestratorError):
    """A human/API action is not valid in the current run/task state."""


class TaskFailure(OrchestratorError):
    """A task exhausted its retries (or hit a non-retryable error)."""

    def __init__(self, task_id: str, cause: BaseException, attempts: int) -> None:
        super().__init__(f"task {task_id!r} failed after {attempts} attempt(s): {cause}")
        self.task_id = task_id
        self.cause = cause
        self.attempts = attempts


class SimulatedCrash(BaseException):  # noqa: N818 - deliberately not an Exception
    """Raised by the fault injector to emulate a hard process crash (kill -9)."""
