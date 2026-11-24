"""The typed LangGraph state of one incident run, its reducers, and the pure scheduling rules.

``RunState`` is what the LangGraph checkpointer persists after every superstep. Everything a node
needs is derived from it (plus the injected services), which is what makes a crashed run resumable
from the last completed superstep and a human edit a plain ``graph.update_state(...)``.

Parallel branches (several specialists finishing in the same superstep) write *patches* to
``tasks``; the :func:`merge_tasks` reducer folds them together per task id, so fan-in needs no
locking. A patch value of ``None`` deletes the task (used by plan edits).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from orchestrator.models import TERMINAL_TASK_STATES, Plan, TaskSpec, TaskStatus

TERMINAL = frozenset(s.value for s in TERMINAL_TASK_STATES)
SUCCEEDED = TaskStatus.SUCCEEDED.value
SKIPPED = TaskStatus.SKIPPED.value
FAILED = TaskStatus.FAILED.value
PENDING = TaskStatus.PENDING.value
HELD = TaskStatus.HELD.value
AWAITING = TaskStatus.AWAITING_APPROVAL.value


def new_task() -> dict[str, Any]:
    """A task record as stored in graph state (JSON-primitive values only, so it checkpoints cleanly)."""
    return {
        "status": PENDING, "attempts": 0, "output": None, "error": None, "reason": None, "input_overrides": {},
        "released": False, "human_skipped": False, "started_at": None, "finished_at": None,
    }


# ------------------------------------------------------------------------------------ reducers
def merge_tasks(left: dict[str, dict] | None, right: dict[str, dict | None] | None) -> dict[str, dict]:
    out = dict(left or {})
    for tid, patch in (right or {}).items():
        if patch is None:
            out.pop(tid, None)
        else:
            out[tid] = {**out.get(tid, new_task()), **patch}
    return out


def merge_guidance(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    out = list(left or [])
    seen = {g["id"] for g in out}
    for g in right or []:
        if g["id"] not in seen:
            out.append(g)
            seen.add(g["id"])
    return out


def last(_old: Any, new: Any) -> Any:
    return new


def either(old: Any, new: Any) -> bool:
    return bool(old) or bool(new)


class RunState(TypedDict, total=False):
    incident: dict[str, Any]
    plan: Annotated[dict[str, Any] | None, last]
    tasks: Annotated[dict[str, dict[str, Any]], merge_tasks]
    guidance: Annotated[list[dict[str, Any]], merge_guidance]
    paused: Annotated[bool, last]
    cancelled: Annotated[bool, either]
    taint: Annotated[bool, either]
    abort: Annotated[dict[str, Any], last]  # {} = none; else {"status", "reason", "dimension"}
    status: Annotated[str, last]
    cmd_ids: Annotated[list[int], last]  # human commands folded into the state at the last dispatch


_REDUCERS = {"tasks": merge_tasks, "guidance": merge_guidance, "cancelled": either, "taint": either}


def fold(into: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply ``patch`` to a plain dict with the same reducer semantics LangGraph uses for ``RunState``."""
    for k, v in patch.items():
        red = _REDUCERS.get(k)
        into[k] = red(into.get(k), v) if red else v
    return into


def fold_patches(patches: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in patches:
        fold(out, p)
    return out


# ------------------------------------------------------------------------------ scheduling rules
def plan_of(state: dict[str, Any]) -> Plan | None:
    return Plan.model_validate(state["plan"]) if state.get("plan") else None


SHORT_CIRCUIT = "short_circuit:false_alarm"


def deliberate_skip(rec: dict[str, Any]) -> bool:
    """Skipped on purpose (by a human, or by the false-alarm short-circuit): satisfies dependants, no degradation."""
    return rec["status"] == SKIPPED and (rec.get("human_skipped", False) or rec.get("reason") == SHORT_CIRCUIT)


def is_ok(rec: dict[str, Any]) -> bool:
    return rec["status"] == SUCCEEDED or deliberate_skip(rec)


def _clean(out: dict[str, Any] | None) -> bool:
    out = out or {}
    return not (out.get("suspected_causes") or out.get("error_signatures") or out.get("anomalies") or out.get("suspect_change"))


def false_alarm(plan: Plan, tasks: dict[str, dict], severity: str, tainted: bool) -> bool:
    """Every first-line diagnostic (logs / metrics / deploys) finished and found nothing, on a sev3, untainted run."""
    from orchestrator.models import AgentName

    triage = [t for t in plan.tasks if t.agent in (AgentName.LOG_ANALYST, AgentName.METRICS_ANALYST, AgentName.DEPLOY_HISTORY)]
    return bool(
        triage and severity == "sev3" and not tainted
        and all(tasks[t.id]["status"] == SUCCEEDED and _clean(tasks[t.id]["output"]) for t in triage)
    )


def dep_state(plan: Plan, tasks: dict[str, dict], spec: TaskSpec) -> str:
    """``wait`` | ``blocked`` (a required upstream failed) | ``ready``."""
    for dep in spec.depends_on:
        rec = tasks.get(dep)
        if rec is None or rec["status"] not in TERMINAL:
            return "wait"
        if is_ok(rec) or plan.get(dep).optional:
            continue
        return "blocked"
    return "ready"


def upstream_of(plan: Plan, tasks: dict[str, dict], spec: TaskSpec) -> list[list[Any]]:
    out = []
    for dep in spec.depends_on:
        rec = tasks[dep]
        payload = rec["output"] if is_ok(rec) and rec["output"] is not None else {
            "_unavailable": True, "reason": rec.get("reason") or rec.get("error") or rec["status"],
        }
        out.append([dep, plan.get(dep).agent.value, payload])
    return out
