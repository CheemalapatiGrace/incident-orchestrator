"""Plan validation and DAG utilities."""

from __future__ import annotations

from collections import defaultdict, deque

from orchestrator.errors import PlanValidationError
from orchestrator.models import AgentName, Plan

MAX_TASKS = 24


def validate_plan(plan: Plan, *, required_agents: frozenset[AgentName] = frozenset()) -> None:
    """Raise :class:`PlanValidationError` unless the plan is a well-formed DAG."""
    issues: list[str] = []
    ids = [t.id for t in plan.tasks]
    if not ids:
        issues.append("plan has no tasks")
    if len(ids) > MAX_TASKS:
        issues.append(f"plan has {len(ids)} tasks (max {MAX_TASKS})")
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        issues.append(f"duplicate task ids: {dupes}")
    idset = set(ids)
    for t in plan.tasks:
        for d in t.depends_on:
            if d not in idset:
                issues.append(f"task {t.id!r} depends on unknown task {d!r}")
            if d == t.id:
                issues.append(f"task {t.id!r} depends on itself")
    if not issues and _has_cycle(plan):
        issues.append("dependency cycle detected")
    present = {t.agent for t in plan.tasks}
    for a in required_agents:
        if a not in present:
            issues.append(f"required agent {a.value!r} missing from plan")
    if issues:
        raise PlanValidationError(issues)


def _has_cycle(plan: Plan) -> bool:
    indeg = {t.id: len(set(t.depends_on)) for t in plan.tasks}
    children: dict[str, list[str]] = defaultdict(list)
    for t in plan.tasks:
        for d in set(t.depends_on):
            children[d].append(t.id)
    q = deque(i for i, n in indeg.items() if n == 0)
    seen = 0
    while q:
        n = q.popleft()
        seen += 1
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    return seen != len(indeg)


def topo_levels(plan: Plan) -> list[list[str]]:
    """Group task ids into parallelisable layers (used for layout and for tests)."""
    level: dict[str, int] = {}
    by_id = {t.id: t for t in plan.tasks}

    def depth(i: str) -> int:
        if i not in level:
            level[i] = 0 if not by_id[i].depends_on else 1 + max(depth(d) for d in by_id[i].depends_on)
        return level[i]

    for t in plan.tasks:
        depth(t.id)
    out: list[list[str]] = [[] for _ in range(max(level.values(), default=-1) + 1)]
    for i, lv in level.items():
        out[lv].append(i)
    return out


def descendants(plan: Plan, task_id: str) -> set[str]:
    children: dict[str, list[str]] = defaultdict(list)
    for t in plan.tasks:
        for d in t.depends_on:
            children[d].append(t.id)
    seen: set[str] = set()
    stack = [task_id]
    while stack:
        for c in children[stack.pop()]:
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen
