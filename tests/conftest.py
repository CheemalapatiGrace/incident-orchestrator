from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import pytest

from orchestrator.config import Config
from orchestrator.core.store import Store
from orchestrator.engine import Orchestrator
from orchestrator.faults import FaultInjector
from orchestrator.llm.mock import MockLLM
from orchestrator.sim.environment import Poison, SimulatedEnvironment
from orchestrator.sim.scenarios import build_scenario


def fast_cfg(**kw: Any) -> Config:
    base: dict[str, Any] = dict(
        db_path=":memory:", retry_base_delay_s=0.001, retry_max_delay_s=0.005, tool_timeout_s=0.3, llm_timeout_s=0.5,
        task_timeout_s=5.0, approval_poll_s=0.01,
    )
    base.update(kw)
    return Config(**base)


def make_engine(*, llm: MockLLM | None = None, injector: FaultInjector | None = None, poisons: list[Poison] | None = None,
                latency_s: float = 0.0, store: Store | None = None, **cfg_kw: Any) -> Orchestrator:
    return Orchestrator(
        fast_cfg(**cfg_kw), store=store, llm=llm or MockLLM(), injector=injector,
        env_factory=lambda a: SimulatedEnvironment.for_scenario(a.labels["scenario"], poisons, latency_s),
    )


def alert_for(sid: str, **upd: Any):
    return build_scenario(sid).alert().model_copy(update=upd)


async def wait_until(pred: Callable[[], Any], timeout: float = 5.0, step: float = 0.005) -> Any:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        await asyncio.sleep(step)
    raise AssertionError("condition not met in time")


async def wait_pending(eng: Orchestrator, run_id: str, n: int = 1, timeout: float = 5.0) -> list[dict]:
    return await wait_until(lambda: (p := eng.store.list_approvals(run_id, "pending")) and len(p) >= n and p, timeout)


async def approve_when_pending(eng: Orchestrator, run_id: str, note: str = "reviewed evidence", actor: str = "tester") -> None:
    while True:
        for ap in eng.store.list_approvals(run_id, "pending"):
            eng.hitl.approve(ap["approval_id"], actor, note)
        await asyncio.sleep(0.005)


async def run_with_approval(eng: Orchestrator, run_id: str, **kw: Any):
    t = asyncio.create_task(approve_when_pending(eng, run_id, **kw))
    try:
        return await eng.execute(run_id)
    finally:
        t.cancel()


def events(eng: Orchestrator, run_id: str, type_: str | None = None) -> list[dict]:
    evs = eng.store.get_events(run_id, limit=100000)
    return [e for e in evs if type_ is None or e["type"] == type_ or e["type"].startswith(type_ + ".")]


@pytest.fixture
def engine() -> Orchestrator:
    return make_engine()
