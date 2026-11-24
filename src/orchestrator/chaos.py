"""Chaos harness: ``python -m orchestrator.chaos``. Injects tool failures, timeouts, malformed LLM
output, hard dependency outages and process crashes, then asserts the system's invariants:

  * no mutating action ever executes without approval, and none outside the expected set
  * runs still finish (completed/degraded) and still find the right root cause
  * a crashed run resumes from its last checkpoint without re-executing completed work
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from collections import Counter
from typing import Any

from orchestrator.config import Config
from orchestrator.engine import Orchestrator
from orchestrator.errors import SimulatedCrash
from orchestrator.faults import FaultInjector, FaultRule
from orchestrator.llm.mock import MockLLM
from orchestrator.sim.scenarios import SCENARIO_IDS, build_scenario


def chaos_config(**kw: Any) -> Config:
    base: dict[str, Any] = dict(db_path=":memory:", retry_base_delay_s=0.002, retry_max_delay_s=0.02, tool_timeout_s=0.25, llm_timeout_s=0.3, task_timeout_s=5.0, approval_poll_s=0.01, agent_concurrency=32, tool_concurrency=64)
    base.update(kw)
    return Config(**base)


async def auto_approve(eng: Orchestrator, stop: asyncio.Event) -> None:
    """Scripted human: approve only the scenario's ground-truth action."""
    while not stop.is_set():
        for ap in eng.store.list_approvals(status="pending"):
            run = eng.store.get_run(ap["run_id"])
            if run is None:
                continue
            exp = {(a["action"], a["service"]) for a in build_scenario(run["incident"]["labels"]["scenario"]).correct_actions}
            p = ap["proposal"]
            try:
                if (p["action"], p["params"].get("service")) in exp:
                    eng.hitl.approve(ap["approval_id"], "chaos:approver", "ground-truth action")
                else:
                    eng.hitl.reject(ap["approval_id"], "chaos:approver", "unexpected action")
            except Exception:  # noqa: BLE001
                pass
        await asyncio.sleep(0.005)


async def soak(n: int, rate: float, seed: int) -> dict[str, Any]:
    inj = FaultInjector.random_chaos(rate, seed)
    # breakers are tuned permissive here so the soak exercises retries/dead-letters; breaker_demo() covers tripping
    eng = Orchestrator(chaos_config(breaker_failure_threshold=40, breaker_recovery_s=0.2), llm=MockLLM(), injector=inj)
    stop = asyncio.Event()
    approver = asyncio.create_task(auto_approve(eng, stop))
    ids = [eng.submit(build_scenario(SCENARIO_IDS[i % 6]).alert().model_copy(update={"id": f"chaos-{i}"})) for i in range(n)]
    t0 = time.perf_counter()
    results = await asyncio.gather(*(eng.execute(r) for r in ids))
    dt = time.perf_counter() - t0
    stop.set()
    await approver
    ev = Counter()
    for r in ids:
        for e in eng.store.get_events(r, limit=100000):
            ev[e["type"]] += 1
    ok = [r for r in results if r and r.status.value in {"completed", "degraded"}]
    correct = sum(1 for r in results if r and r.root_cause == build_scenario(eng.store.get_run(r.run_id)["incident"]["labels"]["scenario"]).root_cause)
    unsafe = 0
    for r in ids:
        env = eng._envs[r]
        exp = {(a["action"], a["service"]) for a in env.scenario.correct_actions}
        unsafe += sum(1 for m in env.mutations if (m["action"], m["params"]["service"]) not in exp)
    return {
        "profile": f"random faults @ {rate:.0%}", "runs": n, "finished": len(ok), "correct_rc": correct, "completed": sum(1 for r in results if r and r.status.value == "completed"),
        "degraded": sum(1 for r in results if r and r.status.value == "degraded"), "failed": n - len(ok), "faults_injected": inj.total_injected(),
        "tool_retries": ev["tool.retry"], "task_retries": ev["task.retry"], "dead_letters": ev["task.dead_lettered"], "repairs": ev["guardrail.output_repaired"],
        "plan_fallbacks": ev["plan.fallback"], "unsafe_mutations": unsafe, "wall_s": round(dt, 2),
    }


async def breaker_demo() -> dict[str, Any]:
    """A hard dependency outage: breaker opens, subsequent calls fail fast, then half-open recovery."""
    inj = FaultInjector([FaultRule("tool:search_logs", "permanent_error", None)])
    eng = Orchestrator(chaos_config(breaker_failure_threshold=3, breaker_recovery_s=0.4, max_task_attempts=1, max_tool_attempts=1), llm=MockLLM(), injector=inj)
    stop = asyncio.Event()
    approver = asyncio.create_task(auto_approve(eng, stop))

    async def one(i: int) -> Any:
        rid = eng.submit(build_scenario("checkout-bad-deploy").alert().model_copy(update={"id": f"br-{i}"}))
        return await eng.execute(rid)

    for i in range(4):
        await one(i)
    b = eng.deps.breakers.get("tool:search_logs")
    ab = eng.deps.breakers.get("agent:log_analyst")
    opened = f"tool={b.state.value}, agent={ab.state.value}"
    rejected_before = b.rejected_count + ab.rejected_count
    await asyncio.sleep(0.45)  # recovery window elapses -> half-open
    inj.rules.clear()  # dependency recovers
    half = b.state.value
    res = await one(99)
    stop.set()
    await approver
    return {"profile": "hard outage + breakers", "state_after_outage": opened, "fast_failed_calls": rejected_before, "state_after_wait": half, "state_after_recovery": f"tool={b.state.value}, agent={ab.state.value}", "recovered_run_status": res.status.value, "opened_count": b.opened_count + ab.opened_count}


async def crash_resume(points: list[int]) -> dict[str, Any]:
    ok = 0
    detail = []
    for k in points:
        path = os.path.join(tempfile.mkdtemp(prefix="io-chaos-"), "run.db")
        inj = FaultInjector(crash_after_tasks=k)
        e1 = Orchestrator(chaos_config(db_path=path, lease_s=0.2), llm=MockLLM(), injector=inj, worker_id="w1")
        rid = e1.submit(build_scenario("checkout-bad-deploy").alert())
        stop = asyncio.Event()
        ap = asyncio.create_task(auto_approve(e1, stop))
        crashed = False
        try:
            await e1.execute(rid)
        except SimulatedCrash:
            crashed = True
        stop.set()
        await ap
        done_before = {t for t, s in e1.store.get_tasks(rid).items() if s["status"] == "succeeded"}
        await asyncio.sleep(0.25)  # w1's lease expires -> another worker may take over
        e1.store.close()
        # a brand-new "process": fresh engine, same database file
        e2 = Orchestrator(chaos_config(db_path=path), llm=MockLLM(), worker_id="w2")
        stop2 = asyncio.Event()
        ap2 = asyncio.create_task(auto_approve(e2, stop2))
        res = await e2.execute(rid)
        stop2.set()
        await ap2
        tasks = e2.store.get_tasks(rid)
        redone = [t for t in done_before if e2.runtimes[rid].tasks[t].attempts == 0 and False]
        rec = [s for s in e2.store.get_spans(rid) if s["kind"] == "task" and s["name"] in done_before]
        reran = len(rec) > len(done_before)  # a completed task got a second task span => re-executed
        good = crashed and res is not None and res.status.value == "completed" and res.root_cause == "bad_deploy" and not reran and all(t["status"] == "succeeded" for t in tasks.values())
        ok += good
        detail.append((k, sorted(done_before), "ok" if good else "FAIL"))
        _ = redone
    return {"profile": "crash + resume", "crash_points": len(points), "resumed_ok": ok, "detail": detail}


async def crash_at_gate() -> dict[str, Any]:
    """Process dies while a human decision is pending; the approval survives and the action runs exactly once."""
    path = os.path.join(tempfile.mkdtemp(prefix="io-chaos-"), "gate.db")
    e1 = Orchestrator(chaos_config(db_path=path), llm=MockLLM(), worker_id="w1")
    rid = e1.submit(build_scenario("search-traffic-spike").alert())
    task = asyncio.create_task(e1.execute(rid))
    for _ in range(400):
        if e1.store.list_approvals(rid, "pending"):
            break
        await asyncio.sleep(0.01)
    pending_before = [a["approval_id"] for a in e1.store.list_approvals(rid, "pending")]
    task.cancel()  # "kill -9" while waiting for a human
    await asyncio.gather(task, return_exceptions=True)
    e1.store.close()
    e2 = Orchestrator(chaos_config(db_path=path), llm=MockLLM(), worker_id="w2")
    run2 = asyncio.create_task(e2.execute(rid))
    for _ in range(400):
        if e2.store.list_approvals(rid, "pending"):
            break
        await asyncio.sleep(0.01)
    still_pending = [a["approval_id"] for a in e2.store.list_approvals(rid, "pending")]
    e2.hitl.approve(still_pending[0], "chaos:approver", "approved after restart")
    res = await run2
    env = e2._envs[rid]
    return {"profile": "crash while awaiting approval", "pending_before_crash": pending_before, "same_approval_after_restart": pending_before == still_pending, "mutations_executed": len(env.mutations), "final_status": res.status.value if res else None}


async def run_all(n: int) -> tuple[list[dict[str, Any]], bool]:
    rows = [await soak(n, 0.15, 1), await soak(n, 0.40, 2), await breaker_demo(), await crash_resume([1, 2, 3, 4, 5]), await crash_at_gate()]
    healthy = True
    for r in rows:
        if "unsafe_mutations" in r:
            healthy &= r["unsafe_mutations"] == 0 and r["finished"] / r["runs"] >= (0.90 if "15%" in r["profile"] else 0.60)
        elif "resumed_ok" in r:
            healthy &= r["resumed_ok"] == r["crash_points"]
        elif "pending_before_crash" in r:
            healthy &= r["same_approval_after_restart"] and r["mutations_executed"] == 1 and r["final_status"] == "completed"
        else:
            healthy &= "tool=open" in r["state_after_outage"] and r["fast_failed_calls"] > 0 and "tool=closed" in r["state_after_recovery"] and r["recovered_run_status"] in {"completed", "degraded"}
    return rows, healthy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m orchestrator.chaos")
    ap.add_argument("-n", type=int, default=60, help="incidents per soak profile")
    args = ap.parse_args(argv)
    rows, healthy = asyncio.run(run_all(args.n))
    print("\nCHAOS REPORT\n" + "=" * 72)
    for r in rows:
        print(f"\n[{r['profile']}]")
        for k, v in r.items():
            if k != "profile":
                print(f"  {k:<24} {v}")
    print("\nRESULT:", "PASS" if healthy else "FAIL")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
