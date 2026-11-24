"""Throughput / latency benchmark: ``python -m orchestrator.bench``.

N concurrent incidents flow through the queue + worker pool with simulated I/O latency (LLM and tool
calls sleep) so concurrency matters the way it would against real services. The numbers characterise
the orchestration layer (scheduling, checkpointing, tracing), not model quality.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import tempfile
import time
from typing import Any

from orchestrator.chaos import auto_approve, chaos_config
from orchestrator.engine import Orchestrator, default_env_factory
from orchestrator.llm.mock import MockLLM
from orchestrator.sim.environment import SimulatedEnvironment
from orchestrator.sim.scenarios import SCENARIO_IDS, build_scenario


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def bench_once(n: int, workers: int, llm_ms: float, tool_ms: float, on_disk: bool) -> dict[str, Any]:
    db = ":memory:"
    if on_disk:
        db = os.path.join(tempfile.mkdtemp(prefix="io-bench-"), "bench.db")
    cfg = chaos_config(db_path=db, agent_concurrency=max(workers, 6), tool_concurrency=max(workers * 2, 16), worker_concurrency=workers, max_parallel_tasks=8)
    eng = Orchestrator(cfg, llm=MockLLM(latency_s=llm_ms / 1000),
                       env_factory=lambda a: SimulatedEnvironment.for_scenario(a.labels["scenario"], latency_s=tool_ms / 1000) if a.labels.get("scenario") else default_env_factory(a))
    done: dict[str, float] = {}
    finished: dict[str, float] = {}
    submitted: dict[str, float] = {}
    sub = eng.bus.subscribe(None)
    sub.__enter__()

    started: dict[str, float] = {}

    async def collect() -> None:
        while len(done) < n:
            ev = await sub.get()
            if not ev:
                continue
            if ev.type == "span.started" and ev.data.get("kind") == "plan":
                started[ev.run_id] = ev.ts
            elif ev.type == "run.completed":
                done[ev.run_id] = time.perf_counter()
                finished[ev.run_id] = ev.ts

    stop = asyncio.Event()
    approver = asyncio.create_task(auto_approve(eng, stop))
    collector = asyncio.create_task(collect())
    eng.start_workers(workers)
    t0 = time.perf_counter()
    for i in range(n):
        rid = eng.submit(build_scenario(SCENARIO_IDS[i % 6]).alert().model_copy(update={"id": f"bench-{i}"}))
        submitted[rid] = time.perf_counter()
        await eng.queue.enqueue(rid)
    await asyncio.wait_for(collector, timeout=300)
    wall = time.perf_counter() - t0
    stop.set()
    await approver
    await eng.stop_workers()
    sub.__exit__()
    lat = [(done[r] - submitted[r]) * 1000 for r in done]
    svc = [(finished[r] - started[r]) * 1000 for r in done if r in started]
    statuses = [eng.store.get_run(r)["status"] for r in done]
    return {
        "workers": workers, "runs": n, "storage": "sqlite-file" if on_disk else "sqlite-memory", "wall_s": round(wall, 2), "throughput_rps": round(n / wall, 1),
        "p50_ms": round(pct(svc, 50)), "p95_ms": round(pct(svc, 95)), "p99_ms": round(pct(svc, 99)), "e2e_p95_ms": round(pct(lat, 95)), "mean_ms": round(statistics.mean(svc)),
        "ok": sum(s in {"completed", "degraded"} for s in statuses),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m orchestrator.bench")
    ap.add_argument("-n", "--runs", type=int, default=120)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--llm-ms", type=float, default=20.0, help="simulated LLM latency per call")
    ap.add_argument("--tool-ms", type=float, default=5.0, help="simulated tool latency per call")
    ap.add_argument("--memory", action="store_true", help="use :memory: SQLite instead of an on-disk WAL file")
    args = ap.parse_args(argv)
    print(f"\nBENCHMARK  runs={args.runs}  simulated latency: llm={args.llm_ms}ms tool={args.tool_ms}ms (6-7 LLM + ~8 tool calls per incident)\n")
    print("per-run latency = execution time (plan start -> completion); e2e_p95 includes time queued behind other runs\n")
    print(f"{'workers':>8}{'storage':>16}{'wall_s':>9}{'runs/s':>9}{'p50_ms':>9}{'p95_ms':>9}{'p99_ms':>9}{'e2e_p95':>9}{'ok':>9}")
    rows = []
    for w in args.workers:
        r = asyncio.run(bench_once(args.runs, w, args.llm_ms, args.tool_ms, not args.memory))
        rows.append(r)
        print(f"{r['workers']:>8}{r['storage']:>16}{r['wall_s']:>9}{r['throughput_rps']:>9}{r['p50_ms']:>9}{r['p95_ms']:>9}{r['p99_ms']:>9}{r['e2e_p95_ms']:>9}{r['ok']:>5}/{r['runs']}")
    return 0 if all(r["ok"] == r["runs"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
