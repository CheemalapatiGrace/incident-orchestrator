"""Command line interface: ``python -m orchestrator <command>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from orchestrator.config import Config
from orchestrator.core.store import Store
from orchestrator.engine import Orchestrator
from orchestrator.hitl.terminal import TerminalApprover
from orchestrator.models import RunStatus
from orchestrator.sim.scenarios import SCENARIO_IDS, build_scenario

SHOWN = {"task.started", "task.succeeded", "task.retry", "task.dead_lettered", "task.held", "approval.requested", "approval.decided", "action.executed", "action.failed", "guardrail.injection_detected", "guardrail.permission_denied", "guardrail.action_blocked", "guardrail.loop_detected", "guardrail.output_repaired", "run.paused", "run.resumed", "run.aborted", "plan.created", "human.action"}


def _line(ev: Any) -> str | None:
    if ev.type not in SHOWN:
        return None
    d = ev.data
    if ev.type == "plan.created":
        return f"  plan: {[t['id'] for t in d['plan']['tasks']]}"
    detail = d.get("task_id") or d.get("action") or d.get("tool") or d.get("approval_id") or d.get("rule") or ""
    extra = d.get("error") or d.get("reason") or d.get("summary") or d.get("status") or ""
    return f"  {ev.type:<32} {str(detail):<22} {str(extra)[:90]}"


async def _print_events(engine: Orchestrator, run_id: str) -> None:
    with engine.bus.subscribe(run_id) as sub:
        while True:
            ev = await sub.get()
            if ev and (line := _line(ev)):
                print(line, flush=True)


def _engine(args: argparse.Namespace) -> Orchestrator:
    cfg = Config.from_env()
    if getattr(args, "db", None):
        cfg.db_path = args.db
    return Orchestrator(cfg)


async def cmd_run(args: argparse.Namespace) -> int:
    eng = _engine(args)
    alert = build_scenario(args.scenario).alert()
    run_id = eng.submit(alert)
    print(f"run {run_id}  scenario={args.scenario}  db={eng.cfg.db_path}")
    printer = asyncio.create_task(_print_events(eng, run_id))
    helpers = []
    if args.approve == "ask":
        helpers.append(asyncio.create_task(TerminalApprover(eng).watch(run_id)))
    elif args.approve in {"auto", "reject"}:
        async def proxy() -> None:  # explicit, audited opt-in: the actor is recorded as cli:auto-*
            while True:
                for row in eng.store.list_approvals(run_id, "pending"):
                    if args.approve == "auto":
                        eng.hitl.approve(row["approval_id"], "cli:auto-approve", "auto-approved via --approve auto")
                    else:
                        eng.hitl.reject(row["approval_id"], "cli:auto-reject", "rejected via --approve reject")
                await asyncio.sleep(0.05)
        helpers.append(asyncio.create_task(proxy()))
    else:
        print("  (no approver attached: actions will wait for `approvals approve`, or time out)")
    res = await eng.execute(run_id)
    await asyncio.sleep(0.05)
    for t in [printer, *helpers]:
        t.cancel()
    assert res is not None
    print(json.dumps(res.model_dump(mode="json"), indent=2, default=str))
    return 0 if res.status in {RunStatus.COMPLETED, RunStatus.DEGRADED} else 1


def cmd_approvals(args: argparse.Namespace) -> int:
    eng = _engine(args)
    if args.action == "list":
        rows = eng.store.list_approvals(status=None if args.all else "pending")
        for r in rows:
            p = r["proposal"]
            print(f"{r['approval_id']:<44} {r['status']:<9} {r['risk']['level']:<6} {p['action']} {json.dumps(p['params'])}")
        if not rows:
            print("no approvals")
        return 0
    actor = args.actor or "cli:operator"
    edits = json.loads(args.edit) if args.edit else None
    if args.action == "approve":
        print(json.dumps(eng.hitl.approve(args.id, actor, args.note or "", edits)))
    else:
        print(json.dumps(eng.hitl.reject(args.id, actor, args.note or "")))
    return 0


async def cmd_resume(args: argparse.Namespace) -> int:
    eng = _engine(args)
    res = await eng.execute(args.run_id)
    print(json.dumps(res.model_dump(mode="json") if res else {"note": "run is leased by another worker"}, indent=2, default=str))
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    store = Store(args.db or Config.from_env().db_path)
    for r in store.list_runs(50):
        print(f"{r['run_id']:<18} {r['status']:<18} {r['incident']['service']:<22} {r['incident']['title'][:50]}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    store = Store(args.db or Config.from_env().db_path)
    spans = store.get_spans(args.run_id)
    kids: dict[str | None, list[dict]] = {}
    for s in spans:
        kids.setdefault(s["parent_id"], []).append(s)

    def walk(pid: str | None, depth: int) -> None:
        for s in kids.get(pid, []):
            a = s["attrs"] or {}
            cost = f" ${a['cost_usd']:.4f} {a.get('tokens_in', 0)}+{a.get('tokens_out', 0)}tok" if "cost_usd" in a else ""
            print(f"{'  ' * depth}{s['kind']:<5} {s['name']:<28} {s['latency_ms']:>8.1f}ms {s['status']}{cost}")
            walk(s["span_id"], depth + 1)

    walk(None, 0)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from orchestrator.api.app import create_app

    cfg = Config.from_env()
    if args.db:
        cfg.db_path = args.db
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="orchestrator", description="Multi-agent incident response orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scenarios", help="list bundled incident scenarios")
    r = sub.add_parser("run", help="run a scenario in-process")
    r.add_argument("scenario", choices=SCENARIO_IDS)
    r.add_argument("--db", help="SQLite checkpoint file (default: env IO_DB_PATH)")
    r.add_argument("--approve", choices=["ask", "auto", "reject", "none"], default="ask", help="how to handle approval gates (default: ask in the terminal)")
    a = sub.add_parser("approvals", help="list/approve/reject pending approvals in a checkpoint DB (works across processes)")
    a.add_argument("action", choices=["list", "approve", "reject"])
    a.add_argument("id", nargs="?")
    a.add_argument("--db")
    a.add_argument("--note")
    a.add_argument("--actor")
    a.add_argument("--edit", help="JSON param overrides when approving")
    a.add_argument("--all", action="store_true")
    rs = sub.add_parser("resume", help="resume a crashed/interrupted run from its last checkpoint")
    rs.add_argument("run_id")
    rs.add_argument("--db")
    ls = sub.add_parser("runs", help="list runs in a checkpoint DB")
    ls.add_argument("--db")
    tr = sub.add_parser("trace", help="print the span tree of a run")
    tr.add_argument("run_id")
    tr.add_argument("--db")
    s = sub.add_parser("serve", help="start the API + dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--db")
    args = p.parse_args(argv)

    if args.cmd == "scenarios":
        for sid in SCENARIO_IDS:
            sc = build_scenario(sid)
            print(f"{sid:<34} {sc.severity.value}  {sc.service:<22} root cause: {sc.root_cause}")
        return 0
    if args.cmd == "run":
        return asyncio.run(cmd_run(args))
    if args.cmd == "approvals":
        if args.action != "list" and not args.id:
            p.error("approve/reject need an approval id")
        return cmd_approvals(args)
    if args.cmd == "resume":
        return asyncio.run(cmd_resume(args))
    if args.cmd == "runs":
        return cmd_runs(args)
    if args.cmd == "trace":
        return cmd_trace(args)
    if args.cmd == "serve":
        return cmd_serve(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
