"""HTTP API.

REST for runs and human interventions, Server-Sent Events for the live event stream (replayable via
``Last-Event-ID``), and the no-build-step dashboard at ``/``.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from orchestrator.config import Config
from orchestrator.engine import Orchestrator
from orchestrator.errors import ApprovalError, InvalidState, PlanValidationError
from orchestrator.models import TERMINAL_RUN_STATES, Alert, Budget, RunStatus, Severity
from orchestrator.sim.environment import Poison, SimulatedEnvironment
from orchestrator.sim.scenarios import SCENARIO_IDS, build_scenario

STATIC = Path(__file__).parent / "static"


class PoisonIn(BaseModel):
    source: str = Field(pattern="^(logs|deploy_summary|runbook|service_note|metric_unit)$")
    text: str = Field(max_length=500)
    level: str = "ERROR"
    doc_id: str = "rb-bad-deploy"


class IncidentIn(BaseModel):
    scenario: str | None = None
    alert: Alert | None = None
    breakpoints: list[str] = Field(default_factory=list, max_length=10)
    budget: Budget | None = None
    poisons: list[PoisonIn] = Field(default_factory=list, max_length=5)  # simulate attacker-controlled data


class Decision(BaseModel):
    actor: str = Field(default="api-user", max_length=64)
    note: str = Field(default="", max_length=1000)
    edits: dict[str, Any] | None = None


class Actor(BaseModel):
    actor: str = Field(default="api-user", max_length=64)
    reason: str = Field(default="", max_length=500)


class SkipIn(Actor):
    output: dict[str, Any] | None = None


class RetryIn(Actor):
    reset_downstream: bool = True


class InputIn(Actor):
    params: dict[str, Any]


class PlanEditIn(Actor):
    ops: list[dict[str, Any]] = Field(max_length=20)


class GuidanceIn(Actor):
    text: str = Field(min_length=1, max_length=1000)
    task_id: str | None = None


def create_app(config: Config | None = None, engine: Orchestrator | None = None, *, start_workers: bool = True) -> FastAPI:
    cfg = config or Config.from_env()
    eng = engine or Orchestrator(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_workers:
            eng.start_workers()
            for run in eng.store.list_runs(1000):  # durable restart: pick unfinished runs back up
                if RunStatus(run["status"]) not in TERMINAL_RUN_STATES:
                    await eng.queue.enqueue(run["run_id"])
        yield
        await eng.stop_workers()

    app = FastAPI(title="incident-orchestrator", version="0.1.0", lifespan=lifespan)
    app.state.engine = eng

    def auth(request: Request, authorization: str | None = Header(default=None), token: str | None = Query(default=None)) -> None:
        expected = eng.cfg.api_token
        if not expected:
            return
        supplied = (authorization or "").removeprefix("Bearer ").strip() or (token or "")
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(401, "missing or invalid API token")

    secured = [Depends(auth)]

    @app.exception_handler(ApprovalError)
    @app.exception_handler(InvalidState)
    async def conflict(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(PlanValidationError)
    async def invalid(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    def run_or_404(run_id: str) -> dict:
        row = eng.store.get_run(run_id)
        if not row:
            raise HTTPException(404, f"unknown run {run_id!r}")
        return row

    # ------------------------------------------------------------------- public
    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "llm": eng.deps.llm.name, "auth": bool(eng.cfg.api_token)}

    # ---------------------------------------------------------------- read APIs
    @app.get("/api/scenarios", dependencies=secured)
    async def scenarios() -> list[dict[str, Any]]:
        out = []
        for sid in SCENARIO_IDS:
            s = build_scenario(sid)
            out.append({"id": sid, "title": s.title, "service": s.service, "severity": s.severity.value})
        return out

    @app.get("/api/policy", dependencies=secured)
    async def policy() -> dict[str, Any]:
        return {"read_tools": eng.deps.policy.matrix(), "may_propose": sorted(eng.deps.policy.may_propose), "mutating_executor": "approval_gate"}

    @app.get("/api/stats", dependencies=secured)
    async def stats() -> dict[str, Any]:
        runs = eng.store.list_runs(1000)
        by_status: dict[str, int] = {}
        for r in runs:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        return {**eng.stats(), "queue_depth": await eng.queue.size(), "runs_by_status": by_status}

    @app.get("/api/runs", dependencies=secured)
    async def list_runs(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
        return [
            {"run_id": r["run_id"], "status": r["status"], "service": r["incident"]["service"], "title": r["incident"]["title"],
             "severity": r["incident"]["severity"], "created_at": r["created_at"], "taint": r["taint"], "paused": r["paused"],
             "root_cause": (r["result"] or {}).get("root_cause")}
            for r in eng.store.list_runs(limit)
        ]

    @app.get("/api/runs/{run_id}", dependencies=secured)
    async def get_run(run_id: str) -> dict[str, Any]:
        row = run_or_404(run_id)
        rt = eng.runtimes.get(run_id)
        plan = rt.plan.model_dump(mode="json") if rt and rt.plan else row["plan"]
        saved = eng.store.get_tasks(run_id)
        tasks: dict[str, Any] = {}
        for spec in (plan or {}).get("tasks", []):
            live = rt.tasks.get(spec["id"]) if rt else None
            t = live.row() if live else saved.get(spec["id"], {"task_id": spec["id"], "status": "pending", "attempts": 0, "output": None})
            tasks[spec["id"]] = {**t, "spec": spec}
        return {
            "run": {k: row[k] for k in ("run_id", "status", "incident", "paused", "taint", "usage", "result", "created_at", "updated_at")},
            "plan": plan, "tasks": tasks, "approvals": eng.store.list_approvals(run_id),
            "guidance": (rt.guidance if rt else row["guidance"]) or [], "dead_letters": eng.store.get_dead_letters(run_id),
        }

    @app.get("/api/runs/{run_id}/trace", dependencies=secured)
    async def trace(run_id: str) -> list[dict[str, Any]]:
        run_or_404(run_id)
        return eng.store.get_spans(run_id)

    @app.get("/api/runs/{run_id}/audit", dependencies=secured)
    async def audit(run_id: str) -> list[dict[str, Any]]:
        run_or_404(run_id)
        return eng.store.get_audit(run_id)

    @app.get("/api/runs/{run_id}/dead-letters", dependencies=secured)
    async def dead_letters(run_id: str) -> list[dict[str, Any]]:
        run_or_404(run_id)
        return eng.store.get_dead_letters(run_id)

    @app.get("/api/approvals", dependencies=secured)
    async def approvals(status: str | None = "pending") -> list[dict[str, Any]]:
        return eng.store.list_approvals(status=status or None)

    @app.get("/api/runs/{run_id}/events", dependencies=secured)
    async def events(run_id: str, request: Request, last_event_id: str | None = Header(default=None), after: int = 0, until_terminal: bool = False) -> StreamingResponse:
        run_or_404(run_id)
        start = int(last_event_id) if last_event_id and last_event_id.isdigit() else after

        async def gen() -> AsyncIterator[str]:
            with eng.bus.subscribe(run_id) as sub:  # subscribe before replay so nothing is missed
                sent = start
                for ev in eng.store.get_events(run_id, start, limit=100000):
                    sent = ev["seq"]
                    yield f"id: {ev['seq']}\ndata: {json.dumps({'seq': ev['seq'], 'ts': ev['ts'], 'type': ev['type'], 'data': ev['data']})}\n\n"
                idle = 0
                while True:
                    if await request.is_disconnected():
                        return
                    ev = await sub.get(timeout=0.25 if until_terminal else 15)
                    if ev is None:
                        idle += 1
                        row = eng.store.get_run(run_id)
                        rt = eng.runtimes.get(run_id)
                        if until_terminal and row and RunStatus(row["status"]) in TERMINAL_RUN_STATES and not (rt and rt.active) and idle >= 2:
                            yield "event: end\ndata: {}\n\n"
                            return
                        if not until_terminal:
                            yield ": keepalive\n\n"
                        continue
                    idle = 0
                    if ev.seq <= sent:
                        continue
                    sent = ev.seq
                    yield f"id: {ev.seq}\ndata: {json.dumps({'seq': ev.seq, 'ts': ev.ts, 'type': ev.type, 'data': ev.data}, default=str)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ----------------------------------------------------------------- mutations
    @app.post("/api/incidents", status_code=202, dependencies=secured)
    async def create_incident(body: IncidentIn) -> dict[str, Any]:
        if body.alert is None and body.scenario is None:
            raise HTTPException(422, "provide `scenario` or `alert`")
        poisons = [Poison(p.source, p.text, p.level, p.doc_id) for p in body.poisons]
        env = None
        if body.scenario:
            if body.scenario not in SCENARIO_IDS:
                raise HTTPException(422, f"unknown scenario; choose from {SCENARIO_IDS}")
            alert = body.alert or build_scenario(body.scenario).alert()
            env = SimulatedEnvironment.for_scenario(body.scenario, poisons)
        else:
            assert body.alert is not None
            alert = body.alert
        alert = alert.model_copy(update={"id": f"{alert.id}-{len(eng.store.list_runs(100000)) + 1}"})
        run_id = eng.submit(alert, budget=body.budget, breakpoints=body.breakpoints, env=env)
        await eng.queue.enqueue(run_id)
        return {"run_id": run_id, "status": "queued"}

    @app.post("/api/approvals/{approval_id:path}/approve", dependencies=secured)
    async def approve(approval_id: str, body: Decision) -> dict[str, Any]:
        return eng.hitl.approve(approval_id, body.actor, body.note, body.edits)

    @app.post("/api/approvals/{approval_id:path}/reject", dependencies=secured)
    async def reject(approval_id: str, body: Decision) -> dict[str, Any]:
        return eng.hitl.reject(approval_id, body.actor, body.note)

    @app.post("/api/runs/{run_id}/pause", dependencies=secured)
    async def pause(run_id: str, body: Actor = Actor()) -> dict[str, str]:
        run_or_404(run_id)
        eng.hitl.pause(run_id, body.actor, body.reason)
        return {"status": "paused"}

    @app.post("/api/runs/{run_id}/resume", dependencies=secured)
    async def resume(run_id: str, body: Actor = Actor()) -> dict[str, str]:
        run_or_404(run_id)
        eng.hitl.resume(run_id, body.actor)
        return {"status": "resumed"}

    @app.post("/api/runs/{run_id}/cancel", dependencies=secured)
    async def cancel(run_id: str, body: Actor = Actor()) -> dict[str, str]:
        run_or_404(run_id)
        eng.hitl.cancel(run_id, body.actor)
        return {"status": "cancelling"}

    @app.post("/api/runs/{run_id}/tasks/{task_id}/retry", dependencies=secured)
    async def retry(run_id: str, task_id: str, body: RetryIn = RetryIn()) -> dict[str, Any]:
        run_or_404(run_id)
        return {"reset": eng.hitl.retry_task(run_id, task_id, body.actor, body.reset_downstream)}

    @app.post("/api/runs/{run_id}/tasks/{task_id}/skip", dependencies=secured)
    async def skip(run_id: str, task_id: str, body: SkipIn = SkipIn()) -> dict[str, str]:
        run_or_404(run_id)
        eng.hitl.skip_task(run_id, task_id, body.actor, body.output, body.reason)
        return {"status": "skipped"}

    @app.post("/api/runs/{run_id}/tasks/{task_id}/release", dependencies=secured)
    async def release(run_id: str, task_id: str, body: Actor = Actor()) -> dict[str, str]:
        run_or_404(run_id)
        eng.hitl.release_task(run_id, task_id, body.actor)
        return {"status": "released"}

    @app.patch("/api/runs/{run_id}/tasks/{task_id}/input", dependencies=secured)
    async def edit_input(run_id: str, task_id: str, body: InputIn) -> dict[str, Any]:
        run_or_404(run_id)
        return eng.hitl.edit_task_input(run_id, task_id, body.actor, body.params)

    @app.post("/api/runs/{run_id}/plan/edit", dependencies=secured)
    async def edit_plan(run_id: str, body: PlanEditIn) -> dict[str, Any]:
        run_or_404(run_id)
        return eng.hitl.edit_plan(run_id, body.actor, body.ops).model_dump(mode="json")

    @app.post("/api/runs/{run_id}/guidance", dependencies=secured)
    async def guidance(run_id: str, body: GuidanceIn) -> dict[str, Any]:
        run_or_404(run_id)
        return eng.hitl.inject_guidance(run_id, body.actor, body.text, body.task_id)

    return app


__all__ = ["create_app", "Severity", "asyncio"]
