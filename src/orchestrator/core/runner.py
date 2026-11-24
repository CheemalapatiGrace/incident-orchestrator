"""The graph driver: runs the compiled LangGraph for one run until it is terminal.

The driver is the single writer of a run's graph. It

* streams the graph (``stream_mode=['updates', 'custom']``): ``custom`` chunks are the events published
  inside nodes and are forwarded to the event bus in superstep order; ``updates`` chunks keep the live
  view (and the ``tasks`` read-model table) in sync;
* when the graph parks, services the human side: an ``interrupt()`` at the approval gate is resumed with
  ``Command(resume=...)`` once the approval row is decided (also on timeout / cancellation, fail-safe
  reject); a stop at the ``hold`` breakpoint (paused run, held node) is resumed after queued human edits
  were folded in with ``graph.update_state(..., as_node='hold')``;
* re-opens a finished graph (retry / skip on a completed run) the same way: ``as_node='hold'`` makes the
  checkpoint look as if the breakpoint just passed, so the next superstep is ``dispatch``.

Nothing about *what* to run next lives here - that is the graph's job (``core/graph.py``).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from orchestrator.core.checkpoint import InlineSqliteSaver
from orchestrator.core.graph import abort_patch, build_graph
from orchestrator.core.runtime import RunRuntime
from orchestrator.core.state import fold_patches
from orchestrator.models import RunStatus

if TYPE_CHECKING:
    from orchestrator.engine import Orchestrator

CONTINUE: Any = object()  # "stream the graph again from its checkpoint"
# a backstop on top of the budgets: no healthy run needs more supersteps than this
RECURSION_LIMIT = 400


class GraphRunner:
    def __init__(self, eng: Orchestrator) -> None:
        self.e = eng
        self._graph: CompiledStateGraph | None = None

    # ------------------------------------------------------------------- graph / checkpointer
    async def graph(self) -> CompiledStateGraph:
        if self._graph is None:
            saver = InlineSqliteSaver(self.e.store)
            saver.setup()
            self._graph = build_graph(self.e, saver)
        return self._graph

    @staticmethod
    def config(rt: RunRuntime, cfg_parallel: int) -> dict[str, Any]:
        return {"configurable": {"thread_id": rt.run_id}, "recursion_limit": RECURSION_LIMIT, "max_concurrency": cfg_parallel}

    # ----------------------------------------------------------------------------- the driver
    def initial_state(self, rt: RunRuntime) -> dict[str, Any]:
        return {
            "incident": rt.incident.model_dump(mode="json"), "plan": rt.plan.model_dump(mode="json") if rt.plan else None,
            "tasks": {tid: {k: v for k, v in ts.row().items() if k != "task_id"} for tid, ts in rt.tasks.items()}, "guidance": list(rt.guidance),
            "paused": rt.paused, "cancelled": False, "taint": rt.taint, "abort": {}, "status": RunStatus.RUNNING.value, "cmd_ids": [],
        }

    async def drive(self, rt: RunRuntime) -> None:
        e = self.e
        graph = await self.graph()
        config = self.config(rt, e.cfg.max_parallel_tasks)
        snap = await graph.aget_state(config)
        go = True
        inp: Any = None
        if snap.created_at is None:  # first execution of this run
            inp = self.initial_state(rt)
        else:  # a checkpoint exists: this process resumes a run that another (or a dead) driver was running
            rt.resumed = True
            self._rehydrate(rt, snap.values)
            go = bool(snap.next) and not self._parked(snap)
        while True:
            if go:
                rt.budget.set_human_waiting(False)
                rt.set_status(RunStatus.PAUSED if rt.paused else RunStatus.RUNNING)
                try:
                    await self._stream(rt, graph, config, inp)
                except GraphRecursionError as exc:  # backstop: the graph looped far beyond any sane plan
                    inp = await self._abort_via_state(rt, graph, config, abort_patch(RunStatus.BUDGET_EXCEEDED, f"graph recursion limit: {exc}", "supersteps"))
                    continue
                snap = await graph.aget_state(config)
                inp, go = None, False
            if not snap.next:  # reached END (or already finished)
                cmds = e.store.pending_commands(rt.run_id)
                if not cmds:
                    return
                await self._apply_commands(rt, graph, config, cmds, as_node="hold")  # re-open a finished run: re-enter at dispatch
                go, inp = True, None
                snap = await graph.aget_state(config)
                continue
            inp = await self._park(rt, graph, config, snap)
            snap = await graph.aget_state(config)
            go = True

    @staticmethod
    def _parked(snap: Any) -> bool:
        return bool(snap.interrupts) or tuple(snap.next) == ("hold",)

    async def _stream(self, rt: RunRuntime, graph: CompiledStateGraph, config: dict[str, Any], inp: Any) -> None:
        bus = self.e.bus
        async for mode, chunk in graph.astream(inp, config, stream_mode=["updates", "custom"]):  # type: ignore[arg-type]
            if mode == "custom":
                bus.emit(chunk["run_id"], chunk["type"], chunk["data"], chunk.get("ts"))
                continue
            for node, upd in chunk.items():
                if node.startswith("__") or not upd:
                    continue
                rt.apply_patch(upd)
                if node == "dispatch" and upd.get("cmd_ids"):
                    self.e.store.mark_commands_applied(upd["cmd_ids"])
                if "status" in upd and node == "finalize":
                    rt.budget.set_human_waiting(False)
                    rt.set_status(RunStatus(upd["status"]))
        rt.persist_run()

    def _rehydrate(self, rt: RunRuntime, values: dict[str, Any]) -> None:
        """Rebuild the live view from the checkpointed state, then re-fold human edits that were queued but
        not yet applied when the previous driver stopped."""
        from orchestrator.core.runtime import TaskState

        if values.get("plan"):
            rt.apply_patch({"plan": values["plan"]}, operator_origin=True)
        rt.tasks = {tid: TaskState.from_rec(tid, rec) for tid, rec in (values.get("tasks") or {}).items()}
        for tid in rt.tasks:
            rt.persist_task(tid)
        rt.guidance = list(values.get("guidance") or rt.guidance)
        rt.taint = rt.taint or bool(values.get("taint"))
        rt.cancelled = rt.cancelled or bool(values.get("cancelled"))
        rt.set_paused(bool(values.get("paused")))
        for c in self.e.store.pending_commands(rt.run_id):
            rt.apply_patch(c["patch"], operator_origin=True)
        rt.persist_run()

    # --------------------------------------------------------------------------- human side
    async def _apply_commands(self, rt: RunRuntime, graph: CompiledStateGraph, config: dict[str, Any], cmds: list[dict], *, as_node: str) -> None:
        patch = fold_patches([c["patch"] for c in cmds])
        patch["cmd_ids"] = [c["id"] for c in cmds]
        await graph.aupdate_state(config, patch, as_node=as_node)
        self.e.store.mark_commands_applied([c["id"] for c in cmds])

    async def _abort_via_state(self, rt: RunRuntime, graph: CompiledStateGraph, config: dict[str, Any], patch: dict[str, Any]) -> Any:
        await graph.aupdate_state(config, patch, as_node="hold")
        return None

    def _set_waiting_status(self, rt: RunRuntime, snap: Any) -> None:
        rt.budget.set_human_waiting(True)
        if rt.paused:
            rt.set_status(RunStatus.PAUSED)
        elif snap.interrupts:
            rt.set_status(RunStatus.AWAITING_APPROVAL)
        else:
            rt.set_status(RunStatus.RUNNING)

    async def _park(self, rt: RunRuntime, graph: CompiledStateGraph, config: dict[str, Any], snap: Any) -> Any:
        """The graph is stopped at an ``interrupt()`` or at the ``hold`` breakpoint. Wait for the human, then
        return what to feed the graph next (a ``Command(resume=...)`` or ``None`` after ``update_state``)."""
        e = self.e
        d = e.deps
        while True:
            rt.wake.clear()
            snap = await graph.aget_state(config)
            if not self._parked(snap):  # someone (another process, an operator script) already moved the graph on
                return None
            self._set_waiting_status(rt, snap)
            if rt.cancelled:
                for ap in d.store.list_approvals(rt.run_id, "pending"):
                    if d.store.decide_approval(ap["approval_id"], "rejected", "system:cancelled", "run cancelled"):
                        d.bus.emit(rt.run_id, "approval.decided", {"approval_id": ap["approval_id"], "status": "rejected", "actor": "system:cancelled"})
            resumes: dict[str, Any] = {}
            for intr in snap.interrupts:
                val = intr.value if isinstance(intr.value, dict) else {}
                aids = val.get("approval_ids", [])
                for aid in aids:
                    self._expire(rt, aid)
                if all((d.store.get_approval(a) or {"status": "rejected"})["status"] != "pending" for a in aids):
                    resumes[intr.id] = {"approval_ids": aids}
            if resumes:
                return Command(resume=resumes)
            if tuple(snap.next) == ("hold",):
                cmds = d.store.pending_commands(rt.run_id)
                if cmds:
                    await self._apply_commands(rt, graph, config, cmds, as_node="hold")
                    return None
            try:
                await asyncio.wait_for(rt.wake.wait(), d.cfg.approval_poll_s)
            except TimeoutError:
                pass

    def _expire(self, rt: RunRuntime, aid: str) -> None:
        """No human decision before the approval timeout means rejection (fail-safe)."""
        d = self.e.deps
        timeout = d.cfg.approval_timeout_s
        row = d.store.get_approval(aid)
        if timeout is None or row is None or row["status"] != "pending":
            return
        if time.time() >= row["created_at"] + timeout:
            if d.store.decide_approval(aid, "rejected", "system:timeout", "no human decision before the approval timeout (fail-safe reject)"):
                d.store.audit(rt.run_id, "system:gate", "approval_timeout", aid, {"timeout_s": timeout})
                d.bus.emit(rt.run_id, "approval.decided", {"approval_id": aid, "status": "rejected", "actor": "system:timeout"})
