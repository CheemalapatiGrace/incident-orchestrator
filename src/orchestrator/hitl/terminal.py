"""Terminal approver: prompts a human in the shell for every pending approval of a run."""

from __future__ import annotations

import asyncio
import getpass
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from orchestrator.errors import ApprovalError
from orchestrator.models import TERMINAL_RUN_STATES, RunStatus

if TYPE_CHECKING:
    from orchestrator.engine import Orchestrator


def describe(row: dict) -> str:
    p, r = row["proposal"], row["risk"]
    lines = [
        "",
        "=" * 70,
        f" APPROVAL REQUIRED  {row['approval_id']}",
        f" action : {p['action']}  {json.dumps(p['params'])}",
        f" why    : {p.get('rationale', '')}",
        f" effect : {p.get('expected_effect', '')}",
        f" risk   : {r['level'].upper()} ({r['score']:.2f})" + ("  [justification note required]" if r.get("requires_note") else ""),
    ]
    lines += [f"          - {x}" for x in r.get("reasons", [])]
    lines.append("=" * 70)
    return "\n".join(lines)


class TerminalApprover:
    def __init__(self, engine: Orchestrator, input_fn: Callable[[str], str] = input, out: Callable[[str], None] = print, actor: str | None = None) -> None:
        self.e, self.input, self.out = engine, input_fn, out
        self.actor = actor or f"cli:{getpass.getuser()}"
        self.handled: set[str] = set()

    async def _ask(self, prompt: str) -> str:
        return (await asyncio.to_thread(self.input, prompt)).strip()

    async def handle(self, row: dict) -> None:
        self.out(describe(row))
        while True:
            choice = (await self._ask("[a]pprove / [r]eject / [e]dit params > ")).lower()
            try:
                if choice in {"r", "reject"}:
                    note = await self._ask("reason (optional) > ")
                    self.e.hitl.reject(row["approval_id"], self.actor, note)
                    return
                if choice in {"a", "approve", "e", "edit"}:
                    edits = None
                    if choice.startswith("e"):
                        edits = json.loads(await self._ask("param overrides as JSON > ") or "{}")
                    note = await self._ask("justification note > " if row["risk"].get("requires_note") else "note (optional) > ")
                    self.e.hitl.approve(row["approval_id"], self.actor, note, edits)
                    return
            except (ApprovalError, json.JSONDecodeError) as exc:
                self.out(f"  ! {exc}")
                continue
            self.out("  please answer a, r or e")

    async def watch(self, run_id: str) -> None:
        while True:
            for row in self.e.store.list_approvals(run_id, "pending"):
                if row["approval_id"] not in self.handled:
                    self.handled.add(row["approval_id"])
                    await self.handle(row)
            run = self.e.store.get_run(run_id)
            if run and RunStatus(run["status"]) in TERMINAL_RUN_STATES:
                return
            await asyncio.sleep(0.1)
