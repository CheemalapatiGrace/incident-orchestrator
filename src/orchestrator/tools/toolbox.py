"""ToolBox: the single choke point through which every tool call passes.

Order of checks: permission policy -> arg validation -> budget -> loop detection -> idempotency cache
-> circuit breaker/bulkhead/timeout/retry (+ fault injection) -> injection scan + redaction -> trace.
Agents only ever see the *sanitised* result.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import ValidationError

from orchestrator.core.budget import LoopDetector
from orchestrator.core.resilience import RetryPolicy, retry_async
from orchestrator.core.runtime import Deps, RunRuntime
from orchestrator.errors import (
    ApprovalError,
    FatalToolError,
    PermissionDenied,
    ToolExecutionError,
    ToolTimeout,
)
from orchestrator.security.policy import GATE_PRINCIPAL
from orchestrator.security.redaction import redact_obj
from orchestrator.tools.registry import Tool, args_hash, canonical, verify_approval


class ToolBox:
    def __init__(self, deps: Deps, rt: RunRuntime, agent: str, task_id: str | None) -> None:
        self.deps, self.rt, self.agent, self.task_id = deps, rt, agent, task_id
        self.loop = LoopDetector(deps.cfg.loop_max_repeats)

    # ----------------------------------------------------------------------- reads
    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        d, rt = self.deps, self.rt
        try:
            d.policy.check_tool(self.agent, tool_name)
        except PermissionDenied as e:
            d.bus.publish(rt.run_id, "guardrail.permission_denied", {"agent": self.agent, "tool": tool_name, "args": redact_obj(args), "reason": e.reason, "task_id": self.task_id})
            d.store.audit(rt.run_id, f"agent:{self.agent}", "permission_denied", tool_name, {"reason": e.reason})
            raise
        tool = d.registry.get(tool_name)
        if tool.mutating:  # defence in depth: policy already refuses, never rely on one layer
            raise PermissionDenied(self.agent, tool_name, "mutating tool")
        try:
            parsed = tool.args_model.model_validate(args)
        except ValidationError as e:
            raise FatalToolError(f"invalid arguments for {tool_name}: {e.errors()[0]['msg']}") from e
        clean_args = parsed.model_dump()
        rt.budget.charge_step()
        try:
            self.loop.record(f"{tool_name}:{canonical(clean_args)}")
        except Exception:
            d.bus.publish(rt.run_id, "guardrail.loop_detected", {"agent": self.agent, "tool": tool_name, "task_id": self.task_id})
            raise
        key = f"{rt.run_id}:{self.task_id}:{tool_name}:{args_hash(clean_args)}"
        async with d.tracer.span(rt.run_id, "tool", tool_name, input=clean_args, attrs={"agent": self.agent}) as sp:
            cached = d.store.idem_get(key)
            if cached is not None:
                sp.set(cache="hit")
                sp.set_output(cached["result"])
                return cached["result"]
            policy = RetryPolicy(d.cfg.max_tool_attempts, d.cfg.retry_base_delay_s, d.cfg.retry_max_delay_s)
            breaker = d.breakers.get(f"tool:{tool_name}")
            bulkhead = d.bulkheads.get(f"tool:{tool_name}")

            def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
                d.bus.publish(rt.run_id, "tool.retry", {"tool": tool_name, "attempt": attempt, "error": str(exc)[:200], "delay_s": round(delay, 4), "task_id": self.task_id})

            async def attempt(_n: int) -> Any:
                async with bulkhead:
                    return await breaker.call(lambda: self._once(tool, parsed))

            raw = await retry_async(attempt, policy, on_retry=on_retry, sleep=d.sleep)
            data, findings = d.scanner.sanitize(raw, f"tool:{tool_name}")
            data = redact_obj(data)
            if findings:
                self._report_findings(tool_name, findings)
                sp.set(injection_findings=len(findings))
            sp.set_output(data)
            d.store.idem_put(key, tool_name, data)
            return data

    async def _once(self, tool: Tool, parsed: Any) -> Any:
        d = self.deps
        rule = d.injector.check(f"tool:{tool.name}") if d.injector else None
        timeout = tool.timeout_s or d.cfg.tool_timeout_s
        try:
            if rule and rule.kind in {"error", "permanent_error"}:
                raise ToolExecutionError(tool.name, "injected failure")
            if rule and rule.kind == "timeout":
                await asyncio.wait_for(asyncio.sleep(rule.delay_s or 3600), timeout)
            if rule and rule.kind == "latency":
                await asyncio.sleep(rule.delay_s)
            return await asyncio.wait_for(tool.fn(self.rt.env, parsed), timeout)
        except TimeoutError as e:
            raise ToolTimeout(f"tool {tool.name!r} timed out after {timeout}s") from e

    def _report_findings(self, tool_name: str, findings: list) -> None:
        d, rt = self.deps, self.rt
        rt.taint = True
        d.store.update_run(rt.run_id, taint=1)
        for f in findings:
            d.bus.publish(rt.run_id, "guardrail.injection_detected", {**{k: redact_obj(v) for k, v in f.as_dict().items()}, "tool": tool_name, "agent": self.agent, "task_id": self.task_id})
        d.store.audit(rt.run_id, "system:scanner", "injection_quarantined", tool_name, {"count": len(findings), "rules": sorted({f.rule for f in findings})})

    # ------------------------------------------------------ approved mutations (gate only)
    async def execute_approved(self, approval_id: str, action: str, params: dict[str, Any]) -> dict[str, Any]:
        d, rt = self.deps, self.rt
        assert self.agent == GATE_PRINCIPAL
        d.policy.check_tool(GATE_PRINCIPAL, action)
        tool = d.registry.get(action)
        approval = d.store.get_approval(approval_id)
        verify_approval(approval, action, params)  # raises ApprovalError
        parsed = tool.args_model.model_validate(params)
        key = f"mut:{rt.run_id}:{action}:{args_hash(params)}"
        async with d.tracer.span(rt.run_id, "tool", f"exec:{action}", input=params, attrs={"agent": GATE_PRINCIPAL, "approval_id": approval_id, "mutating": True}) as sp:
            cached = d.store.idem_get(key)
            if cached is not None:
                sp.set(idempotent_replay=True)
                sp.set_output(cached["result"])
                return {**cached["result"], "idempotent_replay": True}
            # exactly one attempt: mutations are never blindly retried
            result = await asyncio.wait_for(tool.fn(rt.env, parsed, approval_id), d.cfg.tool_timeout_s)
            result = redact_obj(result)
            d.store.idem_put(key, action, result)
            d.store.audit(rt.run_id, f"gate:{approval['decided_by']}", "action_executed", action, {"approval_id": approval_id, "params": params})
            sp.set_output(result)
            return result


__all__ = ["ToolBox", "ApprovalError"]
