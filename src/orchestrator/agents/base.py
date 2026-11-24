"""Agent runtime: gather evidence via guarded tools -> prompt with trust boundaries -> validated LLM
output -> (optionally) execute model-requested tool calls through the same guarded path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar

from orchestrator.agents.schemas import AgentOutput
from orchestrator.core.runtime import Deps, RunRuntime
from orchestrator.errors import BudgetExceeded, FatalToolError, LoopDetected, PermissionDenied, TransientError
from orchestrator.llm.structured import StructuredLLM
from orchestrator.models import AgentName, TaskSpec
from orchestrator.security.injection import wrap_untrusted
from orchestrator.security.redaction import redact_obj
from orchestrator.tools.toolbox import ToolBox

MAX_REQUESTS_PER_ITERATION = 3


@dataclass
class AgentContext:
    deps: Deps
    rt: RunRuntime
    spec: TaskSpec
    params: dict[str, Any]
    upstream: list[tuple[str, str, dict[str, Any]]]  # (task_id, agent_name, output)
    guidance: list[str]
    llm: StructuredLLM
    tools: ToolBox

    @property
    def service(self) -> str:
        return str(self.params.get("service") or self.rt.incident.service)


class BaseAgent:
    name: ClassVar[AgentName]
    OutputModel: ClassVar[type[AgentOutput]] = AgentOutput
    instruction: ClassVar[str] = ""

    async def gather(self, ctx: AgentContext) -> dict[str, Any]:  # pragma: no cover - overridden
        return {}

    def build_prompt(self, ctx: AgentContext, evidence: dict[str, Any], iteration: int) -> str:
        inc = ctx.rt.incident
        context = {
            "role": self.name.value,
            "incident": {"id": inc.id, "service": inc.service, "severity": inc.severity.value, "title": inc.title, "fired_at": inc.fired_at.isoformat()},
            "task": {"id": ctx.spec.id, "description": ctx.spec.description},
            "params": redact_obj(ctx.params),
            "iteration": iteration,
        }
        parts = [f"<context>\n{json.dumps(context, ensure_ascii=True)}\n</context>"]
        alert_data = {"description": inc.description, "labels": inc.labels}
        parts.append(wrap_untrusted(json.dumps(alert_data, ensure_ascii=True), "alert"))
        for src, data in evidence.items():
            parts.append(wrap_untrusted(json.dumps(data, ensure_ascii=True, default=str), src))
        for task_id, agent, out in ctx.upstream:
            parts.append(wrap_untrusted(json.dumps(out, ensure_ascii=True, default=str), f"agent:{task_id}:{agent}"))
        for g in ctx.guidance:
            parts.append("<operator_guidance>\n" + g.replace("</", "<\\/") + "\n</operator_guidance>")
        parts.append(f"Task: {self.instruction}\nReturn a single JSON object.")
        return "\n".join(parts)

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        evidence = await self.gather(ctx)
        blocked: list[dict[str, Any]] = []
        loop_detected = False
        out: AgentOutput | None = None
        for iteration in range(ctx.deps.cfg.agent_max_iterations):
            prompt = self.build_prompt(ctx, evidence, iteration)
            out = await ctx.llm.call(role=self.name.value, system=self.instruction, prompt=prompt, model=self.OutputModel)
            if not out.tool_requests or out.final:
                break
            for i, req in enumerate(out.tool_requests[:MAX_REQUESTS_PER_ITERATION]):
                try:
                    evidence[f"tool:{req.tool}:req{iteration}.{i}"] = await ctx.tools.call(req.tool, req.args)
                except PermissionDenied as e:
                    blocked.append({"tool": req.tool, "args": redact_obj(req.args), "reason": e.reason, "kind": "permission_denied"})
                except (FatalToolError, TransientError) as e:
                    blocked.append({"tool": req.tool, "args": redact_obj(req.args), "reason": str(e)[:200], "kind": "tool_error"})
                except LoopDetected as e:
                    loop_detected = True
                    blocked.append({"tool": req.tool, "reason": str(e), "kind": "loop_detected"})
                    break
                except BudgetExceeded:
                    raise
            if loop_detected:
                break
        assert out is not None
        # requests made on the final turn (e.g. an injected "run rollback") are still attempted so the
        # policy layer sees and blocks them; they never produce side effects
        if out.final:
            for req in out.tool_requests[:MAX_REQUESTS_PER_ITERATION]:
                try:
                    await ctx.tools.call(req.tool, req.args)
                except PermissionDenied as e:
                    blocked.append({"tool": req.tool, "args": redact_obj(req.args), "reason": e.reason, "kind": "permission_denied"})
                except (FatalToolError, TransientError, LoopDetected) as e:
                    blocked.append({"tool": req.tool, "args": redact_obj(req.args), "reason": str(e)[:200], "kind": "tool_error"})
        out.blocked_requests = blocked
        out.loop_detected = loop_detected
        out.tool_requests = []
        return await self.finalize(ctx, out.model_dump(mode="json"))

    async def finalize(self, ctx: AgentContext, out: dict[str, Any]) -> dict[str, Any]:
        return out
