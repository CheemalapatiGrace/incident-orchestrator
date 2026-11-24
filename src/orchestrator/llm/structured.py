"""Instrumented, schema-validated LLM calls with repair/retry.

Every call: charges the budget, records an ``llm`` span (input, output, tokens, cost estimate,
latency), honours fault injection, validates the output against a pydantic schema and - on failure -
re-asks the model with the validation error ("repair") up to ``llm_max_repairs`` times.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel

from orchestrator.config import Config
from orchestrator.core.budget import BudgetTracker
from orchestrator.core.events import EventBus
from orchestrator.core.tracing import Tracer
from orchestrator.errors import LLMError, OutputValidationError
from orchestrator.faults import FaultInjector
from orchestrator.llm.base import LLM, LLMRequest, LLMResponse, estimate_tokens
from orchestrator.security.validation import parse_structured

M = TypeVar("M", bound=BaseModel)


@lru_cache(maxsize=64)
def schema_for(model: type[BaseModel]) -> dict:
    return model.model_json_schema()

GARBAGE = '{"summary": "I will now', "Sure! Here is the analysis you asked for: <not json>"


class StructuredLLM:
    def __init__(
        self, *, llm: LLM, cfg: Config, tracer: Tracer, bus: EventBus, budget: BudgetTracker,
        injector: FaultInjector | None, run_id: str, agent: str,
    ) -> None:
        self.llm, self.cfg, self.tracer, self.bus, self.budget = llm, cfg, tracer, bus, budget
        self.injector, self.run_id, self.agent = injector, run_id, agent

    def cost(self, tin: int, tout: int) -> float:
        return tin / 1e6 * self.cfg.cost_per_mtok_in + tout / 1e6 * self.cfg.cost_per_mtok_out

    async def _invoke(self, req: LLMRequest, attempt: int) -> LLMResponse:
        self.budget.charge_step()
        async with self.tracer.span(
            self.run_id, "llm", f"{self.agent}.llm", input={"role": req.role, "prompt": req.prompt, "attempt": attempt}
        ) as sp:
            rule = self.injector.check(f"llm:{req.role}") if self.injector else None
            try:
                if rule and rule.kind in {"error", "permanent_error"}:
                    raise LLMError("injected LLM failure")
                if rule and rule.kind == "timeout":
                    await asyncio.wait_for(asyncio.sleep(rule.delay_s or 3600), self.cfg.llm_timeout_s)
                if rule and rule.kind == "latency":
                    await asyncio.sleep(rule.delay_s)
                if rule and rule.kind == "malformed":
                    text = GARBAGE[attempt % 2]
                    resp = LLMResponse(text=text, model="fault-injected", input_tokens=estimate_tokens(req.prompt), output_tokens=estimate_tokens(text))
                    sp.set(fault="malformed")
                else:
                    resp = await asyncio.wait_for(self.llm.complete(req), self.cfg.llm_timeout_s)
            except TimeoutError as e:
                raise LLMError("llm call timed out") from e
            except LLMError:
                raise
            except Exception as e:  # adapter bugs/transport errors are transient from our viewpoint
                raise LLMError(f"llm call failed: {type(e).__name__}: {e}") from e
            cost = self.cost(resp.input_tokens, resp.output_tokens)
            sp.set(model=resp.model, tokens_in=resp.input_tokens, tokens_out=resp.output_tokens, cost_usd=round(cost, 6), attempt=attempt)
            sp.set_output(resp.text)
            self.budget.charge_llm(resp.input_tokens, resp.output_tokens, cost)
            return resp

    async def call(self, *, role: str, system: str, prompt: str, model: type[M], max_tokens: int = 1500) -> M:
        schema = schema_for(model)
        last_err: OutputValidationError | None = None
        raw = ""
        for attempt in range(self.cfg.llm_max_repairs + 1):
            p = prompt
            if attempt:
                p += (
                    f"\n\n<validation_error>\nYour previous answer was rejected: {last_err}\n"
                    f"Previous answer (truncated): {raw[:300]}\nReturn ONLY a corrected JSON object.\n</validation_error>"
                )
            req = LLMRequest(role=role, system=system, prompt=p, schema_name=model.__name__, json_schema=schema, max_tokens=max_tokens, metadata={"attempt": attempt, "agent": self.agent})
            resp = await self._invoke(req, attempt)
            raw = resp.text
            try:
                out = parse_structured(raw, model)
            except OutputValidationError as e:
                last_err = e
                self.bus.publish(self.run_id, "guardrail.output_invalid", {"agent": self.agent, "role": role, "attempt": attempt, "error": str(e)[:300]})
                continue
            if attempt:
                self.bus.publish(self.run_id, "guardrail.output_repaired", {"agent": self.agent, "role": role, "attempts": attempt + 1})
            return out
        raise OutputValidationError(f"{role}: output still invalid after {self.cfg.llm_max_repairs} repairs: {last_err}")
