from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    role: str  # planner | log_analyst | metrics_analyst | ...
    system: str = ""
    prompt: str
    schema_name: str | None = None
    json_schema: dict[str, Any] | None = None
    max_tokens: int = 1500
    temperature: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float = 0.0


class LLM(Protocol):
    name: str

    async def complete(self, req: LLMRequest) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)
