"""LLM abstraction: protocol, deterministic MockLLM (default), Anthropic and Ollama adapters."""

from orchestrator.llm.base import LLM, LLMRequest, LLMResponse
from orchestrator.llm.factory import build_llm
from orchestrator.llm.mock import MockLLM

__all__ = ["LLM", "LLMRequest", "LLMResponse", "MockLLM", "build_llm"]
