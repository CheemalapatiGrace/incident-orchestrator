from __future__ import annotations

from orchestrator.config import Config
from orchestrator.llm.base import LLM
from orchestrator.llm.mock import MockLLM


def build_llm(cfg: Config) -> LLM:
    """Pick the LLM backend. The default is the offline deterministic MockLLM."""
    provider = cfg.llm_provider.lower()
    if provider == "mock":
        return MockLLM()
    if provider == "anthropic":
        from orchestrator.llm.adapters import AnthropicLLM

        if not cfg.anthropic_api_key:
            raise ValueError("IO_LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        return AnthropicLLM(cfg.anthropic_api_key, cfg.anthropic_model, timeout=cfg.llm_timeout_s)
    if provider == "ollama":
        from orchestrator.llm.adapters import OllamaLLM

        return OllamaLLM(cfg.ollama_host, cfg.ollama_model, timeout=cfg.llm_timeout_s)
    raise ValueError(f"unknown llm provider {cfg.llm_provider!r}")
