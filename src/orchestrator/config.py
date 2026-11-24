"""Runtime configuration. All knobs have safe offline defaults; env vars use the IO_ prefix."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from orchestrator.models import Budget


class Config(BaseModel):
    # persistence
    db_path: str = "incident_orchestrator.db"

    # LLM
    llm_provider: str = "mock"  # mock | anthropic | ollama
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    llm_timeout_s: float = 30.0
    llm_max_repairs: int = 2
    # illustrative price table used for cost *estimates* (USD per million tokens)
    cost_per_mtok_in: float = 3.0
    cost_per_mtok_out: float = 15.0

    # concurrency / bulkheads
    max_parallel_tasks: int = 8  # per run
    agent_concurrency: int = 6  # per agent type, across runs (bulkhead)
    tool_concurrency: int = 16  # per tool, across runs (bulkhead)

    # timeouts / retries
    task_timeout_s: float = 30.0
    tool_timeout_s: float = 5.0
    max_task_attempts: int = 3
    max_tool_attempts: int = 3
    retry_base_delay_s: float = 0.05
    retry_max_delay_s: float = 2.0
    breaker_failure_threshold: int = 5
    breaker_recovery_s: float = 10.0

    # agent loops
    agent_max_iterations: int = 4
    loop_max_repeats: int = 3

    # human in the loop
    approval_timeout_s: float | None = None  # None = wait forever; on timeout -> auto-reject
    approval_poll_s: float = 0.2

    # horizontal scaling
    lease_s: float = 30.0
    worker_concurrency: int = 4

    # guardrails (the permission policy and approval gate cannot be disabled in production;
    # the scanner/policy switches exist for the ablation study in the eval suite)
    scanner_enabled: bool = True
    policy_enabled: bool = True

    api_token: str | None = None
    default_budget: Budget = Field(default_factory=Budget)

    @classmethod
    def from_env(cls) -> Config:
        env = os.environ
        cfg: dict = {}

        def put(key: str, name: str, cast=str) -> None:
            v = env.get(f"IO_{name.upper()}")
            if v not in (None, ""):
                cfg[key] = cast(v)

        for key, cast in [
            ("db_path", str),
            ("llm_provider", str),
            ("anthropic_model", str),
            ("ollama_host", str),
            ("ollama_model", str),
            ("task_timeout_s", float),
            ("tool_timeout_s", float),
            ("max_task_attempts", int),
            ("agent_concurrency", int),
            ("worker_concurrency", int),
            ("approval_poll_s", float),
            ("api_token", str),
        ]:
            put(key, key, cast)
        if env.get("IO_APPROVAL_TIMEOUT_S"):
            cfg["approval_timeout_s"] = float(env["IO_APPROVAL_TIMEOUT_S"])
        if env.get("ANTHROPIC_API_KEY"):
            cfg["anthropic_api_key"] = env["ANTHROPIC_API_KEY"]
        budget = {}
        for f, cast in [("max_steps", int), ("max_tokens", int), ("max_cost_usd", float), ("max_wall_clock_s", float)]:
            v = env.get(f"IO_BUDGET_{f.upper()}")
            if v:
                budget[f] = cast(v)
        if budget:
            cfg["default_budget"] = Budget(**budget)
        return cls(**cfg)
