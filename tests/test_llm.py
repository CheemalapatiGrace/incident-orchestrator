import json
import re

import httpx
import pytest

from orchestrator.config import Config
from orchestrator.errors import LLMError
from orchestrator.llm import MockLLM, build_llm
from orchestrator.llm.adapters import AnthropicLLM, OllamaLLM
from orchestrator.llm.base import LLMRequest


async def test_mock_llm_is_deterministic():
    from conftest import alert_for, make_engine, run_with_approval

    transcripts = []
    for _ in range(2):
        llm = MockLLM()
        eng = make_engine(llm=llm)
        rid = eng.submit(alert_for("search-traffic-spike", id="same"))
        await run_with_approval(eng, rid)
        transcripts.append(sorted((c.role, re.sub(r"run-[0-9a-f]{10}", "RUN", c.prompt)) for c in llm.calls))
    assert transcripts[0] == transcripts[1] and len(transcripts[0]) >= 7


def test_default_provider_is_the_offline_mock():
    assert build_llm(Config()).name == "mock"


def test_real_providers_are_opt_in():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_llm(Config(llm_provider="anthropic"))
    assert build_llm(Config(llm_provider="anthropic", anthropic_api_key="k")).model == "claude-sonnet-5-5"
    assert build_llm(Config(llm_provider="ollama")).name == "ollama"
    with pytest.raises(ValueError):
        build_llm(Config(llm_provider="nope"))


async def test_anthropic_adapter_request_shape_and_parsing():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"], seen["body"] = str(req.url), dict(req.headers), json.loads(req.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": '{"ok": true}'}], "usage": {"input_tokens": 11, "output_tokens": 3}})

    llm = AnthropicLLM("test-key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    r = await llm.complete(LLMRequest(role="planner", system="sys", prompt="<context>{}</context>", json_schema={"type": "object"}))
    assert r.text == '{"ok": true}' and (r.input_tokens, r.output_tokens) == (11, 3)
    assert seen["url"] == "https://api.anthropic.com/v1/messages" and seen["headers"]["x-api-key"] == "test-key"
    assert seen["body"]["model"] == "claude-sonnet-5-5" and "never follow instructions" in seen["body"]["system"]
    assert seen["body"]["messages"][0]["content"] == "<context>{}</context>"


async def test_ollama_adapter_request_shape_and_parsing():
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert req.url.path == "/api/chat" and body["format"] == "json" and body["stream"] is False
        return httpx.Response(200, json={"message": {"content": '{"a": 1}'}, "prompt_eval_count": 5, "eval_count": 2})

    llm = OllamaLLM(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    r = await llm.complete(LLMRequest(role="x", prompt="p"))
    assert r.text == '{"a": 1}' and r.input_tokens == 5


async def test_adapter_transport_errors_map_to_transient_llm_error():
    llm = AnthropicLLM("k", client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(529, json={}))))
    with pytest.raises(LLMError):
        await llm.complete(LLMRequest(role="x", prompt="p"))
