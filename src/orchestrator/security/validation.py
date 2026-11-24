"""Structured-output validation with deterministic repair (the LLM-assisted repair loop lives in
:mod:`orchestrator.llm.structured`)."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from orchestrator.errors import OutputValidationError

M = TypeVar("M", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(raw: str) -> str:
    """Pull the most plausible JSON object out of free-form model text."""
    m = _FENCE.search(raw)
    if m:
        raw = m.group(1)
    start = raw.find("{")
    if start == -1:
        raise OutputValidationError("no JSON object found in model output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    # truncated output: close open braces as a last-ditch attempt
    return raw[start:] + "}" * max(depth, 0)


def _cheap_repairs(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    s = re.sub(r"(?<![\w\"])'([^'\n]*)'(?=\s*[:,}\]])", r'"\1"', s)  # single-quoted strings
    return s


def parse_structured(raw: str, model: type[M]) -> M:
    """Parse + validate. Raises OutputValidationError (retryable) with a message fit for a repair prompt."""
    try:
        blob = extract_json(raw)
    except OutputValidationError:
        raise
    last: Exception | None = None
    for candidate in (blob, _cheap_repairs(blob)):
        try:
            return model.model_validate(json.loads(candidate))
        except json.JSONDecodeError as e:
            last = e
        except ValidationError as e:
            msgs = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:6])
            raise OutputValidationError(f"schema validation failed: {msgs}") from e
    raise OutputValidationError(f"invalid JSON: {last}")
