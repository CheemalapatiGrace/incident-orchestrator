"""Secret and PII redaction applied to tool output, traces, events and persisted state."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

_SECRET_KEYS = re.compile(r"(pass(word|wd)?|secret|token(?!s)|api[_-]?key|authorization|credential|private[_-]?key)", re.I)


def _luhn(num: str) -> bool:
    digits = [int(c) for c in num if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# (label, compiled pattern). Order matters: specific token formats before generic key=value.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("api_key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    (
        "credential_kv",
        re.compile(r"(?i)\b([\w.-]*(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|token|credential)[\w.-]*|pw)(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;&\"']+)"),
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+?1[ -.]?)?\(?\d{3}\)?[ -.]\d{3}[ -.]\d{4}(?!\d)")),
    ("ipv4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
]
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@lru_cache(maxsize=16384)
def _redact_cached(text: str) -> str:
    return _redact_uncached(text)


def redact_text(text: str) -> str:
    """Replace secrets/PII with ``[REDACTED:<kind>]`` markers (memoised: the same log lines recur)."""
    if not text:
        return text
    return _redact_cached(text) if len(text) <= 20000 else _redact_uncached(text)


def _redact_uncached(text: str) -> str:

    def kv(m: re.Match[str]) -> str:
        return f"{m.group(1)}{m.group(2)}[REDACTED:secret]"

    for label, pat in _PATTERNS:
        if label == "credential_kv":
            text = pat.sub(kv, text)
        else:
            text = pat.sub(f"[REDACTED:{label}]", text)
    text = _CARD.sub(lambda m: "[REDACTED:card]" if _luhn(m.group(0)) else m.group(0), text)
    return text


def redact_obj(obj: Any) -> Any:
    """Recursively redact strings; values under secret-looking keys are masked entirely."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and _SECRET_KEYS.search(k) and isinstance(v, str | int | float):
                out[k] = "[REDACTED:secret]"
            else:
                out[k] = redact_obj(v)
        return out
    if isinstance(obj, list | tuple):
        return [redact_obj(v) for v in obj]
    return obj
