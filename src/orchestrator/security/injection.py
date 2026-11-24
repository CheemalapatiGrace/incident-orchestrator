"""Prompt-injection defence for untrusted data (tool output, alert text, runbooks).

Two mechanisms, deliberately layered:
1. *Quarantine*: lines matching known injection / tool-directive patterns are removed before any
   LLM sees them, and a structured finding is emitted (surfaced on the dashboard + audit trail).
2. *Tagging*: whatever remains is wrapped in ``<untrusted_data>`` tags (with tag-breakout escaping)
   and the system prompt tells the model it is data, never instructions.

A pattern scanner is a heuristic and is *not* the security boundary. The hard boundaries are the
per-agent permission policy and the approval gate, which hold even if an injection slips through
(the eval suite measures exactly that with a deliberately gullible model and the scanner disabled).
"""

from __future__ import annotations

import base64
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})

_ACTIONS = r"(?:rollback(?:_deploy)?|restart(?:_service)?|scale(?:_service)?|revert(?:_config)?|expand(?:_volume)?|failover(?:_dependency)?|delete\w*|drop\w*|wipe|shutdown|terminate|disable\w*)"

RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "override_instructions",
        re.compile(r"\b(ignore|disregard|forget|override|bypass|neglect)\W+(?:\w+\W+){0,3}?(previous|prior|above|earlier|system|safety|all)\W+(?:\w+\W+){0,2}?(instructions?|prompts?|rules?|guardrails?|polic(?:y|ies)|constraints?)"),
        "high",
    ),
    ("role_hijack", re.compile(r"\byou\s+are\s+now\b|\bact\s+as\s+(?:an?\s+)?(?:admin|root|system|developer)|\bnew\s+instructions?\s*:"), "high"),
    ("prompt_markers", re.compile(r"<\|?\s*(?:im_start|im_end|system|endoftext)\s*\|?>|\[/?(?:inst|sys)\]|^#{1,3}\s*(?:system|instruction)s?\b|system\s*prompt"), "high"),
    ("tool_directive", re.compile(rf"\b(?:run|execute|invoke|call|trigger)\b\W+(?:\w+\W+){{0,4}}?{_ACTIONS}\b"), "high"),
    ("tool_name_in_data", re.compile(r"\b(?:rollback_deploy|restart_service|scale_service|revert_config|expand_volume|failover_dependency)\b"), "medium"),
    ("approval_bypass", re.compile(r"\b(?:skip|bypass|disable)\W+(?:the\W+)?(?:approval|confirmation|review|human)|\bwithout\W+(?:waiting\W+for\W+)?(?:approval|human)|\bpre-?approved\b|\bauto-?approve"), "high"),
    ("loop_bait", re.compile(r"\b(?:re[- ]?run|repeat|retry|re[- ]?query|call)\b.{0,60}\b(?:until|forever|indefinitely|again and again|every second)|\bdo not (?:stop|finish|return|conclude)\b"), "medium"),
    ("exfiltration", re.compile(r"\b(?:send|post|upload|exfiltrate|forward|curl|wget)\b.{0,60}\b(?:https?://|ftp://)|curl\s+[^|]*\|\s*(?:ba)?sh|\b(?:reveal|print|output|show|dump)\b.{0,30}\b(?:secrets?|api[_ -]?keys?|passwords?|credentials?|tokens?|system prompt)"), "high"),
    ("authority_claim", re.compile(r"\b(?:message|note|instruction|order)s?\s+from\s+(?:the\s+)?(?:admin|administrator|system|security|developer|anthropic|operator)|\bthis\s+is\s+(?:the\s+)?(?:system|admin)\b|\bas\s+(?:the\s+)?(?:system|admin)\s+i\b"), "medium"),
    ("privilege_grant", re.compile(r"\b(?:grant|give|elevate|escalate)\b.{0,40}\b(?:permissions?|access|privileges?|admin|write|root)\b"), "high"),
]
_SQUASHED = re.compile(r"(?:ignore|disregard|forget|override|bypass)(?:all|the|your|any)?(?:previous|prior|above|earlier|system|safety)(?:instructions?|prompts?|rules?|guardrails?)")
_B64 = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/=])")


@dataclass
class Finding:
    rule: str
    severity: str
    source: str
    excerpt: str

    def as_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "severity": self.severity, "source": self.source, "excerpt": self.excerpt}


def normalize(text: str) -> str:
    """Canonicalise obfuscation (unicode forms, zero-width chars, leetspeak, spacing) for matching."""
    t = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).lower()
    t = t.translate(_LEET)
    t = re.sub(r"[\s_\-.]+", lambda m: "_" if "_" in m.group(0) else " ", t)
    t = re.sub(r"\b(?:\w )+\w\b", lambda m: m.group(0).replace(" ", ""), t)  # "i g n o r e" -> "ignore"
    return t


def _decoded_b64(line: str) -> list[str]:
    out = []
    for m in _B64.finditer(line):
        try:
            raw = base64.b64decode(m.group(0), validate=True).decode("utf-8")
        except Exception:
            continue
        if raw.isprintable() and len(raw) >= 12:
            out.append(raw)
    return out


class InjectionScanner:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def scan_line(self, line: str, source: str) -> list[Finding]:
        if not self.enabled or not line.strip():
            return []
        candidates = [line, *_decoded_b64(line)]
        findings: list[Finding] = []
        for cand in candidates:
            for variant in (cand.lower(), normalize(cand)):
                for rule, pat, sev in RULES:
                    if pat.search(variant) and not any(f.rule == rule for f in findings):
                        findings.append(Finding(rule, sev, source, line.strip()[:160]))
            if _SQUASHED.search(re.sub(r"[\s_\-.]+", "", normalize(cand))) and not any(f.rule == "override_instructions" for f in findings):
                findings.append(Finding("override_instructions", "high", source, line.strip()[:160]))
        return findings

    def sanitize_text(self, text: str, source: str) -> tuple[str, list[Finding]]:
        """Quarantine flagged lines; return (clean_text, findings)."""
        if not self.enabled:
            return text, []
        findings: list[Finding] = []
        clean: list[str] = []
        for line in text.split("\n"):
            f = self.scan_line(line, source)
            if f:
                findings.extend(f)
                clean.append(f"[QUARANTINED: suspected prompt injection ({', '.join(x.rule for x in f)})]")
            else:
                clean.append(line)
        return "\n".join(clean), findings

    def sanitize(self, obj: Any, source: str) -> tuple[Any, list[Finding]]:
        """Recursively sanitise every string leaf. Dict keys are not scanned."""
        findings: list[Finding] = []

        def walk(o: Any) -> Any:
            if isinstance(o, str):
                s, f = self.sanitize_text(o, source)
                findings.extend(f)
                return s
            if isinstance(o, dict):
                return {k: walk(v) for k, v in o.items()}
            if isinstance(o, list | tuple):
                return [walk(v) for v in o]
            return o

        return walk(obj), findings


def wrap_untrusted(payload: str, source: str) -> str:
    """Wrap data in explicit trust-boundary tags; neutralise attempts to close the tag early."""
    safe = payload.replace("</", "<\\/")
    return f'<untrusted_data source="{source}">\n{safe}\n</untrusted_data>'
