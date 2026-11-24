"""Action risk scoring and scope validation. Output drives the approval gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orchestrator.models import RiskLevel
from orchestrator.security.policy import MUTATING_ACTIONS

BASE_RISK = {
    "scale_service": 0.20,
    "expand_volume": 0.25,
    "restart_service": 0.35,
    "revert_config": 0.40,
    "rollback_deploy": 0.50,
    "failover_dependency": 0.55,
}
FORBIDDEN_PREFIXES = ("delete", "drop", "wipe", "terminate", "shutdown", "disable", "truncate", "destroy")

REQUIRED_PARAMS = {
    "rollback_deploy": {"service", "to_version"},
    "restart_service": {"service"},
    "scale_service": {"service", "replicas"},
    "revert_config": {"service", "key", "to_revision"},
    "expand_volume": {"service", "size_gb"},
    "failover_dependency": {"service", "dependency", "target"},
}


@dataclass
class RiskAssessment:
    level: RiskLevel
    score: float
    reasons: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""
    requires_note: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value, "score": round(self.score, 3), "reasons": self.reasons,
            "blocked": self.blocked, "block_reason": self.block_reason, "requires_note": self.requires_note,
        }


def _blocked(reason: str) -> RiskAssessment:
    return RiskAssessment(RiskLevel.HIGH, 1.0, [reason], blocked=True, block_reason=reason)


def assess(
    action: str,
    params: dict[str, Any],
    *,
    incident_service: str,
    service_info: dict[str, Any],
    related_services: set[str],
    severity: str = "sev2",
    confidence: float = 1.0,
    tainted: bool = False,
) -> RiskAssessment:
    """Score a proposed action. ``blocked`` actions are never shown for approval at all."""
    if any(action.lower().startswith(p) for p in FORBIDDEN_PREFIXES):
        return _blocked(f"forbidden action class {action!r}")
    if action not in MUTATING_ACTIONS:
        return _blocked(f"unknown action {action!r}")
    missing = REQUIRED_PARAMS[action] - set(params)
    if missing:
        return _blocked(f"missing parameters {sorted(missing)}")
    target = str(params.get("service", ""))
    if target != incident_service and target not in related_services:
        return _blocked(f"out of scope: target service {target!r} is not related to {incident_service!r}")

    score = BASE_RISK[action]
    reasons = [f"base risk for {action}: {score:.2f}"]
    if action == "scale_service":
        cur = int(service_info.get("replicas", 1) or 1)
        new = params.get("replicas")
        if not isinstance(new, int) or isinstance(new, bool) or not 1 <= new <= 50:
            return _blocked(f"replicas out of bounds: {new!r}")
        if new > cur * 10:
            return _blocked(f"scale factor {new / cur:.1f}x exceeds 10x safety limit")
        if new > cur * 4:
            score += 0.25
            reasons.append(f"large scale-up ({new / cur:.1f}x)")
        elif new < cur:
            score += 0.30
            reasons.append("scale-down during an incident")
    if action == "expand_volume":
        cur_gb = int(service_info.get("volume_gb", 0) or 0)
        new_gb = params.get("size_gb")
        if not isinstance(new_gb, int) or isinstance(new_gb, bool) or new_gb <= cur_gb or new_gb > max(cur_gb * 4, 100):
            return _blocked(f"invalid volume size {new_gb!r} (current {cur_gb} GB)")
    if service_info.get("tier") == "critical":
        score += 0.20
        reasons.append("target service is tier-critical")
    if target != incident_service:
        score += 0.10
        reasons.append("action targets a dependency, not the alerting service")
    if severity == "sev1":
        score += 0.05
        reasons.append("sev1 incident")
    if confidence < 0.6:
        score += 0.20
        reasons.append(f"low root-cause confidence ({confidence:.2f})")
    if tainted:
        score = max(score, 0.60) + 0.10
        reasons.append("evidence contained suspected prompt injection (tainted provenance)")

    level = RiskLevel.LOW if score < 0.40 else RiskLevel.MEDIUM if score < 0.65 else RiskLevel.HIGH
    return RiskAssessment(level, min(score, 1.0), reasons, requires_note=level is RiskLevel.HIGH)
