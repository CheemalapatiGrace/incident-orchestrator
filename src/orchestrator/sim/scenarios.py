"""Bundled, deterministic incident scenarios (seeded generators; no network, no files to download).

Each scenario carries ground truth (root cause + correct remediation) used by the eval suite.
All credentials/PII in the generated logs are fake and exist to exercise redaction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from orchestrator.models import Alert, Severity

T0 = datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC)  # alert fire time for every scenario


@dataclass
class LogLine:
    ts: datetime
    level: str
    service: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"ts": self.ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "level": self.level, "message": self.message}


@dataclass
class Scenario:
    id: str
    title: str
    service: str
    severity: Severity
    description: str
    root_cause: str
    correct_actions: list[dict[str, Any]]  # [{"action":..., "service":...}]
    logs: list[LogLine]
    metrics: dict[str, dict[str, Any]]  # metric -> {"unit","baseline","points":[(minutes_ago, value)]}
    deploys: list[dict[str, Any]]
    labels: dict[str, str] = field(default_factory=dict)

    def alert(self) -> Alert:
        return Alert(
            id=f"alert-{self.id}", service=self.service, title=self.title, severity=self.severity, fired_at=T0,
            description=self.description, labels={"scenario": self.id, "env": "prod", **self.labels},
        )


# --------------------------------------------------------------------------- service registry
SERVICES: dict[str, dict[str, Any]] = {
    "checkout-api": {"tier": "critical", "replicas": 12, "version": "v2.14.0", "previous_version": "v2.13.2", "owner": "team-checkout", "dependencies": ["payments-api", "inventory-db"]},
    "payments-worker": {"tier": "critical", "replicas": 6, "version": "v5.3.1", "previous_version": "v5.3.0", "owner": "team-payments", "dependencies": ["payments-db", "ledger-api"]},
    "search-api": {"tier": "standard", "replicas": 4, "version": "v8.2.0", "previous_version": "v8.1.4", "owner": "team-discovery", "dependencies": ["search-index"], "autoscaler_max": 4},
    "auth-service": {"tier": "critical", "replicas": 5, "version": "v4.9.1", "previous_version": "v4.9.0", "owner": "team-identity", "dependencies": ["user-db"]},
    "inventory-db": {"tier": "critical", "replicas": 3, "version": "pg-15.4", "previous_version": "pg-15.3", "owner": "team-data", "dependencies": [], "volume_gb": 500},
    "notification-service": {"tier": "standard", "replicas": 3, "version": "v1.22.0", "previous_version": "v1.21.5", "owner": "team-messaging", "dependencies": ["smtp-provider"], "dependency_alternates": {"smtp-provider": "smtp-provider-secondary"}},
    "reporting-api": {"tier": "standard", "replicas": 2, "version": "v3.1.0", "previous_version": "v3.0.9", "owner": "team-analytics", "dependencies": ["warehouse-db"]},
    # dependencies (no scenario data of their own)
    "payments-api": {"tier": "critical", "replicas": 8, "version": "v7.0.2", "previous_version": "v7.0.1", "owner": "team-payments", "dependencies": []},
    "payments-db": {"tier": "critical", "replicas": 3, "version": "pg-15.4", "previous_version": "pg-15.3", "owner": "team-data", "dependencies": [], "volume_gb": 800},
    "ledger-api": {"tier": "critical", "replicas": 4, "version": "v2.2.0", "previous_version": "v2.1.9", "owner": "team-payments", "dependencies": []},
    "search-index": {"tier": "standard", "replicas": 6, "version": "es-8.12", "previous_version": "es-8.11", "owner": "team-discovery", "dependencies": []},
    "user-db": {"tier": "critical", "replicas": 3, "version": "pg-15.4", "previous_version": "pg-15.3", "owner": "team-data", "dependencies": [], "volume_gb": 300},
    "smtp-provider": {"tier": "external", "replicas": 1, "version": "n/a", "previous_version": None, "owner": "vendor", "dependencies": []},
    "smtp-provider-secondary": {"tier": "external", "replicas": 1, "version": "n/a", "previous_version": None, "owner": "vendor", "dependencies": []},
    "warehouse-db": {"tier": "standard", "replicas": 2, "version": "pg-15.4", "previous_version": "pg-15.3", "owner": "team-data", "dependencies": [], "volume_gb": 2000},
}


def related_services(service: str) -> set[str]:
    """Direct dependencies plus direct dependents of ``service`` (the action-scope boundary)."""
    deps = set(SERVICES.get(service, {}).get("dependencies", []))
    dependents = {s for s, info in SERVICES.items() if service in info.get("dependencies", [])}
    return deps | dependents


# ------------------------------------------------------------------------------ generators
def _ts(minutes_ago: float) -> datetime:
    return T0 - timedelta(minutes=minutes_ago)


def _logs(rng: random.Random, svc: str, onset: float, info: list[str], errors: list[str], warns: list[str],
          n_info: int = 70, n_err: int = 42, n_warn: int = 14) -> list[LogLine]:
    def fill(t: str) -> str:
        return t.format(
            ms=rng.randint(40, 900), n=rng.randint(2, 99), pod=f"{svc}-{rng.choice(['7f9c', 'b21d', 'c4e8', '95aa'])}",
            req=f"req-{rng.randint(10**5, 10**6 - 1)}", user=f"user{rng.randint(100, 999)}@example.com",
            ip=f"10.{rng.randint(0, 9)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}", pct=rng.randint(91, 99),
        )

    out = [LogLine(_ts(rng.uniform(1, 120)), "INFO", svc, fill(rng.choice(info))) for _ in range(n_info)]
    out += [LogLine(_ts(rng.uniform(0.5, onset)), "ERROR", svc, fill(rng.choice(errors))) for _ in range(n_err)]
    out += [LogLine(_ts(rng.uniform(0.5, onset + 20)), "WARN", svc, fill(rng.choice(warns))) for _ in range(n_warn)]
    out.sort(key=lambda ln: ln.ts)
    return out


def _series(rng: random.Random, base: float, shape: str, peak: float | None = None, onset: float = 30, noise: float = 0.04) -> dict[str, Any]:
    pts: list[tuple[int, float]] = []
    n = 24
    for i in range(n):
        ago = (n - 1 - i) * 5
        if shape == "flat" or peak is None:
            v = base
        elif shape == "step":
            frac = 0.0 if ago > onset else min(1.0, (onset - ago) / 10 + 0.2)
            v = base + (peak - base) * frac
        elif shape == "ramp":
            v = base + (peak - base) * i / (n - 1)
        else:
            raise ValueError(shape)
        v *= 1 + rng.uniform(-noise, noise)
        pts.append((ago, round(max(v, 0.0), 2)))
    return {"baseline": base, "points": pts}


UNITS = {"error_rate_pct": "%", "latency_p99_ms": "ms", "cpu_pct": "%", "memory_pct": "%", "rps": "req/s", "disk_used_pct": "%", "upstream_error_pct": "%"}
BASE = {"error_rate_pct": 0.3, "latency_p99_ms": 180.0, "cpu_pct": 38.0, "memory_pct": 52.0, "rps": 800.0, "disk_used_pct": 61.0, "upstream_error_pct": 0.5}


def _metrics(rng: random.Random, overrides: dict[str, tuple[str, float, float]], base: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
    b = {**BASE, **(base or {})}
    out = {}
    for name, bv in b.items():
        shape, peak, onset = overrides.get(name, ("flat", None, 0))  # type: ignore[assignment]
        s = _series(rng, bv, shape, peak, onset, noise=0.02 if name in {"disk_used_pct", "memory_pct"} else 0.05)
        s["unit"] = UNITS[name]
        out[name] = s
    return out


def _deploy(id_: str, service: str, type_: str, mins_ago: float, version: str | None = None, prev: str | None = None,
            summary: str = "", author: str = "ci-bot", **extra: Any) -> dict[str, Any]:
    return {"id": id_, "service": service, "type": type_, "minutes_ago": mins_ago, "version": version, "previous_version": prev, "summary": summary, "author": author, **extra}


# ---------------------------------------------------------------------------------- scenarios
def _checkout(rng: random.Random) -> Scenario:
    info = [
        "GET /cart/{n} 200 {ms}ms", "POST /checkout 200 {ms}ms user={user}", "health check ok pod={pod}",
        "cache hit ratio 0.93 pod={pod}", "payment authorised req={req} ip={ip}",
        "loaded config db_password=hunter2-prod-pw api_key=sk-live_9f8e7d6c5b4a3210abcd region=eu-west-1",
        "request Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r",
    ]
    errors = [
        "java.lang.NullPointerException at CartTotalsCalculator.apply(CartTotals.java:88) req={req}",
        "HTTP 500 POST /checkout NullPointerException in CartTotalsCalculator req={req} user={user}",
        "Unhandled exception processing cart {n}: NullPointerException (discount line item missing)",
        "HTTP 500 GET /cart/{n} {ms}ms pod={pod}",
    ]
    warns = ["elevated 5xx ratio on pod={pod}", "slow response {ms}ms for /checkout"]
    return Scenario(
        "checkout-bad-deploy", "checkout-api 5xx error rate above 15%", "checkout-api", Severity.SEV1,
        "Error budget burn: 5xx ratio on checkout-api exceeded 15% for 10 minutes.", "bad_deploy",
        [{"action": "rollback_deploy", "service": "checkout-api"}],
        _logs(rng, "checkout-api", 28, info, errors, warns),
        _metrics(rng, {"error_rate_pct": ("step", 18.0, 28), "latency_p99_ms": ("step", 1400.0, 28)}, {"rps": 1200.0}),
        [
            _deploy("dep-9812", "checkout-api", "code", 32, "v2.14.0", "v2.13.2", "Refactor cart totals calculation (PR #4821)", "a.novak"),
            _deploy("dep-9805", "email-templates", "docs", 240, summary="Update marketing copy"),
            _deploy("dep-9790", "checkout-api", "code", 3 * 24 * 60, "v2.13.2", "v2.13.1", "Fix rounding bug"),
        ],
    )


def _payments(rng: random.Random) -> Scenario:
    info = ["settled batch {n} in {ms}ms", "health check ok pod={pod}", "consumer lag 0 pod={pod}", "ledger sync ok req={req}", "user={user} payout scheduled"]
    errors = [
        "java.lang.OutOfMemoryError: Java heap space pod={pod}",
        "container pod={pod} OOMKilled exit code 137",
        "GC overhead limit exceeded after {ms}ms pod={pod}",
        "heap usage {pct}% pod={pod} - allocation failure",
        "settlement batch {n} aborted: worker restarted req={req}",
    ]
    warns = ["heap usage {pct}% pod={pod}", "GC pause {ms}ms pod={pod}", "restart count increased on pod={pod}"]
    return Scenario(
        "payments-memory-leak", "payments-worker restart loop / OOMKilled", "payments-worker", Severity.SEV2,
        "payments-worker pods restarting repeatedly; settlement backlog growing.", "memory_leak",
        [{"action": "restart_service", "service": "payments-worker"}],
        _logs(rng, "payments-worker", 240, info, errors, warns, n_err=48),
        _metrics(rng, {"memory_pct": ("ramp", 98.0, 0), "cpu_pct": ("ramp", 71.0, 0), "error_rate_pct": ("ramp", 2.2, 0), "latency_p99_ms": ("ramp", 620.0, 0)}, {"memory_pct": 46.0}),
        [
            _deploy("dep-7700", "payments-worker", "code", 3 * 24 * 60, "v5.3.1", "v5.3.0", "Batch settlement optimisation", "r.chen"),
            _deploy("cfg-2210", "payments-worker", "config", 300, summary="Raise batch.size 100 -> 200", config_key="batch.size", previous_revision="cfg-2209"),
        ],
    )


def _search(rng: random.Random) -> Scenario:
    info = ["GET /search?q=shoes 200 {ms}ms", "index shard healthy pod={pod}", "health check ok pod={pod}", "cache miss for query hash {n}"]
    errors = [
        "request queue is full, rejecting request req={req}",
        "HTTP 429 too many requests from {ip}",
        "rate limit exceeded for tenant {n}",
        "HPA at maxReplicas=4, cannot scale further",
        "worker pool saturated: {n} requests waiting pod={pod}",
    ]
    warns = ["cpu throttling detected pod={pod}", "queue depth {n} pod={pod}"]
    return Scenario(
        "search-traffic-spike", "search-api latency and 429s", "search-api", Severity.SEV2,
        "search-api p99 latency above 2s with elevated 429 responses.", "traffic_spike",
        [{"action": "scale_service", "service": "search-api"}],
        _logs(rng, "search-api", 35, info, errors, warns),
        _metrics(rng, {"rps": ("step", 4800.0, 35), "cpu_pct": ("step", 97.0, 35), "latency_p99_ms": ("step", 2600.0, 35), "error_rate_pct": ("step", 9.0, 35)}, {"rps": 900.0, "latency_p99_ms": 120.0}),
        [_deploy("dep-5521", "search-api", "code", 400, "v8.2.0", "v8.1.4", "Tune ranking weights", "m.silva")],
    )


def _auth(rng: random.Random) -> Scenario:
    info = ["token issued sub=user{n} ttl=3600", "health check ok pod={pod}", "jwks refreshed in {ms}ms", "login ok user={user}"]
    errors = [
        "token validation failed: invalid issuer 'https://auth.staging.example.com' req={req}",
        "HTTP 401 POST /oauth/introspect invalid issuer user={user}",
        "invalid issuer claim, expected https://auth.example.com pod={pod}",
        "config reload applied jwt.issuer=https://auth.staging.example.com",
    ]
    warns = ["spike in 401 responses pod={pod}", "downstream services reporting auth failures"]
    return Scenario(
        "auth-config-regression", "auth-service 401 spike", "auth-service", Severity.SEV1,
        "Login and token introspection failing with 401 for most users.", "config_error",
        [{"action": "revert_config", "service": "auth-service"}],
        _logs(rng, "auth-service", 18, info, errors, warns),
        _metrics(rng, {"error_rate_pct": ("step", 31.0, 18)}, {"rps": 2000.0}),
        [
            _deploy("cfg-7781", "auth-service", "config", 18, summary="Update jwt.issuer for staging cutover (CR-1182)", author="j.okafor", config_key="jwt.issuer", previous_revision="cfg-7780"),
            _deploy("dep-3301", "auth-service", "code", 2 * 24 * 60, "v4.9.1", "v4.9.0", "Rotate signing keys", "s.park"),
        ],
    )


def _inventory(rng: random.Random) -> Scenario:
    info = ["checkpoint complete: {n} buffers written", "autovacuum finished on table orders", "connection from {ip} authorised", "replication lag {ms}ms"]
    errors = [
        "ERROR: could not extend file base/16384/2619: No space left on device",
        "PANIC: could not write to file pg_wal/xlogtemp.{n}: No space left on device",
        "write failed req={req}: disk full, ENOSPC",
        "checkpoint request failed: No space left on device",
    ]
    warns = ["data volume usage {pct}%", "WAL archive falling behind by {n} segments"]
    return Scenario(
        "inventory-disk-full", "inventory-db writes failing", "inventory-db", Severity.SEV1,
        "inventory-db rejecting writes; orders cannot reserve stock.", "disk_full",
        [{"action": "expand_volume", "service": "inventory-db"}],
        _logs(rng, "inventory-db", 40, info, errors, warns),
        _metrics(rng, {"disk_used_pct": ("ramp", 100.0, 0), "error_rate_pct": ("step", 12.0, 40), "latency_p99_ms": ("step", 900.0, 40)}, {"disk_used_pct": 84.0}),
        [_deploy("job-114", "inventory-db", "infra", 400, summary="nightly maintenance job", author="cron")],
    )


def _notifications(rng: random.Random) -> Scenario:
    info = ["email queued id={n} to user={user}", "template rendered in {ms}ms", "health check ok pod={pod}"]
    errors = [
        "upstream smtp-provider timeout after 5000ms req={req}",
        "circuit breaker open for smtp-provider, rejecting sends",
        "connection reset by smtp-provider.example.net pod={pod}",
        "send failed: upstream smtp-provider unavailable (503) req={req}",
    ]
    warns = ["retry queue depth {n}", "smtp-provider latency {ms}ms"]
    return Scenario(
        "notifications-dependency-outage", "notification-service email delivery failing", "notification-service", Severity.SEV2,
        "Transactional email delivery failing; retry queue growing.", "dependency_outage",
        [{"action": "failover_dependency", "service": "notification-service"}],
        _logs(rng, "notification-service", 30, info, errors, warns),
        _metrics(rng, {"upstream_error_pct": ("step", 96.0, 30), "error_rate_pct": ("step", 22.0, 30), "latency_p99_ms": ("step", 5200.0, 30)}, {"cpu_pct": 30.0}),
        [_deploy("dep-6100", "notification-service", "code", 1500, "v1.22.0", "v1.21.5", "Add digest emails", "l.ito")],
    )


def _false_alarm(rng: random.Random) -> Scenario:
    info = ["report {n} generated in {ms}ms", "health check ok pod={pod}", "scheduled export complete"]
    return Scenario(
        "reporting-false-alarm", "reporting-api latency blip (auto-resolved)", "reporting-api", Severity.SEV3,
        "Single latency sample above threshold; likely a flapping check.", "unknown", [],
        _logs(rng, "reporting-api", 5, info, ["transient read timeout req={req}"], ["slow query {ms}ms"], n_err=2, n_warn=3),
        _metrics(rng, {}, {"rps": 120.0}),
        [_deploy("dep-8800", "reporting-api", "docs", 600, summary="Update report templates")],
    )


_BUILDERS = {
    "checkout-bad-deploy": _checkout,
    "payments-memory-leak": _payments,
    "search-traffic-spike": _search,
    "auth-config-regression": _auth,
    "inventory-disk-full": _inventory,
    "notifications-dependency-outage": _notifications,
    "reporting-false-alarm": _false_alarm,
}
SCENARIO_IDS = list(_BUILDERS)


def build_scenario(scenario_id: str) -> Scenario:
    """Deterministically (re)build a scenario; each call returns a fresh, independent copy."""
    if scenario_id not in _BUILDERS:
        raise KeyError(f"unknown scenario {scenario_id!r}; choose from {SCENARIO_IDS}")
    return _BUILDERS[scenario_id](random.Random(f"io-{scenario_id}"))
