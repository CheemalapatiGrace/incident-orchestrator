"""SimulatedEnvironment: the mock infrastructure the tools talk to (logs, metrics, deploys, runbooks,
and mutating infra actions). One instance per run so concurrent incidents are isolated."""

from __future__ import annotations

import asyncio
import copy
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from orchestrator.errors import FatalToolError
from orchestrator.sim.retrieval import RunbookIndex
from orchestrator.sim.scenarios import SERVICES, T0, LogLine, Scenario, build_scenario, related_services


@dataclass
class Poison:
    """Adversarial content planted in the simulated data to test the guardrails."""

    source: str  # logs | deploy_summary | runbook | service_note | metric_unit
    text: str
    level: str = "ERROR"  # for logs
    doc_id: str = "rb-bad-deploy"  # for runbook poisons


class SimulatedEnvironment:
    def __init__(self, scenario: Scenario, poisons: list[Poison] | None = None, latency_s: float = 0.0) -> None:
        self.scenario = scenario
        self.latency_s = latency_s
        self.now = T0
        self.services = copy.deepcopy(SERVICES)
        self.mutations: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.resolved = False
        poisons = poisons or []
        self._logs = list(scenario.logs)
        self._deploys = copy.deepcopy(scenario.deploys)
        self._metrics = copy.deepcopy(scenario.metrics)
        runbook_extra: dict[str, str] = {}
        for p in poisons:
            if p.source == "logs":
                self._logs.append(LogLine(T0 - timedelta(minutes=3), p.level, scenario.service, p.text))
                self._logs.sort(key=lambda ln: ln.ts)
            elif p.source == "deploy_summary" and self._deploys:
                min(self._deploys, key=lambda d: d["minutes_ago"])["summary"] += " " + p.text  # newest change
            elif p.source == "runbook":
                runbook_extra[p.doc_id] = p.text
            elif p.source == "service_note":
                self.services[scenario.service]["notes"] = p.text
            elif p.source == "metric_unit":
                next(iter(self._metrics.values()))["unit"] = p.text
        self.runbooks = RunbookIndex.load_bundled(runbook_extra)

    @classmethod
    def for_scenario(cls, scenario_id: str, poisons: list[Poison] | None = None, latency_s: float = 0.0) -> SimulatedEnvironment:
        return cls(build_scenario(scenario_id), poisons, latency_s)

    def related_services(self, service: str) -> set[str]:
        return related_services(service)

    async def _io(self, name: str, **args: Any) -> None:
        self.calls.append({"tool": name, "args": args, "ts": time.time()})
        if self.latency_s:
            await asyncio.sleep(self.latency_s)

    # ----------------------------------------------------------------- reads
    async def search_logs(self, service: str, query: str = "", level: str | None = None, since_minutes: int = 120, limit: int = 100) -> dict:
        await self._io("search_logs", service=service, query=query, level=level, since_minutes=since_minutes, limit=limit)
        try:
            rx = re.compile(query, re.I) if query else None
        except re.error as e:
            raise FatalToolError(f"invalid query regex: {e}") from e
        cutoff = self.now - timedelta(minutes=since_minutes)
        rows = [
            ln for ln in self._logs
            if ln.service == service and ln.ts >= cutoff and (not level or ln.level == level.upper()) and (not rx or rx.search(ln.message))
        ]
        total = len(rows)
        rows = rows[-limit:]
        return {"service": service, "total_matched": total, "returned": len(rows), "lines": [ln.as_dict() for ln in rows]}

    async def query_metrics(self, service: str, metric: str = "all", window_minutes: int = 120) -> dict:
        await self._io("query_metrics", service=service, metric=metric, window_minutes=window_minutes)
        if service != self.scenario.service:
            return {"service": service, "metrics": {}}
        out: dict[str, Any] = {}
        for name, m in self._metrics.items():
            if metric not in ("all", name):
                continue
            pts = [{"t": (self.now - timedelta(minutes=ago)).strftime("%Y-%m-%dT%H:%M:%SZ"), "v": v} for ago, v in m["points"] if ago <= window_minutes]
            out[name] = {"unit": m["unit"], "baseline": m["baseline"], "points": pts}
        return {"service": service, "metrics": out}

    async def list_deploys(self, service: str, since_minutes: int = 240) -> dict:
        await self._io("list_deploys", service=service, since_minutes=since_minutes)
        rows = [d for d in self._deploys if d["minutes_ago"] <= since_minutes and d["service"] in {service, *self.services.get(service, {}).get("dependencies", [])}]
        rows.sort(key=lambda d: d["minutes_ago"])
        return {"service": service, "deploys": rows}

    async def get_service_info(self, service: str) -> dict:
        await self._io("get_service_info", service=service)
        if service not in self.services:
            raise FatalToolError(f"unknown service {service!r}")
        return {"service": service, **self.services[service]}

    async def search_runbooks(self, query: str, top_k: int = 3) -> dict:
        await self._io("search_runbooks", query=query, top_k=top_k)
        return {"query": query, "results": [
            {"doc_id": d.doc_id, "title": d.title, "category": d.category, "score": round(s, 4), "steps": d.steps}
            for d, s in self.runbooks.search(query, top_k)
        ]}

    # ------------------------------------------------------- mutations (only via approval gate)
    def _record(self, action: str, params: dict, approval_id: str | None) -> dict:
        m = {"action": action, "params": dict(params), "approval_id": approval_id, "ts": time.time()}
        self.mutations.append(m)
        ok = any(c["action"] == action and c["service"] == params.get("service") for c in self.scenario.correct_actions)
        if ok:
            self.resolved = True
        return {"status": "ok", "action": action, "applied": dict(params), "mitigates_incident": ok}

    def _svc(self, params: dict) -> dict:
        s = self.services.get(params.get("service", ""))
        if s is None:
            raise FatalToolError(f"unknown service {params.get('service')!r}")
        return s

    async def rollback_deploy(self, approval_id: str | None = None, **p: Any) -> dict:
        await self._io("rollback_deploy", **p)
        s = self._svc(p)
        if p["to_version"] not in {s.get("version"), s.get("previous_version")}:
            raise FatalToolError(f"version {p['to_version']!r} is not a known release of {p['service']}")
        s["version"] = p["to_version"]
        return self._record("rollback_deploy", p, approval_id)

    async def restart_service(self, approval_id: str | None = None, **p: Any) -> dict:
        await self._io("restart_service", **p)
        self._svc(p)
        return self._record("restart_service", p, approval_id)

    async def scale_service(self, approval_id: str | None = None, **p: Any) -> dict:
        await self._io("scale_service", **p)
        self._svc(p)["replicas"] = int(p["replicas"])
        return self._record("scale_service", p, approval_id)

    async def revert_config(self, approval_id: str | None = None, **p: Any) -> dict:
        await self._io("revert_config", **p)
        self._svc(p)
        return self._record("revert_config", p, approval_id)

    async def expand_volume(self, approval_id: str | None = None, **p: Any) -> dict:
        await self._io("expand_volume", **p)
        self._svc(p)["volume_gb"] = int(p["size_gb"])
        return self._record("expand_volume", p, approval_id)

    async def failover_dependency(self, approval_id: str | None = None, **p: Any) -> dict:
        await self._io("failover_dependency", **p)
        alts = self._svc(p).get("dependency_alternates", {})
        if alts.get(p["dependency"]) != p["target"]:
            raise FatalToolError(f"{p['target']!r} is not a configured alternate for {p['dependency']!r}")
        return self._record("failover_dependency", p, approval_id)
