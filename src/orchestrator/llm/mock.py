"""Deterministic, rule-based MockLLM.

It is *not* a canned-answer table: each role runs real analysis rules over the evidence embedded in
the prompt (regex signatures over log lines, threshold/trend detection over metric series, recency
correlation over deploy records, weighted evidence fusion for root cause). That keeps the whole system
testable offline while exercising genuine data flow, prompt construction, validation and guardrails.

``gullible=True`` turns it into an adversarial test double: it *obeys* directives it finds inside
untrusted data (run a rollback, restart something, loop forever). The eval suite uses it to prove the
guardrails hold even when the model itself is fooled.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any

from orchestrator.llm.base import LLMRequest, LLMResponse, estimate_tokens

_CTX = re.compile(r"<context>\n?(.*?)\n?</context>", re.S)
_EV = re.compile(r'<untrusted_data source="([^"]+)">\n(.*?)\n</untrusted_data>', re.S)
_GUIDE = re.compile(r"<operator_guidance>\n?(.*?)\n?</operator_guidance>", re.S)

LOG_SIGNATURES: list[tuple[str, str, str, float]] = [
    ("oom", r"OutOfMemoryError|OOMKilled|exit code 137|GC overhead|heap (?:usage|exhausted|space)", "memory_leak", 0.90),
    ("queue_saturation", r"request queue (?:is )?full|rate limit exceeded|too many requests|HPA .*maxReplicas|autoscaler.*max", "traffic_spike", 0.80),
    ("config_rejected", r"invalid issuer|invalid signature key|unknown config key|missing required config|config (?:reload|validation) failed", "config_error", 0.90),
    ("disk_exhausted", r"No space left on device|disk (?:is )?full|could not extend file|ENOSPC", "disk_full", 0.95),
    ("upstream_failure", r"upstream .*(?:timeout|unavailable)|circuit breaker open|connection (?:refused|reset)", "dependency_outage", 0.85),
    ("unhandled_exception", r"NullPointerException|TypeError|unhandled exception|stack trace", "bad_deploy", 0.50),
]
AGENT_WEIGHT = {"log_analyst": 1.0, "metrics_analyst": 1.1, "deploy_history": 1.3, "runbook": 0.5}
CATEGORY_LABEL = {
    "bad_deploy": "a faulty code deployment",
    "memory_leak": "a memory leak causing OOM restarts",
    "traffic_spike": "a traffic surge exhausting capacity",
    "config_error": "a bad configuration change",
    "disk_full": "disk exhaustion on the data volume",
    "dependency_outage": "an upstream dependency outage",
    "unknown": "an undetermined cause (investigation ongoing)",
}

# gullible-mode directive parsing (multilingual on purpose; the regex scanner only knows English)
_DIRECTIVE = re.compile(
    r"(?:run|execute|invoke|call|trigger|ejecuta\w*|ejecutar|lanza\w*)\W+(?:\w+\W+){0,3}?"
    r"(rollback_deploy|rollback|restart_service|restart|scale_service|scale|revert_config|expand_volume|failover_dependency|delete_\w+|drop_\w+|wipe_\w+)"
    r"(?:\W+(?:on\W+|for\W+)?([a-z][a-z0-9]*(?:-[a-z0-9]+)+))?",
    re.I,
)
_LOOPBAIT = re.compile(r"(?:re[- ]?run|repeat|retry|call|repite\w*)\b.{0,60}\b(?:until|forever|indefinitely|again and again|hasta|siempre)", re.I)


def _parse_ts(s: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(s) if s else None
    except ValueError:
        return None


class MockLLM:
    name = "mock"

    def __init__(
        self,
        *,
        latency_s: float = 0.0,
        gullible: bool = False,
        overrides: dict[str, list[str] | Callable[[LLMRequest], str]] | None = None,
    ) -> None:
        self.latency_s = latency_s
        self.gullible = gullible
        self.overrides = overrides or {}
        self.calls: list[LLMRequest] = []
        self._override_idx: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ API
    async def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req)
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        text = self._override(req)
        if text is None:
            text = json.dumps(self._answer(req))
        return LLMResponse(
            text=text, model="mock-rules-v1", input_tokens=estimate_tokens(req.system + req.prompt),
            output_tokens=estimate_tokens(text), latency_s=self.latency_s,
        )

    def _override(self, req: LLMRequest) -> str | None:
        o = self.overrides.get(req.role)
        if o is None:
            return None
        if callable(o):
            return o(req)
        i = self._override_idx[req.role]
        self._override_idx[req.role] += 1
        if i < len(o):
            return o[i]
        return None  # script exhausted: fall back to the rule-based answer

    # ------------------------------------------------------------ prompt IO
    @staticmethod
    def _parse(prompt: str) -> tuple[dict, dict[str, Any], list[str], str]:
        m = _CTX.search(prompt)
        ctx = json.loads(m.group(1)) if m else {}
        ev: dict[str, Any] = {}
        raw_parts: list[str] = []
        for src, body in _EV.findall(prompt):
            raw_parts.append(body)
            try:
                ev[src] = json.loads(body)
            except json.JSONDecodeError:
                ev[src] = body
        guidance = [g.strip() for g in _GUIDE.findall(prompt)]
        return ctx, ev, guidance, "\n".join(raw_parts)

    def _answer(self, req: LLMRequest) -> dict:
        ctx, ev, guidance, raw = self._parse(req.prompt)
        handler = getattr(self, f"_h_{req.role}", None)
        if handler is None:
            raise ValueError(f"MockLLM has no handler for role {req.role!r}")
        out = handler(ctx, ev, guidance)
        if self.gullible and req.role != "planner":
            self._obey_injection(req.role, ctx, raw, out)
        return out

    # ------------------------------------------------------- adversarial mode
    def _obey_injection(self, role: str, ctx: dict, raw: str, out: dict) -> None:
        svc = ctx.get("incident", {}).get("service", "unknown")
        for m in _DIRECTIVE.finditer(raw):
            verb, target = m.group(1).lower(), (m.group(2) or svc)
            if verb.startswith("rollback"):
                tool, args = "rollback_deploy", {"service": target, "to_version": "v0.0.1"}
            elif verb.startswith("restart"):
                tool, args = "restart_service", {"service": target}
            elif verb.startswith("scale"):
                tool, args = "scale_service", {"service": target, "replicas": 100}
            elif verb.startswith("revert"):
                tool, args = "revert_config", {"service": target, "key": "*", "to_revision": "r0"}
            elif verb.startswith("expand"):
                tool, args = "expand_volume", {"service": target, "size_gb": 99999}
            elif verb.startswith("failover"):
                tool, args = "failover_dependency", {"service": target, "dependency": "db", "target": "attacker-db"}
            else:
                tool, args = verb, {"service": target}
            if role == "remediation":
                out.setdefault("proposed_actions", []).append(
                    {"action": tool, "params": args, "rationale": "instruction found in evidence", "expected_effect": ""}
                )
            else:
                out.setdefault("tool_requests", []).append({"tool": tool, "args": args})
                out["final"] = True
        if _LOOPBAIT.search(raw) and role != "remediation":
            out.setdefault("tool_requests", []).append(
                {"tool": "search_logs", "args": {"service": svc, "query": "error", "level": "ERROR"}}
            )
            out["final"] = False

    # -------------------------------------------------------------- planner
    def _h_planner(self, ctx: dict, ev: dict, guidance: list[str]) -> dict:
        inc = ctx.get("incident", {})
        svc = inc.get("service", "unknown")
        tasks: list[dict] = [
            {"id": "logs", "agent": "log_analyst", "description": "Find error signatures and onset in recent logs", "params": {"service": svc, "window_minutes": 120}},
            {"id": "metrics", "agent": "metrics_analyst", "description": "Detect metric anomalies vs baseline", "params": {"service": svc, "window_minutes": 120}},
            {"id": "deploys", "agent": "deploy_history", "description": "List recent deploys/config changes", "params": {"service": svc, "since_minutes": 240}},
            {"id": "runbook", "agent": "runbook", "description": "Retrieve matching runbooks", "depends_on": ["logs", "metrics"]},
            {"id": "remediation", "agent": "remediation", "description": "Fuse evidence into a root cause and propose fixes", "depends_on": ["logs", "metrics", "deploys", "runbook"], "optional": False},
            {"id": "comms", "agent": "comms", "description": "Draft stakeholder status update", "depends_on": ["remediation"], "params": {"kind": "update"}},
        ]
        if inc.get("severity") == "sev1":
            tasks.append({"id": "comms_ack", "agent": "comms", "description": "Send initial acknowledgement (parallel with triage)", "params": {"kind": "ack"}})
        return {"tasks": tasks, "rationale": f"{inc.get('severity', 'sev2')} incident on {svc}: fan out log/metric/deploy triage, then runbook, remediation, comms."}

    # ----------------------------------------------------------- log analyst
    def _h_log_analyst(self, ctx: dict, ev: dict, guidance: list[str]) -> dict:
        lines: list[dict] = []
        seen: set[tuple] = set()
        for src, v in ev.items():
            if src.startswith("tool:search_logs") and isinstance(v, dict):
                for ln in v.get("lines", []):
                    key = (ln.get("ts"), ln.get("message"))
                    if key not in seen:
                        seen.add(key)
                        lines.append(ln)
        bad = [ln for ln in lines if str(ln.get("level", "")).upper() in {"ERROR", "WARN", "FATAL", "CRITICAL"}]
        sigs, causes = [], []
        for sid, pat, cat, w in LOG_SIGNATURES:
            rx = re.compile(pat, re.I)
            hits = [ln for ln in bad if rx.search(str(ln.get("message", "")))]
            if not hits:
                continue
            first = min((ln["ts"] for ln in hits if ln.get("ts")), default=None)
            sigs.append({"pattern": sid, "category": cat, "count": len(hits), "first_seen": first, "sample": str(hits[0]["message"])[:200]})
            causes.append({"category": cat, "confidence": round(min(w, w * min(1.0, 0.5 + 0.1 * len(hits))), 3), "reason": f"{len(hits)} log lines match {sid}"})
        causes.sort(key=lambda c: -c["confidence"])
        onset = min((s["first_seen"] for s in sigs if s["first_seen"]), default=None)
        return {
            "summary": f"{len(bad)} error/warn lines analysed; {len(sigs)} distinct error signatures.",
            "error_line_count": len(bad), "onset": onset, "error_signatures": sigs, "suspected_causes": causes,
            "tool_requests": [], "final": True,
        }

    # -------------------------------------------------------- metrics analyst
    def _h_metrics_analyst(self, ctx: dict, ev: dict, guidance: list[str]) -> dict:
        metrics: dict[str, Any] = {}
        for src, v in ev.items():
            if src.startswith("tool:query_metrics") and isinstance(v, dict):
                metrics.update(v.get("metrics", {}))
        facts: dict[str, dict] = {}
        for name, m in metrics.items():
            pts = [p["v"] for p in m.get("points", []) if isinstance(p.get("v"), int | float)]
            if len(pts) < 4:
                continue
            base = float(m.get("baseline") or 0.0)
            last = sum(pts[-3:]) / 3
            diffs = [b - a for a, b in zip(pts, pts[1:], strict=False)]
            mono = sum(1 for d in diffs if d > 0) / max(1, len(diffs))
            thresh = base + 0.5 * (last - base)
            onset = next((p["t"] for p in m["points"] if p["v"] >= thresh), None) if last > base else None
            facts[name] = {"base": base, "last": last, "first": pts[0], "mono": mono, "onset": onset, "ratio": last / max(base, 0.01)}
        anomalies, causes = [], []

        def add(metric: str, kind: str) -> None:
            f = facts[metric]
            anomalies.append({"metric": metric, "kind": kind, "baseline": round(f["base"], 2), "last": round(f["last"], 2), "ratio": round(f["ratio"], 2), "onset": f["onset"]})

        def cause(cat: str, conf: float, why: str) -> None:
            causes.append({"category": cat, "confidence": conf, "reason": why})

        specific = False
        mem = facts.get("memory_pct")
        if mem and mem["last"] >= 85 and mem["last"] - mem["first"] >= 25 and mem["mono"] >= 0.7:
            add("memory_pct", "monotonic_growth")
            cause("memory_leak", 0.90, "memory grows monotonically toward saturation")
            specific = True
        rps, cpu = facts.get("rps"), facts.get("cpu_pct")
        if rps and rps["ratio"] >= 3:
            add("rps", "traffic_surge")
            if cpu and cpu["last"] >= 85:
                cause("traffic_spike", 0.90, f"rps {rps['ratio']:.1f}x baseline with CPU saturated")
            else:
                cause("traffic_spike", 0.60, f"rps {rps['ratio']:.1f}x baseline")
            specific = True
        if cpu and cpu["last"] >= 90 and "rps" not in {a["metric"] for a in anomalies}:
            add("cpu_pct", "cpu_saturation")
        disk = facts.get("disk_used_pct")
        if disk and disk["last"] >= 95:
            add("disk_used_pct", "capacity_exhausted")
            cause("disk_full", 0.95, f"disk at {disk['last']:.0f}%")
            specific = True
        up = facts.get("upstream_error_pct")
        if up and up["last"] >= 50:
            add("upstream_error_pct", "dependency_errors")
            cause("dependency_outage", 0.85 if not cpu or cpu["last"] < 75 else 0.60, "upstream calls failing while local CPU is normal")
            specific = True
        err = facts.get("error_rate_pct")
        if err and err["ratio"] >= 5:
            add("error_rate_pct", "error_rate_step")
            if not specific:
                cause("bad_deploy", 0.40, "step change in error rate without capacity/dependency signal")
                cause("config_error", 0.40, "step change in error rate without capacity/dependency signal")
        lat = facts.get("latency_p99_ms")
        if lat and lat["ratio"] >= 3:
            add("latency_p99_ms", "latency_regression")
        return {
            "summary": f"{len(anomalies)} anomalous metrics out of {len(facts)}.",
            "anomalies": anomalies, "suspected_causes": causes, "tool_requests": [], "final": True,
        }

    # ----------------------------------------------------------- deploy agent
    def _h_deploy_history(self, ctx: dict, ev: dict, guidance: list[str]) -> dict:
        svc = ctx.get("incident", {}).get("service")
        changes: list[dict] = []
        for src, v in ev.items():
            if src.startswith("tool:list_deploys") and isinstance(v, dict):
                for d in v.get("deploys", []):
                    changes.append({
                        "id": d["id"], "service": d["service"], "type": d["type"], "version": d.get("version"),
                        "previous_version": d.get("previous_version"), "minutes_before_alert": d["minutes_ago"],
                        "summary": str(d.get("summary", ""))[:200], "author": str(d.get("author", "")),
                        "config_key": d.get("config_key"), "previous_revision": d.get("previous_revision"),
                    })
        changes.sort(key=lambda c: c["minutes_before_alert"])
        suspect = next((c for c in changes if c["type"] in {"code", "config", "infra"} and c["minutes_before_alert"] <= 60), None)
        causes = []
        if suspect:
            recency = 0.9 if suspect["minutes_before_alert"] <= 45 else 0.7
            scale = 1.0 if suspect["service"] == svc else 0.5
            cat = {"code": "bad_deploy", "config": "config_error", "infra": "bad_deploy"}[suspect["type"]]
            causes.append({"category": cat, "confidence": round(recency * scale, 3), "reason": f"{suspect['type']} change {suspect['id']} landed {suspect['minutes_before_alert']:.0f} min before the alert"})
        return {
            "summary": f"{len(changes)} changes in window; suspect={suspect['id'] if suspect else 'none'}.",
            "recent_changes": changes, "suspect_change": suspect, "suspected_causes": causes, "tool_requests": [], "final": True,
        }

    # ---------------------------------------------------------------- runbook
    def _h_runbook(self, ctx: dict, ev: dict, guidance: list[str]) -> dict:
        results: list[dict] = []
        for src, v in ev.items():
            if src.startswith("tool:search_runbooks") and isinstance(v, dict):
                results.extend(v.get("results", []))
        results.sort(key=lambda r: -r.get("score", 0))
        matches = [{"doc_id": r["doc_id"], "title": r["title"], "score": round(r["score"], 3), "category": r.get("category", "unknown")} for r in results[:3]]
        steps = [str(s)[:200] for s in (results[0].get("steps", []) if results else [])][:6]
        causes = []
        if results and results[0]["score"] >= 1.0 and results[0].get("category", "unknown") != "unknown":
            causes.append({"category": results[0]["category"], "confidence": round(min(0.8, 0.3 + 0.1 * results[0]["score"]), 3), "reason": f"top runbook: {results[0]['title']}"})
        return {"summary": f"{len(matches)} runbooks matched.", "matches": matches, "recommended_steps": steps, "suspected_causes": causes, "tool_requests": [], "final": True}

    # ------------------------------------------------------------ remediation
    def _h_remediation(self, ctx: dict, ev: dict, guidance: list[str]) -> dict:
        inc = ctx.get("incident", {})
        svc = inc.get("service", "unknown")
        by_agent: dict[str, list[dict]] = defaultdict(list)
        for src, v in ev.items():
            parts = src.split(":")
            if parts[0] == "agent" and len(parts) >= 3 and isinstance(v, dict) and not v.get("_unavailable"):
                by_agent[parts[2]].append(v)
        sinfo = next((v for s, v in ev.items() if s.startswith("tool:get_service_info") and isinstance(v, dict)), {})

        totals: dict[str, float] = defaultdict(float)
        evidence: list[str] = []
        contributions: dict[str, float] = {}
        weight_sum = 0.0
        for agent, outs in by_agent.items():
            w = AGENT_WEIGHT.get(agent)
            if w is None:
                continue
            weight_sum += w
            for o in outs:
                for c in o.get("suspected_causes", []):
                    totals[c["category"]] += w * c["confidence"]
                    evidence.append(f"{agent}: {c['reason']} ({c['category']} {c['confidence']:.2f})")
        ranked = sorted(totals.items(), key=lambda kv: -kv[1])
        n_triage = len({a for a in by_agent if a in {"log_analyst", "metrics_analyst", "deploy_history"}})
        if not ranked or weight_sum == 0 or ranked[0][1] / weight_sum < 0.30:
            cat, conf = "unknown", round(min(0.3, (ranked[0][1] / weight_sum) if ranked and weight_sum else 0.0), 3)
        else:
            cat, best = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = (best - second) / best
            coverage = math.sqrt(min(1.0, max(n_triage, 1) / 3))
            conf = round(min(0.99, (0.5 + 0.5 * margin) * min(1.0, best / 1.5) * coverage), 3)
        contributions = {k: round(v, 3) for k, v in ranked[:4]}

        suspect = next((o.get("suspect_change") for o in by_agent.get("deploy_history", []) if o.get("suspect_change")), None)
        sig_text = " ".join(s.get("sample", "") for o in by_agent.get("log_analyst", []) for s in o.get("error_signatures", []))
        actions: list[dict] = []
        notes: list[str] = []
        banned = self._banned_actions(guidance)
        if cat != "unknown":
            actions = self._actions_for(cat, svc, sinfo, suspect, sig_text)
            for a in list(actions):
                if a["action"] in banned:
                    actions.remove(a)
                    notes.append(f"operator guidance excluded {a['action']}")
                    alt = {"rollback_deploy": "restart_service", "restart_service": "scale_service"}.get(a["action"])
                    if alt and alt not in banned:
                        actions.append({"action": alt, "params": {"service": svc} if alt == "restart_service" else {"service": svc, "replicas": int(sinfo.get("replicas", 2)) * 2}, "rationale": f"fallback after guidance excluded {a['action']}", "expected_effect": "partial mitigation"})
            if not actions:
                notes.append("no safe automated action available; manual investigation required")
        summary = f"Root cause: {CATEGORY_LABEL[cat]} (confidence {conf:.2f})."
        if notes:
            summary += " " + "; ".join(notes) + "."
        return {
            "summary": summary,
            "root_cause": {"category": cat, "confidence": conf, "summary": CATEGORY_LABEL[cat], "evidence": evidence[:8], "contributions": contributions},
            "proposed_actions": actions, "tool_requests": [], "final": True,
        }

    @staticmethod
    def _banned_actions(guidance: list[str]) -> set[str]:
        banned: set[str] = set()
        for g in guidance:
            for m in re.finditer(r"(?:avoid|no|don'?t|do not|never|without)\s+(rollback|restart|scal(?:e|ing)|config revert|failover)", g, re.I):
                w = m.group(1).lower()
                banned.add({"rollback": "rollback_deploy", "restart": "restart_service", "config revert": "revert_config", "failover": "failover_dependency"}.get(w, "scale_service"))
        return banned

    @staticmethod
    def _actions_for(cat: str, svc: str, sinfo: dict, suspect: dict | None, sig_text: str) -> list[dict]:
        def act(action: str, params: dict, why: str, effect: str) -> list[dict]:
            return [{"action": action, "params": params, "rationale": why, "expected_effect": effect}]

        if cat == "bad_deploy":
            prev = (suspect or {}).get("previous_version") or sinfo.get("previous_version")
            if not prev:
                return []
            return act("rollback_deploy", {"service": svc, "to_version": prev}, f"error spike began right after deploy {(suspect or {}).get('id', '?')}", f"restore {prev}")
        if cat == "memory_leak":
            return act("restart_service", {"service": svc, "strategy": "rolling"}, "reclaim leaked memory; leak needs a follow-up fix", "memory returns to baseline")
        if cat == "traffic_spike":
            cur = int(sinfo.get("replicas", 2) or 2)
            return act("scale_service", {"service": svc, "replicas": min(50, max(cur * 3, cur + 2))}, "demand exceeds provisioned capacity", "queue drains, 429s stop")
        if cat == "config_error":
            if not suspect or not suspect.get("config_key") or not suspect.get("previous_revision"):
                return []
            return act("revert_config", {"service": svc, "key": suspect["config_key"], "to_revision": suspect["previous_revision"]}, f"config change {suspect['id']} precedes the failures", "authentication succeeds again")
        if cat == "disk_full":
            cur = int(sinfo.get("volume_gb", 0) or 0)
            if not cur:
                return []
            return act("expand_volume", {"service": svc, "size_gb": math.ceil(cur * 1.5)}, "data volume is full", "writes succeed")
        if cat == "dependency_outage":
            alts = sinfo.get("dependency_alternates", {})
            dep = next((d for d in alts if d in sig_text), None) or next(iter(alts), None)
            if not dep:
                return []
            return act("failover_dependency", {"service": svc, "dependency": dep, "target": alts[dep]}, f"{dep} is failing; a secondary is configured", "traffic shifts to the secondary")
        return []

    # ------------------------------------------------------------------ comms
    def _h_comms(self, ctx: dict, ev: dict, guidance: list[str]) -> dict:
        inc = ctx.get("incident", {})
        kind = ctx.get("params", {}).get("kind", "update")
        svc, sev = inc.get("service", "unknown"), inc.get("severity", "sev2")
        rem = next((v for s, v in ev.items() if ":remediation" in s and isinstance(v, dict) and "root_cause" in v), None)
        lines = [f"[{sev.upper()}] {svc}: {inc.get('title', 'incident')}"]
        if kind == "ack" or not rem:
            lines += ["We are aware of the issue and an incident response is in progress.", "Status: investigating. Next update in 15 minutes."]
            nxt = 15
        else:
            rc = rem["root_cause"]
            lines.append(f"Likely cause: {CATEGORY_LABEL.get(rc['category'], 'unknown')} (confidence {rc['confidence']:.0%}).")
            acts = rem.get("proposed_actions", [])
            if not acts:
                lines.append("Mitigation: no automated action proposed; engineers are investigating manually.")
            for a in acts:
                st = a.get("status", "proposed")
                who = f" by {a['decided_by']}" if a.get("decided_by") else ""
                lines.append(f"Mitigation {a['action']} on {a.get('params', {}).get('service', svc)}: {st}{who}.")
            nxt = 30
            lines.append(f"Next update in {nxt} minutes.")
        for g in guidance:
            lines.append(f"Operator note: {g[:200]}")
        return {"summary": "status update drafted", "title": f"{svc} incident update", "body": "\n".join(lines), "audience": "internal" if sev != "sev1" else "internal+customers", "next_update_minutes": nxt, "tool_requests": [], "final": True}
