"""Durable SQLite *domain* store: the run list, a read-model of task state, spans, events, audit log,
approvals, idempotency records, dead letters, queued human commands and run leases.

What it is not: the execution checkpoint. Run state (plan, per-task results, pending interrupts) is owned
by LangGraph's checkpointer (``core/checkpoint.py``), which shares this connection. The ``runs`` and
``tasks`` rows are projections for the API/dashboard and are never used to decide what to re-run.

Design notes
- One connection in autocommit mode + WAL: every write is its own durable transaction.
- Calls are synchronous. Each statement is microseconds-scale; this keeps the driver simple and
  deterministic. A Postgres implementation would sit behind the same method surface.
- All payloads are JSON text; callers are responsible for redaction *before* persisting.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, status TEXT NOT NULL, incident TEXT NOT NULL, plan TEXT,
  budget TEXT, usage TEXT, result TEXT, guidance TEXT, paused INTEGER DEFAULT 0, taint INTEGER DEFAULT 0,
  breakpoints TEXT, created_at REAL, updated_at REAL, lease_owner TEXT, lease_expires REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tasks(
  run_id TEXT, task_id TEXT, status TEXT NOT NULL, attempts INTEGER DEFAULT 0, output TEXT, error TEXT,
  reason TEXT, input_overrides TEXT, released INTEGER DEFAULT 0, human_skipped INTEGER DEFAULT 0,
  started_at REAL, finished_at REAL, PRIMARY KEY(run_id, task_id)
);
CREATE TABLE IF NOT EXISTS spans(
  span_id TEXT PRIMARY KEY, run_id TEXT, parent_id TEXT, task_id TEXT, kind TEXT, name TEXT,
  start_ts REAL, end_ts REAL, latency_ms REAL, status TEXT, input TEXT, output TEXT, attrs TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id);
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, type TEXT, data TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq);
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, actor TEXT, action TEXT, target TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_run ON audit(run_id);
CREATE TABLE IF NOT EXISTS approvals(
  approval_id TEXT PRIMARY KEY, run_id TEXT, task_id TEXT, status TEXT NOT NULL, proposal TEXT, risk TEXT,
  final_params TEXT, decided_by TEXT, decided_at REAL, note TEXT, created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id);
CREATE TABLE IF NOT EXISTS idempotency(key TEXT PRIMARY KEY, tool TEXT, result TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS commands(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, op TEXT, patch TEXT, applied INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_commands_run ON commands(run_id, applied);
CREATE TABLE IF NOT EXISTS dead_letters(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, task_id TEXT, error TEXT, payload TEXT, ts REAL
);
"""


def _dumps(v: Any) -> str:
    return json.dumps(v, default=str, separators=(",", ":"))


def _loads(v: str | None) -> Any:
    return None if v is None else json.loads(v)


class Store:
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            if path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("PRAGMA busy_timeout=5000")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _x(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, args)

    # ------------------------------------------------------------------ runs
    def create_run(self, run_id: str, incident: dict, budget: dict, breakpoints: list[str]) -> None:
        now = time.time()
        self._x(
            "INSERT INTO runs(run_id,status,incident,budget,usage,guidance,breakpoints,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (run_id, "pending", _dumps(incident), _dumps(budget), _dumps({}), _dumps([]), _dumps(breakpoints), now, now),
        )

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols, vals = [], []
        for k, v in fields.items():
            cols.append(f"{k}=?")
            vals.append(_dumps(v) if k in {"plan", "usage", "result", "guidance", "budget", "incident", "breakpoints"} else v)
        cols.append("updated_at=?")
        vals.append(time.time())
        self._x(f"UPDATE runs SET {', '.join(cols)} WHERE run_id=?", (*vals, run_id))

    def get_run(self, run_id: str) -> dict | None:
        row = self._x("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return self._run_row(row) if row else None

    def list_runs(self, limit: int = 100) -> list[dict]:
        rows = self._x("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._run_row(r) for r in rows]

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        for k in ("incident", "plan", "budget", "usage", "result", "guidance", "breakpoints"):
            d[k] = _loads(d.get(k))
        d["paused"] = bool(d["paused"])
        d["taint"] = bool(d["taint"])
        return d

    # ----------------------------------------------------------------- leases
    def claim_run(self, run_id: str, owner: str, lease_s: float) -> bool:
        """Atomically take (or renew) the execution lease for a run. Expired leases can be stolen."""
        now = time.time()
        cur = self._x(
            "UPDATE runs SET lease_owner=?, lease_expires=? WHERE run_id=? AND (lease_owner IS NULL OR "
            "lease_owner=? OR lease_expires<?)",
            (owner, now + lease_s, run_id, owner, now),
        )
        return cur.rowcount == 1

    def orphaned_runs(self) -> list[str]:
        """Unfinished runs whose worker died: a lease that was taken and has since expired."""
        rows = self._x(
            "SELECT run_id FROM runs WHERE status NOT IN ('completed','degraded','failed','budget_exceeded','cancelled') "
            "AND lease_owner IS NOT NULL AND lease_expires < ? ORDER BY created_at", (time.time(),),
        ).fetchall()
        return [r["run_id"] for r in rows]

    def release_run(self, run_id: str, owner: str) -> None:
        self._x("UPDATE runs SET lease_owner=NULL, lease_expires=0 WHERE run_id=? AND lease_owner=?", (run_id, owner))

    # ------------------------------------------------------------------ tasks
    def save_task(self, run_id: str, t: dict) -> None:
        self._x(
            "INSERT INTO tasks(run_id,task_id,status,attempts,output,error,reason,input_overrides,released,"
            "human_skipped,started_at,finished_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id,task_id) DO UPDATE SET status=excluded.status, attempts=excluded.attempts, "
            "output=excluded.output, error=excluded.error, reason=excluded.reason, "
            "input_overrides=excluded.input_overrides, released=excluded.released, "
            "human_skipped=excluded.human_skipped, started_at=excluded.started_at, finished_at=excluded.finished_at",
            (
                run_id, t["task_id"], t["status"], t["attempts"], _dumps(t["output"]) if t["output"] is not None else None,
                t["error"], t["reason"], _dumps(t["input_overrides"]), int(t["released"]), int(t["human_skipped"]),
                t["started_at"], t["finished_at"],
            ),
        )

    def get_tasks(self, run_id: str) -> dict[str, dict]:
        out = {}
        for r in self._x("SELECT * FROM tasks WHERE run_id=?", (run_id,)).fetchall():
            d = dict(r)
            d["output"] = _loads(d["output"])
            d["input_overrides"] = _loads(d["input_overrides"]) or {}
            d["released"] = bool(d["released"])
            d["human_skipped"] = bool(d["human_skipped"])
            out[d["task_id"]] = d
        return out

    def delete_task(self, run_id: str, task_id: str) -> None:
        self._x("DELETE FROM tasks WHERE run_id=? AND task_id=?", (run_id, task_id))

    # ------------------------------------------------------------------ spans
    def add_span(self, s: dict) -> None:
        self._x(
            "INSERT OR REPLACE INTO spans(span_id,run_id,parent_id,task_id,kind,name,start_ts,end_ts,latency_ms,"
            "status,input,output,attrs) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                s["span_id"], s["run_id"], s["parent_id"], s["task_id"], s["kind"], s["name"], s["start_ts"], s["end_ts"],
                s["latency_ms"], s["status"], _dumps(s["input"]), _dumps(s["output"]), _dumps(s["attrs"]),
            ),
        )

    def get_spans(self, run_id: str) -> list[dict]:
        rows = self._x("SELECT * FROM spans WHERE run_id=? ORDER BY start_ts, rowid", (run_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("input", "output", "attrs"):
                d[k] = _loads(d[k])
            out.append(d)
        return out

    # ----------------------------------------------------------------- events
    def append_event(self, run_id: str, ts: float, type_: str, data: dict) -> int:
        cur = self._x("INSERT INTO events(run_id,ts,type,data) VALUES(?,?,?,?)", (run_id, ts, type_, _dumps(data)))
        return int(cur.lastrowid or 0)

    def get_events(self, run_id: str, after_seq: int = 0, limit: int = 5000) -> list[dict]:
        rows = self._x(
            "SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?", (run_id, after_seq, limit)
        ).fetchall()
        return [{**dict(r), "data": _loads(r["data"])} for r in rows]

    # ------------------------------------------------------------------ audit
    def audit(self, run_id: str, actor: str, action: str, target: str = "", detail: dict | None = None) -> None:
        self._x(
            "INSERT INTO audit(run_id,ts,actor,action,target,detail) VALUES(?,?,?,?,?,?)",
            (run_id, time.time(), actor, action, target, _dumps(detail or {})),
        )

    def get_audit(self, run_id: str) -> list[dict]:
        rows = self._x("SELECT * FROM audit WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
        return [{**dict(r), "detail": _loads(r["detail"])} for r in rows]

    # -------------------------------------------------------------- approvals
    def create_approval(self, approval_id: str, run_id: str, task_id: str, proposal: dict, risk: dict) -> bool:
        cur = self._x(
            "INSERT OR IGNORE INTO approvals(approval_id,run_id,task_id,status,proposal,risk,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (approval_id, run_id, task_id, "pending", _dumps(proposal), _dumps(risk), time.time()),
        )
        return cur.rowcount == 1

    def get_approval(self, approval_id: str) -> dict | None:
        r = self._x("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
        return self._approval_row(r) if r else None

    def list_approvals(self, run_id: str | None = None, status: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM approvals WHERE 1=1", []
        if run_id:
            sql += " AND run_id=?"
            args.append(run_id)
        if status:
            sql += " AND status=?"
            args.append(status)
        rows = self._x(sql + " ORDER BY created_at, approval_id", tuple(args)).fetchall()
        return [self._approval_row(r) for r in rows]

    def decide_approval(
        self, approval_id: str, status: str, actor: str, note: str = "", final_params: dict | None = None
    ) -> bool:
        """Compare-and-set pending -> decided. Returns False if it was already decided."""
        cur = self._x(
            "UPDATE approvals SET status=?, decided_by=?, decided_at=?, note=?, final_params=? "
            "WHERE approval_id=? AND status='pending'",
            (status, actor, time.time(), note, _dumps(final_params) if final_params is not None else None, approval_id),
        )
        return cur.rowcount == 1

    @staticmethod
    def _approval_row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("proposal", "risk", "final_params"):
            d[k] = _loads(d[k])
        return d

    # ------------------------------------------------------------ idempotency
    def idem_get(self, key: str) -> Any | None:
        r = self._x("SELECT result FROM idempotency WHERE key=?", (key,)).fetchone()
        return None if r is None else {"result": _loads(r["result"])}

    def idem_put(self, key: str, tool: str, result: Any) -> None:
        self._x("INSERT OR REPLACE INTO idempotency(key,tool,result,ts) VALUES(?,?,?,?)", (key, tool, _dumps(result), time.time()))

    # --------------------------------------------------- human commands (state patches)
    def add_command(self, run_id: str, op: str, patch: dict) -> int:
        """Queue a human edit as a *state patch*. The graph driver folds it into the LangGraph state at the
        next safe point (a superstep boundary or a parked graph) and marks it applied. Patches are absolute,
        so re-applying one after a crash is harmless."""
        cur = self._x("INSERT INTO commands(run_id,ts,op,patch) VALUES(?,?,?,?)", (run_id, time.time(), op, _dumps(patch)))
        return int(cur.lastrowid or 0)

    def pending_commands(self, run_id: str) -> list[dict]:
        rows = self._x("SELECT * FROM commands WHERE run_id=? AND applied=0 ORDER BY id", (run_id,)).fetchall()
        return [{"id": r["id"], "op": r["op"], "patch": _loads(r["patch"])} for r in rows]

    def mark_commands_applied(self, ids: list[int]) -> None:
        for i in ids:
            self._x("UPDATE commands SET applied=1 WHERE id=?", (i,))

    # ----------------------------------------------------------- dead letters
    def add_dead_letter(self, run_id: str, task_id: str, error: str, payload: dict) -> None:
        self._x(
            "INSERT INTO dead_letters(run_id,task_id,error,payload,ts) VALUES(?,?,?,?,?)",
            (run_id, task_id, error, _dumps(payload), time.time()),
        )

    def get_dead_letters(self, run_id: str | None = None) -> list[dict]:
        if run_id:
            rows = self._x("SELECT * FROM dead_letters WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
        else:
            rows = self._x("SELECT * FROM dead_letters ORDER BY id").fetchall()
        return [{**dict(r), "payload": _loads(r["payload"])} for r in rows]
