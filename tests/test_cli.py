import asyncio
import json

import pytest

from conftest import alert_for, make_engine, wait_pending
from orchestrator.cli import main
from orchestrator.hitl.terminal import TerminalApprover


def test_scenarios_command(capsys):
    assert main(["scenarios"]) == 0
    out = capsys.readouterr().out
    assert "checkout-bad-deploy" in out and "notifications-dependency-outage" in out


def test_run_with_auto_approval_prints_result_and_exits_zero(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("IO_RETRY_BASE_DELAY_S", "0.001")
    rc = main(["run", "inventory-disk-full", "--approve", "auto", "--db", str(tmp_path / "c.db")])
    out = capsys.readouterr().out
    assert rc == 0 and '"root_cause": "disk_full"' in out and "approval.requested" in out and "cli:auto-approve" in out


def test_run_with_reject_does_not_execute(capsys, tmp_path):
    assert main(["run", "payments-memory-leak", "--approve", "reject", "--db", str(tmp_path / "c.db")]) == 0
    assert '"status": "rejected"' in capsys.readouterr().out


async def test_terminal_approver_approve_edit_reject_flow():
    eng = make_engine()
    rid = eng.submit(alert_for("search-traffic-spike"))
    answers = iter(["e", '{"replicas": 9}', "scaling for traffic"])
    shown = []
    appr = TerminalApprover(eng, input_fn=lambda prompt: next(answers), out=shown.append, actor="cli:tester")
    watcher = asyncio.create_task(appr.watch(rid))
    res = await eng.execute(rid)
    await asyncio.wait_for(watcher, 5)
    assert eng._envs[rid].mutations[0]["params"]["replicas"] == 9
    assert res.actions[0]["decided_by"] == "cli:tester" and "APPROVAL REQUIRED" in shown[0] and "scale_service" in shown[0]


async def test_terminal_approver_requires_note_for_high_risk_then_reprompts():
    eng = make_engine()
    rid = eng.submit(alert_for("checkout-bad-deploy"))
    answers = iter(["a", "", "a", "dep-9812 caused it"])  # first attempt has an empty note -> refused -> asks again
    shown = []
    watcher = asyncio.create_task(TerminalApprover(eng, input_fn=lambda p: next(answers), out=shown.append).watch(rid))
    await eng.execute(rid)
    await asyncio.wait_for(watcher, 5)
    assert any("justification" in s for s in shown) and eng._envs[rid].mutations


def test_cross_process_approvals_cli(tmp_path, capsys):
    db = str(tmp_path / "x.db")

    async def scenario():
        eng = make_engine(db_path=db)
        rid = eng.submit(alert_for("inventory-disk-full"))
        task = asyncio.create_task(eng.execute(rid))
        (ap,) = await wait_pending(eng, rid)
        capsys.readouterr()
        assert main(["approvals", "list", "--db", db]) == 0
        assert ap["approval_id"] in capsys.readouterr().out
        # (main() is synchronous and uses its own engine on the same file, exactly like a second shell)
        await asyncio.to_thread(main, ["approvals", "approve", ap["approval_id"], "--db", db, "--actor", "cli:bob", "--note", "volume at 100%"])
        res = await asyncio.wait_for(task, 5)
        return res

    res = asyncio.run(scenario())
    assert res.status.value == "completed" and res.actions[0]["decided_by"] == "cli:bob"


def test_runs_and_trace_commands(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    assert main(["run", "reporting-false-alarm", "--approve", "none", "--db", db]) == 0
    capsys.readouterr()
    assert main(["runs", "--db", db]) == 0
    run_id = capsys.readouterr().out.split()[0]
    assert main(["trace", run_id, "--db", db]) == 0
    out = capsys.readouterr().out
    assert "run " in out and "llm " in out and "tool" in out and "$" in out


def test_approve_requires_id():
    with pytest.raises(SystemExit):
        main(["approvals", "approve"])


def test_json_output_of_run_is_machine_readable(capsys, tmp_path):
    main(["run", "reporting-false-alarm", "--approve", "none", "--db", str(tmp_path / "j.db")])
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{\n"):])
    assert payload["status"] == "completed" and payload["root_cause"] == "unknown" and payload["actions"] == []
