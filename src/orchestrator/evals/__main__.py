"""``python -m orchestrator.evals`` - print the report; exit non-zero below thresholds (CI gate)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from orchestrator.evals.cases import build_cases
from orchestrator.evals.runner import format_report, run_suite


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m orchestrator.evals")
    ap.add_argument("--ablation", action="store_true", help="also run the adversarial set with the scanner / allow-list disabled")
    ap.add_argument("--filter", help="substring of case id")
    ap.add_argument("--json", help="write machine-readable results to this path")
    args = ap.parse_args(argv)
    cases = [c for c in build_cases() if not args.filter or args.filter in c.id]
    rep = asyncio.run(run_suite(cases, ablation=args.ablation))
    print(format_report(rep))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"metrics": rep.metrics, "breakdown": rep.breakdown, "ablation": rep.ablation, "cases": [{"id": r.case.id, "passed": r.passed, "status": r.status, "root_cause": r.root_cause} for r in rep.results]}, f, indent=2)
    return 0 if not rep.failures() else 1


if __name__ == "__main__":
    sys.exit(main())
