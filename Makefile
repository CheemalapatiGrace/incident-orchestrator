PY      ?= .venv/bin/python
PIP     ?= .venv/bin/pip
SCENARIO ?= checkout-bad-deploy

.PHONY: install test lint eval run serve bench chaos graph ci clean

install:  ## create .venv and install the package (pulls in langgraph + langgraph-checkpoint-sqlite) with dev dependencies
	python3.12 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:  ## unit + integration tests (fully offline; includes the LangGraph property tests)
	$(PY) -m pytest tests

lint:
	.venv/bin/ruff check src tests

eval:  ## golden-set eval + defence-in-depth ablation; exits non-zero below thresholds (CI gate)
	$(PY) -m orchestrator.evals --ablation

run:  ## run one incident in the terminal with interactive approval (SCENARIO=...)
	$(PY) -m orchestrator run $(SCENARIO) --db /tmp/io-demo.db

serve:  ## API + live dashboard on http://127.0.0.1:8000
	$(PY) -m orchestrator serve --db incident_orchestrator.db

bench:  ## throughput / latency vs worker count
	$(PY) -m orchestrator.bench -n 120

chaos:  ## fault injection: random failures, breaker, crash+resume, crash at approval gate
	$(PY) -m orchestrator.chaos -n 60

graph:  ## regenerate docs/graph.md (Mermaid export of the compiled LangGraph; a test fails if it is stale)
	$(PY) -m orchestrator.graphdoc

ci: lint test eval chaos  ## what the GitHub Action runs (tests include the docs/graph.md freshness check)

clean:
	rm -rf .pytest_cache .ruff_cache *.db *.db-wal *.db-shm build *.egg-info
