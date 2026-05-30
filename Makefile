.PHONY: eval eval-offline test test-cov test-snapshots lint

PYTHON := .venv/bin/python
GOLDEN := evals/golden_prs.yaml
OUTPUT := evals/results/latest.csv
BASELINE := evals/results/baseline.csv

# Run full RAG evaluation (requires LLM API key in env)
eval:
	$(PYTHON) evals/run_eval.py \
		--golden $(GOLDEN) \
		--output $(OUTPUT) \
		--baseline $(BASELINE)

# Offline mode (no LLM calls — for CI)
eval-offline:
	$(PYTHON) evals/run_eval.py \
		--golden $(GOLDEN) \
		--output $(OUTPUT) \
		--offline

# Save current results as the new baseline
eval-set-baseline: eval
	cp $(OUTPUT) $(BASELINE)
	@echo "Baseline updated: $(BASELINE)"

# Run unit tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# Run unit tests with coverage (fails if coverage < 75%)
test-cov:
	$(PYTHON) -m pytest tests/ \
		--ignore=tests/test_integration.py \
		--ignore=tests/test_phase5_integration.py \
		--ignore=tests/test_security.py \
		--ignore=tests/e2e \
		--cov=src --cov-report=term-missing --cov-fail-under=75 -q

# Run snapshot tests only (no LLM calls needed)
test-snapshots:
	$(PYTHON) -m pytest tests/test_m11_snapshots.py -v

# Regenerate all pinned snapshots (run after intentional template changes)
update-snapshots:
	UPDATE_SNAPSHOTS=1 $(PYTHON) -m pytest tests/test_m11_snapshots.py -v

# Lint
lint:
	$(PYTHON) -m ruff check src/ evals/ tests/ 2>/dev/null || true
