# Root Makefile (P4). Includes ml/'s own targets (P2, see progress.md Decisions, 2026-10-06).
PYTHON ?= python

include ml/Makefile.inc

.PHONY: setup dev test eval demo smoke contracts

setup: ## install backend (editable, dev+ml extras), web deps, and pre-commit hooks
	$(PYTHON) -m pip install -e "backend[dev,ml]"
	cd web && pnpm install
	pre-commit install

contracts: ## regenerate contracts/*.schema.json and web/src/contracts.ts from schemas.py
	$(PYTHON) scripts/export_contract.py
	node scripts/gen_contracts_ts.mjs

test: ## everything: backend unit tests + ml unit/synthetic tests
	cd backend && $(PYTHON) -m pytest -q
	$(MAKE) ml-test

# fast subset for pre-merge: backend tests (incl. contract tests). `tests/api/test_studies.py`
# already runs a fixture image through the real pipeline + API end to end in well under 60s,
# which is plan.md section 8's smoke criterion in substance; there's no separate script for it.
smoke:
	cd backend && $(PYTHON) -m pytest -q

dev: ## run the API with autoreload (web/'s dev server is still a separate `cd web && pnpm dev` until P4.6)
	cd backend && $(PYTHON) -m uvicorn medproof.api.app:app --reload

eval: ## not yet implemented: see P2.14
	@echo "make eval: not yet implemented, see P2.14"

demo: ## not yet implemented: see P4.9
	@echo "make demo: not yet implemented, see P4.9"
