# Targets for the data and training slice (P2). The top-level Makefile (P4) includes this file:
#     include ml/Makefile.inc
# Kaggle-hosted sets need KAGGLE_API_TOKEN in .env; sets that cannot be fetched report why and the rest continue.

PYTHON ?= python

.PHONY: data data-extended data-verify splits ml-test notebooks

data: ## download every core dataset, check manifests, build leakage-audited splits
	$(PYTHON) ml/data/download.py core
	$(PYTHON) ml/data/download.py core --verify
	$(PYTHON) ml/data/make_splits.py all

data-extended: ## also fetch ISIC Task 1 eval sets, RSNA pneumonia and CIFAR-10 (about 7 GB more)
	$(PYTHON) ml/data/download.py all

data-verify: ## re-hash every downloaded file against its manifest
	$(PYTHON) ml/data/download.py all --verify

splits: ## rebuild split files and reports/leakage.json from the downloaded data
	$(PYTHON) ml/data/make_splits.py all

ml-test: ## unit and synthetic end-to-end tests for ml/
	$(PYTHON) -m pytest ml/tests -q

notebooks: ## rebuild ml/train/notebooks/*.ipynb from ml/train/src/*.py
	$(PYTHON) ml/train/build_notebooks.py
