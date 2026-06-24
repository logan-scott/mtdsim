# Makefile for the MTD-vs-AI attacker simulation.
# All targets assume a Python 3.11+ interpreter (PY) on PATH.

PY ?= python3
CONFIG ?= configs/paper.yaml
OUTDIR ?= results

.PHONY: help venv install test lint fmt all clean distclean

help:
	@echo "Targets:"
	@echo "  install   - install package + pinned deps (editable)"
	@echo "  test      - run the full pytest suite"
	@echo "  test-fast - run tests excluding slow markers"
	@echo "  all       - regenerate every figure + table + manifest from CONFIG"
	@echo "  lint      - ruff check (if installed)"
	@echo "  fmt       - black + ruff --fix (if installed)"
	@echo "  clean     - remove generated results/figures/tables"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

# Headline one-shot regeneration: every figure, table, and the manifest.
all:
	$(PY) -m mtdsim.experiments.run_all --config $(CONFIG) --outdir $(OUTDIR)

lint:
	-$(PY) -m ruff check src tests

fmt:
	-$(PY) -m black src tests
	-$(PY) -m ruff check --fix src tests

clean:
	rm -rf figures/* tables/* $(OUTDIR)/*

distclean: clean
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
