# Terser task runner (POSIX). Mirror of tasks.ps1.
PY ?= python
# The project venv's interpreter: install and the targets below run inside .venv.
ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python
else
VENV_PY := .venv/bin/python
endif

.PHONY: help install test test-all lint format typecheck hooks data \
        train-classical eval mlflow-ui gpu-queue hello clean

help:
	@echo "Usage: make <target>"
	@echo "  help             List available targets (default)"
	@echo "  install          Create .venv if missing and install the CI extras into it"
	@echo "  test             Run CPU tests (not gpu, not api)"
	@echo "  test-all         Run the full test suite"
	@echo "  lint             ruff check + black --check"
	@echo "  format           ruff check --fix + black"
	@echo "  typecheck        mypy src"
	@echo "  hooks            pre-commit run --all-files"
	@echo "  data             dvc repro (if dvc.yaml exists)"
	@echo "  train-classical  Train classical baselines (Phase 3)"
	@echo "  eval             Run evaluation (Phase 3)"
	@echo "  mlflow-ui        Start the MLflow UI on ./mlruns"
	@echo "  gpu-queue        Dry-run the GPU job queue"
	@echo "  hello            Run the hello smoke module"
	@echo "  clean            Remove caches (never data/ or mlruns/)"

install:
	test -d .venv || $(PY) -m venv .venv
	# The same extras as CI, so a local `make test` runs the tests CI runs.
	$(VENV_PY) -m pip install -e ".[dev,hpo,serve,mlops,orchestrate]"

test:
	$(VENV_PY) -m pytest -m "not gpu and not api"

test-all:
	$(VENV_PY) -m pytest

lint:
	$(VENV_PY) -m ruff check .
	$(VENV_PY) -m black --check .

format:
	$(VENV_PY) -m ruff check --fix .
	$(VENV_PY) -m black .

typecheck:
	$(VENV_PY) -m mypy src

hooks:
	$(VENV_PY) -m pre_commit run --all-files

data:
	@if [ -f dvc.yaml ]; then $(VENV_PY) -m dvc repro; \
	else echo "No dvc.yaml pipeline yet; nothing to reproduce."; fi

train-classical:
	$(VENV_PY) -m terser.phase3.studies --help

eval:
	@echo "eval: the final evaluation is python -m terser.phase12.results --split test ... (see its --help); it needs the row 12.1 unlock."

mlflow-ui:
	$(VENV_PY) -m mlflow ui --backend-store-uri ./mlruns

gpu-queue:
	$(VENV_PY) -m terser.gpu_queue dry-run

hello:
	$(VENV_PY) -m terser.hello

clean:
	find . \( -path ./.venv -o -path ./.git -o -path ./data -o -path ./mlruns \) -prune -o \
		-type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache \
		-o -name .ruff_cache -o -name htmlcov -o -name '*.egg-info' \) -print -exec rm -rf {} +
	rm -f .coverage
