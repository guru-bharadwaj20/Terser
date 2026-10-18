# Checkpoint C0 — Foundations Ready

| Field | Value |
| :--- | :--- |
| Date | 2026-09-10 |
| Git SHA at measurement | `4036b2e` |
| Machine | Windows 11 (10.0.22000), Intel Core i3-5005U, 8 GB RAM, **no GPU** |
| Python | 3.13.1 (`.venv`); project supports >= 3.11 |
| DVC data hash | none (no data tracked yet) |
| Seeds | fixed per test; see `tests/test_seed.py` |

## Checks

| # | Check | Target | Measured | Result |
| :---: | :--- | :--- | :--- | :---: |
| C0.1 | CI | Green on `main` | GitHub Actions run `34499625158` on `3b8582c`: **all 4 jobs green** (ubuntu-latest and windows-latest × Python 3.11 and 3.13: install, ruff, black, mypy, CPU tests). Earlier runs were cancelled only by newer pushes (`cancel-in-progress`) | ✅ |
| C0.2 | Tokenizer correctness | `count_tokens` equals each official tokenizer on a 200-string fixture set exactly (100%) | `tests/test_tokenizers.py`: gpt-4o (`o200k_base`), qwen2.5 and llama-3.1 each match direct library encoding on **200/200** strings in `data/monitoring/tokenizer_strings.jsonl`; snapshot `data/monitoring/tokenizer_counts.json` also matches. Claude counting verified only against a mocked client (no API key available) | ✅ for the three local tokenizers |
| C0.3 | Reproducibility | Same seed ⇒ identical metrics across two runs | `python -m terser.utils.dummy_experiment --seed 0` run twice in fresh subprocesses with different inherited `PYTHONHASHSEED` (1 and 12345): **byte-identical JSON metrics**; seeds 0 and 1 differ | ✅ |
| C0.4 | CPU fallback | Full test suite passes on a machine with no GPU | `pytest -m "not gpu and not api"`: **136 passed, 1 deselected** (the single `gpu` test) in 93.7 s. `get_device(hardware=auto)` → `cpu`; `hardware=rtx4500ada` → `HardwareUnavailableError` with an actionable message (suggests `hardware=cpu`/`auto` or `gpu_queue/`) | ✅ |

## Deviations and known issues

- **No hosted API keys.** The Claude token counter and the Anthropic/OpenAI transports of the budget-guarded client are implemented and unit-tested with mocks, but have never been exercised live. Work that the plan tags `API` needs open-source models served locally instead; an OpenAI-compatible transport is being added to `terser.llm` for this.
- **mypy target.** `[tool.mypy] python_version = "3.12"` because numpy 2.x stubs use PEP 695 syntax that mypy only parses for targets >= 3.12. Runtime support remains >= 3.11.
- **MLflow.** Tracking uses a local SQLite store (`mlruns/mlflow.db`) opened with `mlflow ui`, not a long-running `mlflow server`. Registry stages are implemented as aliases (`staging`, `production`, `archived`) because stage transitions are deprecated in MLflow 3.
- **Makefile** is untested (no `make` on this machine); `tasks.ps1` targets were all executed.
- **Slow tests.** `tests/test_tracking.py` (SQLite registry creation) and `tests/test_seed.py` (subprocess sklearn imports) dominate suite time on this CPU.

## Decision

`PASS`. The condition (a green GitHub Actions run on `main`) was met by run `34499625158`, so C0.1 and row 0.2.9 are ✅.
