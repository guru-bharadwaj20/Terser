# CI duration and coverage (row 11.4.1)

Row 11.4.1: "Every PR: lint, types, unit tests, CPU smoke training, `dvc repro --dry`, feature
parity", done when the **median run is <= 15 min**. Sources: `.github/workflows/ci.yml`, the
GitHub Actions history of `guru-bharadwaj20/Terser` fetched with
`python scripts/ci_run_stats.py --out reports/p11_ci.json` on 2026-10-17 (`gh` authenticated;
every run and every job, not only the latest 50), and one local run of `scripts/ci_local.sh`.

## Result

| Measure (real GitHub runs) | Runs | Median | Max | Source |
| :--- | ---: | ---: | ---: | :--- |
| Successful runs (all on the 2026-09-10/11 workflow, before the 11.4.1 steps were added) | 9 | **302 s (5.0 min)** | 595 s | `success_seconds` |
| All executed, finished runs (success + failure) | 40 | 323.5 s (5.4 min) | 855 s | `median_seconds_executed_finished` |
| Current workflow (since `b84e4df`, 2026-10-01), finished | 21 | 398 s (6.6 min) | 855 s | `executed_runs[full_workflow]` |
| Current workflow, runs that executed every step (they failed in the last step, Tests) | 8 | **639 s (10.7 min)** | 855 s (14.3 min) | same |

Every measure is under the 15-minute bar, including the slowest full-length run of the
current workflow (14.3 min). The four matrix jobs run in parallel; the Windows jobs set the wall
time (median job 256-273 s vs 127-142 s on Ubuntu in the successful runs). In those runs
`Install` (median 94.5 s), `Tests (CPU)` (45.5 s) and `Mypy` (30.5 s) dominate; the test suite
has since grown (2,842 tests locally), which is why full-length current runs take ~10 min. The
workflow's own `timeout-minutes: 15` caps each job.

**But CI is not green.** Pass rate over executed, non-cancelled runs: **9 / 40 = 22.5%**, and
the current workflow has **never passed on GitHub**:

| Current-workflow failures (21 finished runs) | Runs | Cause |
| :--- | ---: | :--- |
| `Black` (every run since 2026-10-03 18:40) | 12 | CI installs the latest black (26.10.1 in the 2026-10-17 run), which would reformat `src/terser/data/pii.py`; the local toolchain is black 26.5.1, under which that file passes. Tool versions are not pinned. |
| `Tests (CPU)` (2026-10-01 to 10-03) | 8 | Test failures at the time (e.g. a `code_runner` sandbox test on 2026-10-03); fixed since |
| `Ruff` | 1 | lint error at that commit |

And since 2026-10-17 03:10 UTC **no job has started at all**: 138 of 224 runs (all of the latest
50) were refused by GitHub with "The job was not started because recent account payments have
failed or your spending limit needs to be increased" (job length 2-4 s, no runner, no steps).
Those runs are excluded from every duration above; they are a billing block, not test failures.
46 further runs were cancelled by the workflow's `concurrency` group (a newer push superseded
them) and are excluded too.

## Coverage of the row's checklist

| Row item | `ci.yml` step | Covered |
| :--- | :--- | :---: |
| Every PR | `on: pull_request` (and `push` to `main`), 2 OS x 2 Python | yes |
| Lint | `ruff check .`, `black --check .` | yes |
| Types | `mypy src` | yes |
| Unit tests | `pytest -m "not gpu and not api" --cov=terser` | yes |
| CPU smoke training | `tests/rewriter/test_train_tiny.py::test_tiny_sft_loss_decreases` (30-step SFT on CPU, row 8.3.2) inside `Tests (CPU)`; CPU-only torch is installed so it runs (the 2026-10-03 CI log shows the file's 3 tests passing) | yes, implicitly |
| `dvc repro --dry` | `dvc repro --dry --allow-missing` | yes, with `--allow-missing` |
| Feature parity | `pytest tests/features -q` (row 11.1.3) | yes |

Extra steps: test-set isolation audit (row 10.8) and tokenizer drift (row 11.5.3).

**Gaps** (not fixed here; the workflow file cannot be pushed with the current token, which
lacks the `workflow` scope):

1. **Unpinned tool versions.** `ruff`, `black` and `mypy` float to the newest release, so CI and
   local disagree (black 26.10.1 vs 26.5.1 today, the cause of 12 of 21 current failures). Pin
   them in the `dev` extra or a constraints file, or run black in CI with the local version.
2. **Smoke training has no step of its own.** It runs inside the full test step and is guarded
   by `pytest.importorskip("torch")`: if the CPU torch install ever fails, the smoke test is
   silently skipped rather than failing CI. A dedicated step
   (`pytest tests/rewriter/test_train_tiny.py`) without the skip would make it explicit.
3. **`dvc repro --dry --allow-missing`** checks the DAG and stage definitions but not that the
   data hashes resolve, because raw data is not pulled in CI. That is weaker than the row's plain
   `dvc repro --dry`, and is documented in the workflow comment.
4. Feature-parity tests run twice (their own step and again in `Tests (CPU)`); harmless, ~2 s.
5. The account's Actions billing block must be lifted before any of this runs on GitHub again;
   until then `scripts/ci_local.sh` is the only way to run the checks.

## Local proxy (clearly not GitHub)

`bash scripts/ci_local.sh --keep-going` on the WSL2 workstation, 2026-10-17 13:22 UTC, HEAD
`22fb4c2` with 32 uncommitted files from parallel work in progress, while other jobs were
running on the same CPU (a second full pytest, mypy, Prefect flows, a Q-learning verifier and a
GPU training job's data loader): **933 s total**, of which `Tests (CPU)` 867 s (2,842 passed, 5
skipped, 8 deselected), `Mypy` 57 s, the other steps 1-5 s each. `Ruff` and `Black` failed, on
two files another task was editing at that moment (`src/terser/flows/__init__.py`,
`src/terser/monitoring/canary_metrics.py`), not on committed code. This run was CPU-contended
and is not a timing of a clean tree; an earlier uncontended local run took about 163 s.

## Verdict

The duration criterion is met on real GitHub runs (median 5.0 min for green runs, 10.7 min for
full-length runs of the current workflow, slowest 14.3 min), and the workflow covers every item
on the row's checklist (smoke training implicitly). It is not yet a working CI: the current
workflow has no green run on GitHub, because of black version drift and then the account's
billing block. Pinning the formatter and lifting the billing block are the two remaining steps.
