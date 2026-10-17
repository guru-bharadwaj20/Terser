# Continuous training flow (row 11.4.3) and fresh-clone orchestration (C11.1)

Date: 2026-10-17. Everything ran on CPU with `CUDA_VISIBLE_DEVICES=""`. No GPU, no vLLM server
and no LLM call was used. Code: `src/terser/flows/` (`ct_flow` in `flows.py`, its tasks in
`steps.py`, the deployment and its trigger in `registry.py`). Tests: `tests/test_flows.py`. How
to run it: `docs/orchestration.md`, section "Continuous training".

## Verdicts

| Item | Result | Evidence |
| :--- | :--- | :--- |
| **C11.1** `data → train_classical → evaluate → promote` unattended on a fresh clone (CPU) | **Pass** with the working-tree fixes below. The committed code at `2442057` failed (two fresh-clone bugs) | Run 3: four flows `Completed`, 585 s of flows, 591 s from `git clone` |
| **11.4.3** CT: on a drift alert or a new data batch, judge labels → retrain gate and tagger (CPU) → queue the rewriter retrain (GPU) → evaluate → register as Staging if the checkpoints pass | **One full cycle completed** on a fresh clone (new-batch trigger), 1,099 s, state `Completed`. Staging was refused by the promotion guard, as expected, because no rung passes C8.3 | Run 2 below, and 6 CT tests in the test harness |

## C11.1: fresh clone

### Procedure

1. `git clone ~/Terser /home/temp/terser_fresh` (committed HEAD).
2. The clone ran from the existing venv (`~/Terser/.venv/bin` first on `PATH`) with
   `PYTHONPATH=<clone>/src`, so that `terser` and `REPO_ROOT` resolve to the clone. This was
   asserted before each run. Its own Prefect home (`PREFECT_HOME=<clone>/.prefect_home`, a
   temporary server per command) and its own MLflow store (`<clone>/mlruns`) were used.
3. `dvc pull` from the DVC remote in the committed config (`remote.local.url =
   ../../terser-dvc-remote`; the clone needs no `config.local`).
4. `python -m terser.flows run data_flow --param label=false`, then `train_classical_flow`,
   `evaluate_flow` and `promote_flow`, one after another, unattended.

**LLM steps.** `data_flow`'s DVC stages `casualise` and `candidates` were up to date after
`dvc pull`. DVC skipped them ("didn't change, skipping") and Prefect returned the LLM tasks as
`Cached`. No server was needed. The `respond` / `judge` steps cannot run on a fresh clone: the
LLM response cache (`.cache/llm_responses`, 4.2 GB) is gitignored and not DVC-tracked, and
`data/interim/responses/*.jsonl` is not tracked either. The supported way to skip them is
`data_flow(label=false)`, which keeps the DVC-tracked `data/processed/labels.jsonl`. With
`label=true`, a fresh clone needs a vLLM server (GPU) or a copy of the response cache.

### Runs

| Run | Code | `data_flow` | `train_classical_flow` | `evaluate_flow` | `promote_flow` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | HEAD `2442057` as committed | Completed, 24.7 s | **Failed**, 767.6 s (3 attempts): `PermissionError: 'data/processed/pair_features.parquet'` | Completed, 223.0 s | Completed, 7.3 s, but `register` was **skipped** ("didn't change") and left the MLflow registry empty |
| 2 | HEAD `132cd2a` + fixes | Completed, 35.6 s | **Killed by my 3,000 s timeout**: `build_features` did not finish (see note) | Completed, 749.4 s | Completed, 65.3 s, `register --force` ran |
| **3** | HEAD `22fb4c2` + fixes | **Completed, 13.6 s** | **Completed, 305.2 s** | **Completed, 243.3 s** | **Completed, 23.0 s** (`register --force`, guard refused) |

Run 3 wall time: setup (clone, overlay, `dvc pull`, fallback pull) 6.1 s; flows 585.1 s; 591.3 s
from `git clone` to the end of `promote_flow`. Each flow exited 0. The alert log is empty: no
task or flow failed.

Run 3 details:

- `data_flow`: six stages skipped as up to date; `casualise` and `candidates` came back `Cached`.
- `train_classical_flow`: `dvc unprotect` of the three outputs took 0.4 s. `build_features`
  wrote 14,732 pairs over 1,262 prompts (degraded rate 0.431) and 129,380 span rows. Then
  `train_fast_path` ran: gate AUC 0.699, sensitivity 0.974, specificity 0.333. `pair_features`
  and `span_features` came out byte-identical to the DVC versions (not listed by `dvc status`).
  `models/fast` shows as modified: the ONNX export is not byte-identical.
- `evaluate_flow`: the C3.3 best gate is Tree depth 8, AUC 0.761, specificity 0.402. The C8.3
  report covers sft05, sft15 and sft3b, and no rung passes. These are the committed numbers.
- `promote_flow`: `dvc repro --single-item --force register` rebuilt the registry: gate
  `production`, tagger `production`, rewriter `staging` (v3) plus 2 archived. The promotion
  guard refused Production for sft3b (preservation 0.747 < 0.92, every category under 0.90,
  C8.3 not passed), so `registry_promote` did not run. A refusal is a decision; the flow
  completed.

**Run 2 note.** In run 2 the standalone `build_features` was still running when the timeout hit
at 50 min. In run 3 it took 4 min 23 s at about 1,400 % CPU, and inside the run 2 CT cycle,
minutes later on the same clone, it took 10 min 46 s. Two things overlapped with the stalled
attempt: the Prefect temporary server logged `sqlite3.OperationalError: database is locked`
(telemetry insert) at startup, and other CPU jobs were running on the machine. I could not
reproduce the stall, so its cause is open. Runs 1 and 3 and the CT cycle all completed this
step.

### Bugs found on a fresh clone and fixed (working tree, not committed)

1. **DVC-protected outputs.** With `cache.type: hardlink`, DVC checks outputs out as read-only
   hardlinks into its cache. `train_classical_flow` rewrites `pair_features.parquet`,
   `span_features.parquet` and `models/fast`, and failed with `PermissionError`; `data_flow`'s
   `judge` step would fail the same way on `labels.jsonl`. Fix: a `dvc unprotect` step runs
   before those writes (`unprotect_outputs`, `unprotect:labels`), covering only paths that
   exist.
2. **Empty registry on a fresh clone.** `mlruns/` is gitignored, but the `register` stage's
   only output (`reports/p11_registry.md`, `cache: false`) is committed. `dvc.lock` therefore
   calls the stage up to date, and `promote_flow` skipped it, leaving no registry. Its alias
   move would then have failed on a guard pass. Fix: when the registry does not know the
   component (`steps.registry_has`), `promote_flow` runs `dvc repro --single-item --force
   register`. Test: `test_promote_flow_forces_register_on_empty_registry`.

### Data gap not fixed: the shared DVC remote is incomplete

`/home/temp/terser-dvc-remote` does not hold every output committed in `.dvc` files. In run 3 a
plain `dvc pull` printed "45 files fetched and 40 files added" and **exited 0**, yet five
committed outputs were still absent: `data/interim/rewrites/`,
`data/processed/{labels.jsonl, oracle.csv, pair_features.parquet, span_features.parquet}`.
In run 1 `dvc pull` reported `CheckoutError` for them. Not on the remote at all:
`data/raw/*`, `data/processed/real_slice`, `labels_judge2.jsonl`, `p4_pairs.parquet` and
`data/interim/p2_smoke_20261003` (raw data and the real slice are not in `~/Terser/.dvc/cache`
either).

The objects needed were in `~/Terser/.dvc/cache`. Pushing them to the shared remote is a
shared-resource write, and the permission policy refused it in this session. The fresh clones
therefore pulled the missing targets read-only from a second remote, set in the clone's own
`.dvc/config.local` only:

```bash
dvc remote add --local devcache /home/temp/Terser/.dvc/cache
dvc pull -r devcache data/interim/rewrites.dvc data/processed/labels.jsonl.dvc \
  data/processed/oracle.csv.dvc data/processed/pair_features.parquet.dvc \
  data/processed/span_features.parquet.dvc
```

The objects are copied, not linked: the clone's cache files have their own inodes (checked).
Nothing under `~/Terser` was written. **To make C11.1 hold for a clone on another machine**, the
owner should run, once from `~/Terser`:

```bash
dvc push data/interim/rewrites.dvc data/processed/labels.jsonl.dvc data/processed/oracle.csv.dvc \
  data/processed/pair_features.parquet.dvc data/processed/span_features.parquet.dvc \
  data/processed/labels_judge2.jsonl.dvc data/processed/p4_pairs.parquet.dvc
```

`data/raw/*` and `real_slice` need `dvc commit` / `dvc push` from a workspace that still has
them. Only the `collect` stage (stale by design) and the `judge` references depend on those.

## Row 11.4.3: `ct_flow`

### Design

`ct_flow` composes the existing flows as Prefect sub-flows, in the order the row gives. Each step
appears in the plan with a `<flow>/<step>` prefix.

| # | Step | What runs |
| :--- | :--- | :--- |
| 0 | trigger | `trigger="check"`: tokenizer and input drift (`drift_check`, `input_drift_check`; the input-drift JSON verdict) when `check_drift`, then `new_data_check`: `dvc status --json <data stages>` is not empty, or the content digest of `batch=` differs from the last cycle's. `drift`, `batch` or `force` skip the check. With no trigger the run completes with "nothing to do" |
| 1 | judge labels | `data_flow(stages, label, pairs)` (`respond` + `judge` cached LLM steps; `label=false` on a CPU-only box) |
| 2 | retrain gate and tagger (CPU) | `train_classical_flow()` |
| 3 | queue the rewriter retrain (GPU) | `train_rewriter_flow("sft_qwen3b_lora", "ct-sft-qwen3b-lora-<YYYYMMDD>", 20 GB, 3 h)`: CPU smoke test, spec validated against the whole queue, written to `gpu_queue/`. `gpu_worker` is a handoff and is never executed |
| 4 | evaluate | `evaluate_flow()` (C3.3, C8.3 validation reports) |
| 5 | register as Staging if the checkpoints pass | `promote_flow(stage="Staging", from_alias="latest")`: the existing row 11.4.4 `promotion_guard` (C8.3 required, preservation floors, no regression against `champion_rung`) runs first, and `registry_promote` moves the newest rewriter version to Staging only if the guard allows it |
| 6 | record | `record_ct_cycle` writes `logs/ct_state.json`: trigger, reasons, batch digests, the queued job, the Staging decision |

The rewriter job queued in a cycle is trained by the GPU worker afterwards, so the next cycle
evaluates it. A guard refusal is recorded and the run still completes. A failing sub-flow fails
`ct_flow`, and the existing failure hook notifies.

**Deployment.** `ct_flow` is the seventh deployment, next to the existing six. It has two starts:

- the automation `ct-on-drift-alert`: any `prefect.flow-run.Failed` event of a `drift_flow`
  run starts `ct_flow` with `{"trigger": "drift"}`. A drift alert fails `drift_flow` by design
  (row 11.5.2 / 11.5.3).
- a daily cron at 04:00 with `check_drift=false`: it looks for new data only, since drift has its
  own trigger. Running the drift tasks inside `ct_flow` rather than the `drift_flow` sub-flow
  keeps a check-mode run from emitting a `drift_flow` failure event and starting a second cycle.

Registration in an isolated Prefect API (`prefect_test_harness`):

```text
data_flow/data_flow                           on demand    terser.flows.flows.data_flow  tags=terser, row-11.6
train_classical_flow/train_classical_flow     on demand    terser.flows.flows.train_classical_flow  tags=terser, row-11.6
train_rewriter_flow/train_rewriter_flow       on demand    terser.flows.flows.train_rewriter_flow  tags=terser, row-11.6, gpu_queue
evaluate_flow/evaluate_flow                   on demand    terser.flows.flows.evaluate_flow  tags=terser, row-11.6
promote_flow/promote_flow                     on demand    terser.flows.flows.promote_flow  tags=terser, row-11.6
drift_flow/drift_flow                         0 3 * * *    terser.flows.flows.drift_flow  tags=terser, row-11.6, monitoring
ct_flow/ct_flow                               0 4 * * *    terser.flows.flows.ct_flow  tags=terser, row-11.6, row-11.4.3, gpu_queue
7 deployment(s) registered
trigger ct-on-drift-alert: on prefect.flow-run.Failed [prefect.resource.name=drift_flow, prefect.resource.role=flow] -> ct_flow/ct_flow {"trigger": "drift"} enabled=True
```

### Evidence 1: one full cycle on a fresh clone (run 2)

`python -m terser.flows run ct_flow --param check_drift=false --param label=false --param
batch='"<new batch>.jsonl"'` (the batch is a copy of the val split, never seen by a cycle).
State `Completed` in 1,098.9 s:

```text
ct_flow:
  1. [trigger] new_data: dvc status --json casualise dedupe split label_subset candidates token_counts  (FIRED)
     NEW DATA new batch /home/temp/ct_logs/run2/batch_val.jsonl (sha256 b1a2fdf676fe)
  2. [trigger] ct_trigger: ct_flow check  (ok)
  3-8. data_flow/dvc:<6 stages>  (ok; up to date; casualise/candidates Cached)
  9. [shell] train_classical_flow/unprotect_outputs: dvc unprotect ...pair_features.parquet ...span_features.parquet models/fast  (ok)
  10. [shell] train_classical_flow/pair_features: python -m terser.phase3.build_features  (ok)   14,732 pairs
  11. [shell] train_classical_flow/fast_path_onnx: python -m terser.phase11.train_fast_path  (ok)
  12. [shell] train_rewriter_flow/cpu_smoke_test: python -m pytest -q tests/rewriter/test_train_tiny.py  (ok)
  13. [gpu_queue] train_rewriter_flow/gpu_queue_submit: python -m terser.gpu_queue validate --dir gpu_queue  (ok)
  14. [handoff] train_rewriter_flow/gpu_worker: python -m terser.gpu_queue run --dir gpu_queue  (planned)
  15. [shell] evaluate_flow/gate_checkpoint  (ok)   Tree depth 8: AUC 0.761, spec 0.402
  16. [shell] evaluate_flow/rewriter_val_report  (ok)   c8_3 false for sft05 / sft15 / sft3b
  17. [shell] promote_flow/register: dvc repro --single-item register  (ok)   ran: models/fast changed
  18. [python] promote_flow/promotion_guard: terser.pipelines.promotion.decide sft3b -  (REFUSED)
     refused: preservation 0.747 < floor 0.92; code preservation 0.578 < floor 0.9; ... checkpoints not passed: ['C8.3']
  19. [python] record_ct_cycle: terser.flows.steps.record_ct_cycle logs/ct_state.json  (ok)
```

The queued job (written to the clone's `gpu_queue/`, not run):

```yaml
id: ct-sft-qwen3b-lora-20261017
phase: 8
depends_on: [C7.1, C2.3]
command: python -m terser.models.rewriter.train experiment=sft_qwen3b_lora hardware=rtx4500ada
est_vram_gb: 20.0
est_hours: 3.0
exclusive: true
priority: 2
cpu_smoke_test: tests/rewriter/test_train_tiny.py
outputs: [models/rewriter/ct-sft-qwen3b-lora-20261017/, mlruns/]
```

`logs/ct_state.json` after the cycle: `trigger: check`, the batch reason, the batch sha256, the
`rewriter_job` path, and `staging: "refused: preservation 0.747 < floor 0.92; ... checkpoints not
passed: ['C8.3']"`. The same command run again with the same batch completed in 7.6 s with
`new_data (ok) no new data` and `ct_trigger: no drift alert and no new data: nothing to do`. No
second job was queued and no retraining happened.

### Evidence 2: test harness (trigger paths and Staging logic)

The tests run against `prefect_test_harness`. Shell, LLM, input-drift subprocess, `dvc status`
and registry calls are faked; the real Prefect tasks, sub-flows, GPU-queue validation and
promotion guard run.

| Test | Shows |
| :--- | :--- |
| `test_flows_registered` | 7 deployments; `ct_flow` cron 04:00; the `ct-on-drift-alert` automation (`prefect.flow-run.Failed`, `drift_flow`, `{"trigger": "drift"}`); re-applying is idempotent |
| `test_every_flow_plans_without_running[ct_flow]` | 25-step dry-run plan; only the read-only guard executes; `gpu_queue/` and `logs/ct_state.json` untouched |
| `test_ct_flow_no_trigger_does_nothing` | no drift and no new data: drift, new-data, trigger records only; nothing trained, queued or promoted |
| `test_ct_flow_drift_alert_runs_full_cycle_and_registers_staging` | input-drift verdict `alert: true` → the cycle in order (judge LLM steps, classical retrain, smoke test, gate C3.3, C8.3 report), job written to the queue and not run, guard **passes** on a C8.3-passing candidate → `promote("terser-rewriter", <latest>, "Staging")`; no failure alert |
| `test_ct_flow_guard_refuses_staging` | `trigger="drift"` (the automation's parameters) with a candidate failing C8.3: no `registry_promote`, the refusal reasons are recorded, the run is `Completed` |
| `test_ct_flow_new_batch_triggers_once` | a new batch fires; the same batch does not; a changed batch fires again; a DVC data-stage change (`dvc status`) fires and rebuilds that stage |
| `test_ct_flow_rejects_unknown_trigger`, `test_cli_prints_fired_trigger` | an invalid trigger fails and notifies; the CLI prints `(FIRED)` for a firing trigger |
| `test_promote_flow_forces_register_on_empty_registry` | fresh-clone fix 2 |

The promotion guard itself is unchanged; row 11.4.4's guard tests still cover it.

## Limitations

- The `register` stage (row 11.2.2, `terser.phase11.register`) re-registers every component with
  its fixed stage walk, independent of CT. When `models/fast` changes, gate and tagger go to
  Production and the best rewriter rung to Staging. CT's own alias move (`from_alias="latest"`)
  is guarded, but that pre-existing walk is not. Gating the gate and tagger needs a guard summary
  for them (`EvalSummary.gate_auc` exists) and a change to `register.py`; not done here.
- `label=true` (re-judging) needs the LLM response cache or a vLLM server. The CT cycle above used
  `label=false`. The test harness covers the judge path with faked LLM steps.
- A real drift-alert run (bge-small embeddings over the request window) was not executed here.
  The drift trigger path is shown by the registered automation and the drift tests.
  `drift_flow` itself is evidenced in `reports/p11_flows.md`.
- After a cycle, rewritten DVC outputs (`models/fast`) and `dvc.lock` (the `register` stage;
  `autostage` stages it) differ from git. Committing them is the owner's decision.
