# Orchestration: Prefect flows

Plan rows 11.6.1 (Prefect flows) and 11.6.2 (reliability). Code:
[`src/terser/flows/`](../src/terser/flows/). Tests: `tests/test_flows.py`. Evidence:
[`reports/p11_flows.md`](../reports/p11_flows.md).

```bash
pip install -e ".[orchestrate]"     # Prefect 3
```

## The flows

Each flow is a short, ordered list of named tasks over entry points that already exist, so any
run can be repeated by hand from its plan.

| Flow | Steps | Schedule |
| :--- | :--- | :--- |
| `data_flow` | `dvc repro --single-item <stage>` for `casualise`, `dedupe`, `split`, `label_subset`, `candidates`, `token_counts`; then `respond` per target and `judge` (`terser.phase2.respond`, `terser.phase2.label_stage`); then `terser.data.validation` if `pairs=` is given | on demand |
| `train_classical_flow` | `terser.phase3.build_features`, `terser.phase11.train_fast_path` (ONNX gate and tagger) | on demand |
| `train_rewriter_flow` | CPU smoke test (`pytest tests/rewriter/test_train_tiny.py`), then a job spec validated against the whole queue and written to `gpu_queue/<id>.yaml`. The GPU machine trains it with `python -m terser.gpu_queue run`; the flow never trains | on demand |
| `evaluate_flow` | `terser.phase3.gate_checkpoint` (C3.3), `terser.models.rewriter.val_report` (C8.3) | on demand |
| `promote_flow` | `dvc repro --single-item register` (`terser.phase11.register`), the row 11.4.4 promotion guard on `reports/p8_rewriter_val.json`, then `terser.tracking.promote` only if the guard allows it | on demand |
| `drift_flow` | tokenizer drift (row 11.5.3), then input drift (row 11.5.2, `python -m terser.monitoring.input_drift --name <date> [--current <jsonl>]`); any alert fails the run | daily 03:00 |

`collect` is left out of `data_flow` by default because it is stale by design (see
`reports/p2_dvc_repro.md`). Pass `stages=[...]` to choose stages. Steps and commands of
`train_classical_flow` and `evaluate_flow` can be overridden with `steps={name: argv}`.

## Commands

Run these from the repo root:

```bash
python -m terser.flows list                     # flows, entrypoints, schedules, parameters
python -m terser.flows plan                     # dry run of every flow: the command plan
python -m terser.flows plan data_flow
python -m terser.flows run evaluate_flow        # run one flow once
python -m terser.flows run drift_flow --param current='"logs/requests.jsonl"'
python -m terser.flows run train_rewriter_flow --dry-run --param experiment='"sft_tiny"'
python -m terser.flows deploy                   # register the six deployments
python -m terser.flows serve                    # long-running: execute deployments on schedule
python -m terser.flows demo --out reports/p11_flows.md
```

`--param` values are JSON. A bare word that is not valid JSON is taken as a string.

**Dry run.** Every flow takes `dry_run`. In a dry run the tasks still run through Prefect but
only record their commands; the flow returns them as `StepResult` records. Two cheap, read-only
checks also run in a dry run: the GPU-queue spec is validated in a scratch copy of the queue, and
the promotion guard is evaluated.

**Registration.** `deploy` applies one deployment per flow to the configured Prefect API.
Deployments use module entrypoints (`terser.flows.flows:<flow>`) and the tags `terser` and
`row-11.6`. `train_rewriter_flow` is also tagged `gpu_queue`, and `drift_flow` is tagged
`monitoring`. If `PREFECT_API_URL` is unset, Prefect starts a temporary local server for the
command and stops it on exit. The deployments stay in `~/.prefect/prefect.db`, and
`prefect deployment ls` lists them. To run them on their schedules, keep
`python -m terser.flows serve` running, or point `PREFECT_API_URL` at a `prefect server start`
instance.

## Reliability (row 11.6.2)

| Task | Retries | Delay | Cache |
| :--- | ---: | :--- | :--- |
| `shell` (dvc stages, CLI modules) | 2 | 10 s | none |
| `llm` (`casualise`, `candidates`, `respond`, `judge`) | 3 | 10, 30, 90 s | input-hash key, persisted |
| `gpu_queue_submit`, `promotion_guard` | 0 | - | none (deterministic) |
| `tokenizer_drift`, `input_drift` | 1 | 10 s | none |
| `registry_promote` | 2 | 10 s | none |

- **Cached LLM steps.** The cache key is a SHA-256 of the step name, the command and the content
  of every input file or directory: the DVC stage deps plus `params.yaml`, or the prompts,
  candidates and responses plus the calling code. When an LLM step's inputs have not changed, a
  re-run returns the stored result in state `Cached` and starts nothing. Any change to an input
  runs the step again. Results are stored under `PREFECT_LOCAL_STORAGE_PATH` (default
  `~/.prefect/storage`). Set `PREFECT_TASKS_REFRESH_CACHE=true` to force a re-run. Inside a step,
  individual requests are also disk-cached by `terser.llm`.
- **Failure notifications.** Every task has the `on_failure` hook `notify_failure`, and every
  flow has `notify_flow_failure`. A hook does three things. It appends a JSON record (kind, name,
  run, attempts, state, message) to `logs/flow_alerts.jsonl`, or to the path in
  `TERSER_ALERT_LOG`. It logs `CRITICAL` on `terser.alerts`. It POSTs the record to
  `TERSER_ALERT_WEBHOOK` only when that variable is set; when it is unset, no network call is
  made. Hooks never raise.
- **Drift alerts are results, not crashes.** An input-drift alert (exit 1 with a JSON verdict)
  or a tokenizer-drift alert is not retried. It fails the flow run, which notifies. A crash of
  the input-drift CLI (any other exit, or no verdict) is retried once.
- **Failure injection.** `terser.flows.reliability.flaky_command(counter, n)` fails its first
  `n` runs and then succeeds. The tests and the demo use it to show recovery through retries
  and notification after a permanent failure.
